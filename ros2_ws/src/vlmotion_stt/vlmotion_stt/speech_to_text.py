#!/usr/bin/env python3
"""Transcribe complete ReSpeaker utterances with a local Whisper model."""

import math
import queue
import threading

from audio_common_msgs.msg import AudioData
from faster_whisper import WhisperModel
import numpy as np
from opencc import OpenCC
import rclpy
from rclpy.node import Node
from speech_recognition_msgs.msg import SpeechRecognitionCandidates


DEFAULT_HOTWORDS = (
    "Stretch, Stretch 3, gripper, D405, D435i, camera, robot, elevator, "
    "move left, move right, move forward, move backward, pick up, put down, "
    "夾爪, 相機, 機器人, 電梯, 往左, 往右, 前進, 後退, 抬高, 放下, 抓取"
)


def normalize_language(language):
    value = str(language or "").strip().lower().replace("_", "-")
    if value in ("", "auto", "none"):
        return None
    if value.startswith("zh"):
        return "zh"
    if value.startswith("en"):
        return "en"
    return value.split("-", 1)[0]


class SpeechToText(Node):
    """Receive utterance-sized PCM messages and publish recognition results."""

    def __init__(self):
        super().__init__("vlmotion_speech_to_text")

        self.sample_rate = int(self.declare_parameter("sample_rate", 16000).value)
        self.sample_width = int(self.declare_parameter("sample_width", 2).value)
        self.language = str(self.declare_parameter("language", "auto").value)
        self.beam_size = int(self.declare_parameter("beam_size", 5).value)
        self.hotwords = str(
            self.declare_parameter("hotwords", DEFAULT_HOTWORDS).value
        )
        self.traditional_chinese = bool(
            self.declare_parameter("traditional_chinese", True).value
        )

        model_path = str(
            self.declare_parameter(
                "model_path", "small"
            ).value
        )
        model_dir = str(
            self.declare_parameter(
                "model_dir", "/models/faster-whisper"
            ).value
        )
        cpu_threads = int(self.declare_parameter("cpu_threads", 8).value)
        input_topic = str(
            self.declare_parameter("input_topic", "/speech_audio").value
        )
        output_topic = str(
            self.declare_parameter(
                "output_topic", "/vlmotion/speech_to_text"
            ).value
        )

        if self.sample_width not in (1, 2, 4):
            raise ValueError("sample_width must be 1, 2, or 4 bytes")

        self.get_logger().info(
            "Loading faster-whisper model=%s cache=%s (CPU INT8, threads=%d)"
            % (model_path, model_dir, cpu_threads)
        )
        self.model = WhisperModel(
            model_path,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
            download_root=model_dir,
            local_files_only=True,
        )
        self.opencc = OpenCC("s2twp") if self.traditional_chinese else None

        self.text_pub = self.create_publisher(
            SpeechRecognitionCandidates, output_topic, 10
        )
        self.audio_sub = self.create_subscription(
            AudioData, input_topic, self.audio_callback, 10
        )

        self._jobs = queue.Queue(maxsize=2)
        self._stopping = threading.Event()
        self._worker = threading.Thread(
            target=self._recognition_worker,
            name="vlmotion-whisper-worker",
            daemon=True,
        )
        self._worker.start()

        self.get_logger().info(
            "Offline STT ready: %s (%d Hz mono PCM%d) -> %s"
            % (input_topic, self.sample_rate, self.sample_width * 8, output_topic)
        )

    def audio_callback(self, msg):
        raw_audio = bytes(msg.data)
        if not raw_audio:
            return

        duration = len(raw_audio) / float(self.sample_rate * self.sample_width)
        if duration < 0.15:
            self.get_logger().debug("Ignoring %.2f second audio chunk" % duration)
            return
        try:
            self._jobs.put_nowait(raw_audio)
            self.get_logger().info(
                "Queued %.2f seconds of speech for recognition" % duration
            )
        except queue.Full:
            self.get_logger().warning(
                "Recognition queue is full; dropping one utterance"
            )

    def _pcm_to_float32(self, raw_audio):
        if self.sample_width == 1:
            samples = np.frombuffer(raw_audio, dtype=np.uint8)
            return (samples.astype(np.float32) - 128.0) / 128.0
        if self.sample_width == 2:
            samples = np.frombuffer(raw_audio, dtype="<i2")
            return samples.astype(np.float32) / 32768.0
        samples = np.frombuffer(raw_audio, dtype="<i4")
        return samples.astype(np.float32) / 2147483648.0

    @staticmethod
    def _confidence(segments):
        weighted_sum = 0.0
        total_duration = 0.0
        for segment in segments:
            duration = max(float(segment.end - segment.start), 0.01)
            probability = math.exp(min(float(segment.avg_logprob), 0.0))
            weighted_sum += probability * duration
            total_duration += duration
        if total_duration == 0.0:
            return 0.0
        return max(0.0, min(1.0, weighted_sum / total_duration))

    def _recognition_worker(self):
        while not self._stopping.is_set():
            try:
                raw_audio = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                segments_iter, info = self.model.transcribe(
                    self._pcm_to_float32(raw_audio),
                    language=normalize_language(self.language),
                    task="transcribe",
                    beam_size=self.beam_size,
                    vad_filter=True,
                    multilingual=True,
                    hotwords=self.hotwords or None,
                    condition_on_previous_text=False,
                )
                segments = list(segments_iter)
                text = "".join(segment.text for segment in segments).strip()
                if not text:
                    self.get_logger().info("No speech recognized")
                    continue
                if self.opencc is not None:
                    text = self.opencc.convert(text)
                confidence = self._confidence(segments)
                self.text_pub.publish(
                    SpeechRecognitionCandidates(
                        transcript=[text], confidence=[confidence]
                    )
                )
                self.get_logger().info(
                    "Recognized [%s %.2f]: %s"
                    % (info.language, confidence, text)
                )
            except Exception as exc:
                self.get_logger().error("Speech recognition failed: %s" % exc)
            finally:
                self._jobs.task_done()

    def close(self):
        self._stopping.set()
        self._worker.join(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SpeechToText()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
