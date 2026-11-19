#!/usr/bin/env python3
"""Offline Mandarin/English speech recognition for ReSpeaker audio."""

import math
import os

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node

from audio_common_msgs.msg import AudioData
from sound_play_msgs.action import SoundRequest as SoundRequestAction
from speech_recognition_msgs.msg import SpeechRecognitionCandidates

from faster_whisper import WhisperModel
from opencc import OpenCC


DEFAULT_HOTWORDS = (
    "Stretch, Stretch 3, gripper, D405, D435i, camera, robot, elevator, "
    "move left, move right, move forward, move backward, pick up, put down, "
    "夾爪, 相機, 機器人, 電梯, 往左, 往右, 前進, 後退, 抬高, 放下, 抓取"
)


class SpeechToText(Node):
    """Transcribe complete utterances from ``/speech_audio`` locally."""

    def __init__(self):
        super().__init__("speech_to_text")

        self.sample_rate = int(self.declare_parameter("sample_rate", 16000).value)
        self.sample_width = int(self.declare_parameter("sample_width", 2).value)

        # Use ``auto`` for Mandarin, English, and code-switched commands.
        self.language = str(self.declare_parameter("language", "auto").value)
        self.model_size = str(self.declare_parameter("model_size", "small").value)
        self.model_dir = os.path.expanduser(str(self.declare_parameter(
            "model_dir", "/workspace/models/faster-whisper").value))
        self.device = str(self.declare_parameter("device", "cpu").value)
        self.compute_type = str(self.declare_parameter("compute_type", "int8").value)
        self.cpu_threads = int(self.declare_parameter("cpu_threads", 8).value)
        self.beam_size = int(self.declare_parameter("beam_size", 5).value)
        self.vad_filter = bool(self.declare_parameter("vad_filter", True).value)
        self.multilingual = bool(self.declare_parameter("multilingual", True).value)
        self.hotwords = str(self.declare_parameter("hotwords", DEFAULT_HOTWORDS).value)
        self.traditional_chinese = bool(self.declare_parameter(
            "traditional_chinese", True).value)
        self.local_files_only = bool(self.declare_parameter(
            "local_files_only", True).value)

        self.self_cancellation = bool(self.declare_parameter(
            "self_cancellation", True).value)
        self.tts_tolerance = Duration(seconds=float(self.declare_parameter(
            "tts_tolerance", 1.0).value))

        os.makedirs(self.model_dir, exist_ok=True)
        self.get_logger().info(
            "Loading faster-whisper model=%s device=%s compute_type=%s threads=%d"
            % (self.model_size, self.device, self.compute_type, self.cpu_threads)
        )
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                download_root=self.model_dir,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to load the local speech model. Run "
                "bash docker/download_speech_model.sh first."
            ) from exc

        self.opencc = OpenCC("s2twp") if self.traditional_chinese else None

        self.tts_action = None
        self.last_tts = None
        self.is_canceling = False
        if self.self_cancellation:
            self.tts_action = ActionClient(self, SoundRequestAction, "sound_play")
            if self.tts_action.wait_for_server(timeout_sec=5.0):
                self.tts_timer = self.create_timer(0.1, self.tts_timer_cb)
            else:
                self.get_logger().warning(
                    "sound_play is unavailable; speech self-cancellation is disabled."
                )
                self.tts_action = None

        self.pub_speech = self.create_publisher(
            SpeechRecognitionCandidates, "speech_to_text", 1)
        self.sub_audio = self.create_subscription(
            AudioData, "speech_audio", self.audio_cb, 1)
        self.get_logger().info("Offline speech recognition is ready.")

    def tts_timer_cb(self):
        # The original cancellation hook is intentionally retained for future
        # sound_play status integration. ReSpeaker echo suppression still runs.
        return

    def _language_code(self):
        language = self.language.strip().lower().replace("_", "-")
        if language in ("", "auto", "none"):
            return None
        if language.startswith("zh"):
            return "zh"
        if language.startswith("en"):
            return "en"
        return language.split("-", 1)[0]

    def _pcm_to_float32(self, raw_audio):
        if self.sample_width == 1:
            samples = np.frombuffer(raw_audio, dtype=np.uint8)
            return (samples.astype(np.float32) - 128.0) / 128.0
        if self.sample_width == 2:
            samples = np.frombuffer(raw_audio, dtype="<i2")
            return samples.astype(np.float32) / 32768.0
        if self.sample_width == 4:
            samples = np.frombuffer(raw_audio, dtype="<i4")
            return samples.astype(np.float32) / 2147483648.0
        raise ValueError("Unsupported sample_width: %d" % self.sample_width)

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

    def audio_cb(self, msg):
        if self.is_canceling:
            self.get_logger().info("Speech input cancelled while robot is speaking.")
            return

        raw_audio = bytes(msg.data)
        if not raw_audio:
            return

        try:
            audio = self._pcm_to_float32(raw_audio)
            duration = len(audio) / float(self.sample_rate)
            self.get_logger().info("Transcribing %.2f seconds of audio..." % duration)

            generated_segments, info = self.model.transcribe(
                audio,
                language=self._language_code(),
                task="transcribe",
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                multilingual=self.multilingual,
                hotwords=self.hotwords or None,
                condition_on_previous_text=False,
            )
            segments = list(generated_segments)
            result = "".join(segment.text for segment in segments).strip()
            if not result:
                self.get_logger().warning("No speech was recognized.")
                return

            if self.opencc is not None:
                result = self.opencc.convert(result)

            confidence = self._confidence(segments)
            output = SpeechRecognitionCandidates(
                transcript=[result], confidence=[confidence])
            self.pub_speech.publish(output)
            self.get_logger().info(
                "Recognized [%s %.2f]: %s"
                % (info.language, confidence, result)
            )
        except Exception as exc:
            self.get_logger().error("Local speech recognition failed: %s" % exc)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SpeechToText()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
