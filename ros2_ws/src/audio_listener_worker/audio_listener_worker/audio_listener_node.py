#!/usr/bin/env python3
"""
Audio Listener Worker Node
麥克風端節點：監聽 /start_listen 訊號並進行語音錄音 + 識別
"""

import sys
import os
import io
import wave
import threading
import ctypes
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from speech_recognition_msgs.msg import SpeechRecognitionCandidates


# ============================================================
# PyAudio 初始化（類似 white_point_gui.py 的做法）
# ============================================================
_alsa_handle = None
try:
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
        None,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
    )

    def _alsa_error_handler(filename, line, function, err, fmt):
        # Suppress ALSA warnings caused by missing default devices
        pass

    _alsa_handler = ERROR_HANDLER_FUNC(_alsa_error_handler)
    try:
        _alsa_handle = ctypes.cdll.LoadLibrary("libasound.so.2")
        _alsa_handle.snd_lib_error_set_handler(_alsa_handler)
    except OSError:
        try:
            _alsa_handle = ctypes.cdll.LoadLibrary("libasound.so.1")
            _alsa_handle.snd_lib_error_set_handler(_alsa_handler)
        except OSError:
            _alsa_handle = None
except Exception:
    _alsa_handle = None

# PyAudio 導入
try:
    import pyaudio
except Exception:
    pyaudio = None

try:
    if _alsa_handle is not None:
        _alsa_handle.snd_lib_error_set_handler(None)
except Exception:
    pass

# Wit.ai 導入
try:
    from wit import Wit
except Exception:
    Wit = None

try:
    import websocket
except Exception:
    websocket = None


# ============================================================
# 配置變數
# ============================================================
WIT_TOKEN_PATH = os.getenv("WIT_TOKEN_PATH", "/workspace/wit_token.txt")
DEFAULT_RECORD_DURATION = 4.0  # 秒
DEFAULT_SAMPLE_RATE = 16000    # Hz
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_WS_URL = "ws://10.0.0.3:9090"
DEFAULT_WS_START_LISTEN_TOPIC = "/start_listen"
DEFAULT_WS_SPEECH_TO_TEXT_TOPIC = "/speech_to_text"
DEFAULT_WS_RECONNECT_SEC = 2.0
DEFAULT_USE_ROS_TOPIC_TRIGGER = False
DEFAULT_USE_WS_TRIGGER = True
DEFAULT_USE_ROS_TOPIC_RESULT_PUBLISH = False
DEFAULT_USE_WS_RESULT_PUBLISH = True


class AudioListenerWorker(Node):
    """
    遠端麥克風節點：
    1. 訂閱 /start_listen (String) - 接收開始錄音訊號
    2. 進行本地語音錄音
    3. 使用 Wit.ai 進行語音識別
    4. 發布結果到 /speech_to_text (SpeechRecognitionCandidates)
    5. 可選：作為 rosbridge websocket client 接收/回送訊息
    """

    def __init__(self):
        super().__init__('audio_listener_worker')
        
        # 參數設定
        self.declare_parameter('record_duration', DEFAULT_RECORD_DURATION)
        self.declare_parameter('sample_rate', DEFAULT_SAMPLE_RATE)
        self.declare_parameter('chunk_size', DEFAULT_CHUNK_SIZE)
        self.declare_parameter('ws_client_enabled', False)
        self.declare_parameter('ws_url', DEFAULT_WS_URL)
        self.declare_parameter('ws_start_listen_topic', DEFAULT_WS_START_LISTEN_TOPIC)
        self.declare_parameter('ws_speech_to_text_topic', DEFAULT_WS_SPEECH_TO_TEXT_TOPIC)
        self.declare_parameter('ws_reconnect_sec', DEFAULT_WS_RECONNECT_SEC)
        self.declare_parameter('use_ros_topic_trigger', DEFAULT_USE_ROS_TOPIC_TRIGGER)
        self.declare_parameter('use_ws_trigger', DEFAULT_USE_WS_TRIGGER)
        self.declare_parameter('use_ros_topic_result_publish', DEFAULT_USE_ROS_TOPIC_RESULT_PUBLISH)
        self.declare_parameter('use_ws_result_publish', DEFAULT_USE_WS_RESULT_PUBLISH)
        
        self.record_duration = self.get_parameter('record_duration').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.chunk_size = self.get_parameter('chunk_size').value
        self.ws_client_enabled = self.get_parameter('ws_client_enabled').value
        self.ws_url = self.get_parameter('ws_url').value
        self.ws_start_listen_topic = self.get_parameter('ws_start_listen_topic').value
        self.ws_speech_to_text_topic = self.get_parameter('ws_speech_to_text_topic').value
        self.ws_reconnect_sec = float(self.get_parameter('ws_reconnect_sec').value)
        self.use_ros_topic_trigger = self.get_parameter('use_ros_topic_trigger').value
        self.use_ws_trigger = self.get_parameter('use_ws_trigger').value
        self.use_ros_topic_result_publish = self.get_parameter('use_ros_topic_result_publish').value
        self.use_ws_result_publish = self.get_parameter('use_ws_result_publish').value
        
        # 初始化 Wit 客戶端
        self.wit_client = self._init_wit_client()
        
        # 訂閱 /start_listen 訊號（可選）
        self.start_listen_sub = None
        if self.use_ros_topic_trigger:
            self.start_listen_sub = self.create_subscription(
                String,
                '/start_listen',
                self.start_listen_callback,
                10
            )
        
        # 發布 /speech_to_text 結果（可選）
        self.speech_to_text_pub = None
        if self.use_ros_topic_result_publish:
            self.speech_to_text_pub = self.create_publisher(
                SpeechRecognitionCandidates,
                '/speech_to_text',
                10
            )
        
        # 内部狀態
        self.is_recording = False
        self.recording_thread = None
        self.recording_lock = threading.Lock()

        # WebSocket client 狀態
        self.ws = None
        self.ws_lock = threading.Lock()
        self.ws_stop_event = threading.Event()
        self.ws_thread = None
        
        self.get_logger().info("Audio Listener Worker initialized")
        self.get_logger().info(f"Wit.ai support: {self.wit_client is not None}")
        self.get_logger().info(f"PyAudio support: {pyaudio is not None}")
        self.get_logger().info(f"WebSocket client enabled: {self.ws_client_enabled}")
        self.get_logger().info(f"ROS topic trigger enabled: {self.use_ros_topic_trigger}")
        self.get_logger().info(f"WebSocket trigger enabled: {self.use_ws_trigger}")
        self.get_logger().info(f"ROS topic result publish enabled: {self.use_ros_topic_result_publish}")
        self.get_logger().info(f"WebSocket result publish enabled: {self.use_ws_result_publish}")

        if self.ws_client_enabled:
            self._start_ws_client_thread()

    def destroy_node(self):
        """Ensure websocket thread exits cleanly before node shutdown."""
        self._stop_ws_client_thread()
        return super().destroy_node()

    def _init_wit_client(self):
        """初始化 Wit.ai 客戶端"""
        if Wit is None:
            self.get_logger().warn("Wit package not installed, speech recognition disabled")
            return None
        
        token_path = WIT_TOKEN_PATH
        try:
            with open(token_path, "r") as token_file:
                token = token_file.read().strip()
            if not token:
                self.get_logger().error(f"Empty token in {token_path}")
                return None
            
            client = Wit(token)
            self.get_logger().info(f"Wit.ai client initialized successfully from {token_path}")
            return client
        except FileNotFoundError:
            self.get_logger().error(f"Wit token file '{token_path}' not found")
        except Exception as exc:
            self.get_logger().error(f"Failed to initialize Wit client: {exc}")
        
        return None

    def start_listen_callback(self, msg: String):
        """
        監聽 /start_listen 訊號，觸發錄音
        """
        signal = msg.data.strip().lower()
        self.get_logger().info(f"Received start_listen signal: {signal}")

        self._start_recording(source="ros_topic")

    def _start_recording(self, source="unknown"):
        """Start recording in a worker thread if not already recording."""
        with self.recording_lock:
            if self.is_recording:
                self.get_logger().warn(f"Already recording, ignoring new request from {source}")
                return
            self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_and_recognize)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        self.get_logger().info(f"Recording triggered from {source}")

    def _start_ws_client_thread(self):
        """Start websocket receiver loop in background."""
        if websocket is None:
            self.get_logger().error("websocket-client is not installed; disable ws_client_enabled or install dependency")
            return
        self.ws_thread = threading.Thread(target=self._ws_loop, daemon=True)
        self.ws_thread.start()
        self.get_logger().info(f"WebSocket thread started, target={self.ws_url}")

    def _stop_ws_client_thread(self):
        """Stop websocket loop and close active connection."""
        self.ws_stop_event.set()
        with self.ws_lock:
            if self.ws is not None:
                try:
                    self.ws.close()
                except Exception:
                    pass
                self.ws = None

    def _ws_loop(self):
        """Maintain rosbridge websocket connection and handle incoming messages."""
        while not self.ws_stop_event.is_set():
            try:
                self.get_logger().info(f"Connecting to rosbridge websocket: {self.ws_url}")
                ws_conn = websocket.create_connection(self.ws_url, timeout=5)
                ws_conn.settimeout(1.0)

                with self.ws_lock:
                    self.ws = ws_conn

                if self.use_ws_trigger:
                    subscribe_msg = {
                        "op": "subscribe",
                        "topic": self.ws_start_listen_topic,
                        "type": "std_msgs/msg/String",
                    }
                    ws_conn.send(json.dumps(subscribe_msg))
                    self.get_logger().info(
                        f"Subscribed to websocket topic: {self.ws_start_listen_topic}"
                    )

                while not self.ws_stop_event.is_set():
                    try:
                        raw = ws_conn.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not raw:
                        continue
                    self._handle_ws_message(raw)

            except Exception as exc:
                if not self.ws_stop_event.is_set():
                    self.get_logger().warn(f"WebSocket disconnected: {exc}")
            finally:
                with self.ws_lock:
                    if self.ws is not None:
                        try:
                            self.ws.close()
                        except Exception:
                            pass
                    self.ws = None

            if not self.ws_stop_event.is_set():
                time.sleep(max(0.1, self.ws_reconnect_sec))

    def _handle_ws_message(self, raw):
        """Process rosbridge messages and trigger recording when requested."""
        try:
            payload = json.loads(raw)
        except Exception:
            return

        if payload.get("op") != "publish":
            return
        if payload.get("topic") != self.ws_start_listen_topic:
            return

        msg = payload.get("msg", {})
        signal = str(msg.get("data", "")).strip().lower()
        self.get_logger().info(f"Received websocket start signal: {signal}")
        self._start_recording(source="websocket")

    def _record_and_recognize(self):
        """
        錄音並進行語音識別
        """
        try:
            self.get_logger().info("Starting audio recording...")
            
            # 1. 錄音
            audio_bytes = self._record_audio()
            if not audio_bytes:
                self.get_logger().error("Failed to record audio")
                return
            
            self.get_logger().info(f"Recording completed. Audio size: {len(audio_bytes)} bytes")
            
            # 2. 語音識別
            recognized_text = self._recognize_speech(audio_bytes)
            
            # 3. 發布結果
            self._publish_result(recognized_text)
            
        except Exception as e:
            self.get_logger().error(f"Error in record_and_recognize: {e}")
        finally:
            with self.recording_lock:
                self.is_recording = False

    def _record_audio(self, duration=None, sample_rate=None, chunk_size=None):
        """
        使用 PyAudio 進行音訊錄製
        """
        if pyaudio is None:
            self.get_logger().error("PyAudio not available")
            return None
        
        duration = duration or self.record_duration
        sample_rate = sample_rate or self.sample_rate
        chunk_size = chunk_size or self.chunk_size
        
        audio = pyaudio.PyAudio()
        stream = None
        
        try:
            # 尋找合適的輸入裝置（優先USB設備，如麥克風）
            input_device_index = None
            for i in range(audio.get_device_count()):
                try:
                    device_info = audio.get_device_info_by_host_api_device_index(0, i)
                    if device_info.get("maxInputChannels", 0) > 0:
                        device_name = device_info.get("name", "").lower()
                        # 優先選擇 USB 設備
                        if "usb" in device_name or "mic" in device_name:
                            input_device_index = i
                            self.get_logger().info(
                                f"Using input device: {device_info.get('name')} (index: {i})"
                            )
                            break
                        if input_device_index is None:
                            input_device_index = i
                except Exception:
                    pass
            
            # 開啟音訊串流
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=chunk_size,
            )
            
            self.get_logger().info(
                f"Recording: {duration}s @ {sample_rate}Hz, "
                f"device_index={input_device_index}"
            )
            
            # 錄音
            frames = []
            frame_count = int(sample_rate / float(chunk_size) * float(duration))
            for _ in range(max(1, frame_count)):
                try:
                    data = stream.read(chunk_size, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    self.get_logger().warn(f"Error reading audio frame: {e}")
            
            # 儲存為 WAV 格式
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(b"".join(frames))
            
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as exc:
            self.get_logger().error(f"Audio recording failed: {exc}")
            return None
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            try:
                audio.terminate()
            except Exception:
                pass

    def _recognize_speech(self, audio_bytes):
        """
        使用 Wit.ai 進行語音識別
        """
        if not audio_bytes:
            self.get_logger().warn("No audio bytes to recognize")
            return ""
        
        if self.wit_client is None:
            self.get_logger().error("Wit.ai client not available")
            return ""
        
        try:
            audio_buffer = io.BytesIO(audio_bytes)
            audio_buffer.name = "voice.wav"
            audio_buffer.seek(0)
            
            self.get_logger().info("Sending audio to Wit.ai for recognition...")
            response = self.wit_client.speech(audio_buffer, {"Content-Type": "audio/wav"})
            
            if isinstance(response, dict):
                # 提取識別結果
                text = response.get("text", "").strip()
                if text:
                    self.get_logger().info(f"Recognized text: {text}")
                    return text
                else:
                    self.get_logger().warn("No text found in Wit response")
            
        except Exception as exc:
            self.get_logger().error(f"Wit.ai speech recognition failed: {exc}")
        
        return ""

    def _publish_result(self, text):
        """
        發布識別結果到 /speech_to_text
        """
        msg = SpeechRecognitionCandidates()
        msg.transcript = [text] if text else [""]
        msg.confidence = [1.0] if text else [0.0]

        if self.use_ros_topic_result_publish and self.speech_to_text_pub is not None:
            self.speech_to_text_pub.publish(msg)
            self.get_logger().info("Published speech recognition result to ROS topic: /speech_to_text")

        if self.use_ws_result_publish:
            self._publish_result_via_ws(msg)

    def _publish_result_via_ws(self, ros_msg):
        """Publish speech result back to rosbridge websocket topic."""
        if not self.ws_client_enabled:
            return

        with self.ws_lock:
            ws_conn = self.ws

        if ws_conn is None:
            return

        ws_msg = {
            "op": "publish",
            "topic": self.ws_speech_to_text_topic,
            "msg": {
                "transcript": list(ros_msg.transcript),
                "confidence": list(ros_msg.confidence),
            },
        }
        try:
            ws_conn.send(json.dumps(ws_msg))
            self.get_logger().info(
                f"Published speech result via websocket topic: {self.ws_speech_to_text_topic}"
            )
        except Exception as exc:
            self.get_logger().warn(f"Failed to publish websocket speech result: {exc}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = AudioListenerWorker()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
