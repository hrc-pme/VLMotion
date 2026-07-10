#!/usr/bin/env python3
"""
white_point_gui_standalone.py
─────────────────────────────
獨立測試用 GUI — 不需要機器人、相機、RViz 也能使用。

與原版 white_point_gui.py 的差異：
  ● 影像來源可選「從檔案開啟」（QFileDialog），不再依賴 ROS camera topic
  ● 仍然保留 ROS2 spin 以便透過 /white_point_pixel 與 /user_input topic
    跟其他節點溝通（若節點未啟動也不影響 GUI 運作）
  ● 不需要 stretch_driver / RealSense / RViz
  ● 可直接用 launch 檔或 ros2 run 啟動
"""

import sys
import os
import re
import io
import wave
import json
import ctypes
import subprocess
import argparse
import pathlib

# ── ALSA 警告壓制（必須在 pyaudio import 前）──────────────────────────────────
_alsa_handle = None
try:
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
        None, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
    )

    def _alsa_error_handler(filename, line, function, err, fmt):
        pass

    _alsa_handler = ERROR_HANDLER_FUNC(_alsa_error_handler)
    try:
        _alsa_handle = ctypes.cdll.LoadLibrary("libasound.so.2")
        _alsa_handle.snd_lib_error_set_handler(_alsa_handler)
    except OSError:
        try:
            _alsa_handle = ctypes.cdll.LoadLibrary("libasound.so")
            _alsa_handle.snd_lib_error_set_handler(_alsa_handler)
        except OSError:
            pass
except Exception:
    _alsa_handle = None

try:
    import pyaudio  # type: ignore
except Exception:
    pyaudio = None

try:
    if _alsa_handle is not None:
        _alsa_handle.snd_lib_error_set_handler(None)
except Exception:
    pass

try:
    from wit import Wit
except Exception:
    Wit = None

# ── Qt 環境清理（必須在所有 GUI import 前）────────────────────────────────────
def clean_qt_environment():
    for p in [
        '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms',
        '/usr/lib/x86_64-linux-gnu/qt/plugins/platforms',
        '/usr/lib/qt/plugins/platforms',
    ]:
        if os.path.isdir(p):
            os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', p)
            break
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    for key in ['QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH']:
        if key in os.environ:
            val = os.environ[key]
            filtered = ':'.join(
                p for p in val.split(':')
                if 'cv2' not in p and 'opencv' not in p.lower()
            )
            if filtered:
                os.environ[key] = filtered
            else:
                del os.environ[key]

clean_qt_environment()

# ── ROS2 ─────────────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge
import requests

# ── OpenCV（在 Qt 之前設定環境）─────────────────────────────────────────────
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
import cv2
import numpy as np
clean_qt_environment()

# ── PIL ───────────────────────────────────────────────────────────────────────
from PIL import Image as PILImage, ImageDraw

# ── PyQt5 ─────────────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QLineEdit, QPushButton, QSplitter,
    QMessageBox, QFileDialog,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont

# ── RoboPoint conversation ────────────────────────────────────────────────────
try:
    from point.conversation import default_conversation, conv_templates, SeparatorStyle
except ImportError:
    print("Warning: Could not import point.conversation, LLM features may be limited",
          file=sys.stderr)
    default_conversation = None
    conv_templates = None
    SeparatorStyle = None

# ── Wit.ai ────────────────────────────────────────────────────────────────────
VOICE_CLIENT = None
VOICE_INIT_ERROR = None
WIT_TOKEN_PATH = os.getenv("WIT_TOKEN_PATH", "/workspace/wit_token.txt")


def get_wit_client():
    global VOICE_CLIENT, VOICE_INIT_ERROR
    if VOICE_CLIENT is not None:
        return VOICE_CLIENT
    if Wit is None:
        VOICE_INIT_ERROR = "wit package not installed."
        return None
    try:
        with open(WIT_TOKEN_PATH, "r") as f:
            token = f.read().strip()
        if not token:
            VOICE_INIT_ERROR = "Wit token file is empty."
            return None
        VOICE_CLIENT = Wit(token)
        VOICE_INIT_ERROR = None
        return VOICE_CLIENT
    except FileNotFoundError:
        VOICE_INIT_ERROR = f"Wit token file '{WIT_TOKEN_PATH}' not found."
    except Exception as exc:
        VOICE_INIT_ERROR = f"Failed to initialize Wit client: {exc}"
    return None


def voice_support_status():
    if pyaudio is None:
        return False, "pyaudio not installed."
    client = get_wit_client()
    if client is None:
        return False, VOICE_INIT_ERROR or "Wit client unavailable."
    return True, ""


def split_worker_response(response: str):
    marker = "__SNAPPED_POINTS__:"
    if marker not in response:
        return response, None
    display_text, metadata = response.split(marker, 1)
    try:
        points = json.loads(metadata.strip())
    except Exception:
        points = None
    return display_text.strip(), points


def model_name_from_path(model_path: str) -> str:
    path = str(model_path or "").rstrip("/")
    if not path:
        return ""
    name = pathlib.PurePosixPath(path).name
    if name.startswith("checkpoint-"):
        parent = pathlib.PurePosixPath(path).parent.name
        return f"{parent}_{name}"
    return name


def record_audio_pyaudio(duration=3, rate=16000, chunk=1024):
    if pyaudio is None:
        return None
    audio = pyaudio.PyAudio()
    stream = None
    try:
        input_device_index = None
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                name = info.get('name', '').lower()
                if any(k in name for k in ['usb', 'webcam', 'microphone', 'mic']):
                    input_device_index = i
                    break
        if input_device_index is None:
            for i in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(i)
                if info.get('maxInputChannels', 0) > 0:
                    input_device_index = i
                    break
        stream = audio.open(
            format=pyaudio.paInt16, channels=1, rate=rate,
            input=True, input_device_index=input_device_index,
            frames_per_buffer=chunk,
        )
        frames = []
        for _ in range(max(1, int(rate / float(chunk) * float(duration)))):
            frames.append(stream.read(chunk, exception_on_overflow=False))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
        buf.seek(0)
        return buf.getvalue()
    except Exception as exc:
        print(f"Voice capture failed: {exc}", file=sys.stderr)
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


def recognize_speech_with_wit(audio_bytes, client=None):
    if not audio_bytes:
        return None
    if client is None:
        client = get_wit_client()
    if client is None:
        return None
    try:
        buf = io.BytesIO(audio_bytes)
        buf.name = "voice.wav"
        buf.seek(0)
        response = client.speech(buf, {"Content-Type": "audio/wav"})
        if isinstance(response, dict):
            return response.get('text') or response.get('_text')
    except Exception as exc:
        print(f"Wit.ai speech recognition failed: {exc}", file=sys.stderr)
    return None


# ── 工具函式 ─────────────────────────────────────────────────────────────────
def find_vectors(text):
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    vectors = []
    for match in matches:
        vector = [float(n) if '.' in n else int(n) for n in match.split(',')]
        vectors.append(vector)
    return vectors


def visualize_2d(img, points, bboxes, scale=1.0, cross_size=9, cross_width=4):
    if isinstance(img, np.ndarray):
        img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    draw = ImageDraw.Draw(img)
    size = int(cross_size * scale)
    width = int(cross_width * scale)
    for x, y in points:
        draw.line((x - size, y - size, x + size, y + size), fill='red', width=width)
        draw.line((x - size, y + size, x + size, y - size), fill='red', width=width)
    for x1, y1, x2, y2 in bboxes:
        draw.rectangle([x1, y1, x2, y2], outline='red', width=width)
    return img.convert('RGB')


# ══════════════════════════════════════════════════════════════════════════════
# Qt Threads
# ══════════════════════════════════════════════════════════════════════════════
class VoiceRecognitionThread(QThread):
    result_ready = pyqtSignal(str)
    error = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(self, duration_sec=4.0):
        super().__init__()
        self.duration_sec = duration_sec

    def run(self):
        try:
            self.status_changed.emit("Recording...")
            audio = record_audio_pyaudio(duration=self.duration_sec)
            if audio is None:
                self.error.emit("Failed to record audio.")
                return
            self.status_changed.emit("Recognizing...")
            text = recognize_speech_with_wit(audio)
            if text:
                self.result_ready.emit(text)
            else:
                self.error.emit("Could not recognize speech.")
        except Exception as e:
            self.error.emit(str(e))


class LLMWorkerThread(QThread):
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished_request = pyqtSignal()
    controller_connected = pyqtSignal(str)
    worker_processing = pyqtSignal()

    def __init__(self, controller_url="http://10.0.0.30:11000"):
        super().__init__()
        self.controller_url = controller_url
        self.request_data = None

    def set_request_data(self, data):
        self.request_data = data

    def run(self):
        try:
            ret = requests.post(
                self.controller_url + "/get_worker_address",
                json={"model": self.request_data["model"]},
                timeout=5,
            )
            worker_addr = ret.json()["address"]
            if worker_addr == "":
                self.error_occurred.emit("No available worker")
                return
            self.controller_connected.emit(worker_addr)
            self.worker_processing.emit()
            response = requests.post(
                worker_addr + "/worker_generate_stream",
                headers={"User-Agent": "White Point GUI Client"},
                json=self.request_data,
                stream=True,
                timeout=180,
            )
            full_output = ""
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if not chunk:
                    continue
                data = json.loads(chunk.decode())
                if data.get("error_code", 1) == 0:
                    full_output = data["text"][len(self.request_data["prompt"]):].strip()
                else:
                    self.error_occurred.emit(
                        f"Error: {data.get('text', '')} (code: {data.get('error_code')})"
                    )
                    return
            if full_output:
                self.response_received.emit(full_output)
            self.finished_request.emit()
        except Exception as e:
            self.error_occurred.emit(f"Request failed: {str(e)}")


class ROSSignalBridge(QObject):
    image_signal = pyqtSignal(object)
    model_output_signal = pyqtSignal(str)
    point3d_signal = pyqtSignal(float, float, float)
    selection_phase_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()


class ROS2Thread(QThread):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.running = True

    def run(self):
        while self.running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def stop(self):
        self.running = False


# ══════════════════════════════════════════════════════════════════════════════
# 可點擊影像標籤
# ══════════════════════════════════════════════════════════════════════════════
class ImageLabel(QLabel):
    file_selection_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid #ccc; cursor: hand;")
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def mousePressEvent(self, event):
        # Always request file selection when clicked (no point selection on image)
        if event.button() == Qt.LeftButton:
            self.file_selection_requested.emit()

    def set_image(self, cv_image):
        h, w = cv_image.shape[:2]
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scale_x = scaled_pixmap.width() / w
        self.scale_y = scaled_pixmap.height() / h
        self.offset_x = (self.width() - scaled_pixmap.width()) / 2
        self.offset_y = (self.height() - scaled_pixmap.height()) / 2
        self.setPixmap(scaled_pixmap)


# ══════════════════════════════════════════════════════════════════════════════
# ROS2 節點（獨立版：不訂閱相機 topic，改為接受外部呼叫傳入影像）
# ══════════════════════════════════════════════════════════════════════════════
class WhitePointGUIStandalone(Node):
    """
    獨立測試用 ROS2 節點。
    ● 不訂閱 /d435i/color/image_raw（影像由使用者從檔案載入）
    ● 保留發布 /white_point_pixel 與 /user_input 以便與其他節點互動
    ● 仍訂閱 /white_point_base、/model_output、/white_point_selection_phase
    """

    def __init__(self, signal_bridge):
        super().__init__('white_point_gui_standalone')
        self.bridge = CvBridge()
        self.signal_bridge = signal_bridge

        # 發布像素點
        self.pixel_pub = self.create_publisher(Point, '/white_point_pixel', 10)

        # 訂閱 3D 結果（選用，若其他節點有執行）
        self.point3d_sub = self.create_subscription(
            PointStamped, '/white_point_base', self.point3d_callback, 10)

        # 訂閱模型輸出
        self.model_output_sub = self.create_subscription(
            String, '/model_output', self.model_output_callback, 10)

        # 訂閱選點階段
        self.selection_phase_sub = self.create_subscription(
            String, '/white_point_selection_phase', self.selection_phase_callback, 10)

        # 發布使用者輸入
        self.user_input_pub = self.create_publisher(String, '/user_input', 10)

        self.last_click = None
        self.last_3d = None

    def point3d_callback(self, msg: PointStamped):
        self.last_3d = (msg.point.x, msg.point.y, msg.point.z)
        self.signal_bridge.point3d_signal.emit(msg.point.x, msg.point.y, msg.point.z)

    def model_output_callback(self, msg: String):
        self.signal_bridge.model_output_signal.emit(msg.data)

    def selection_phase_callback(self, msg: String):
        self.signal_bridge.selection_phase_signal.emit(msg.data.strip())

    def publish_pixel(self, x: int, y: int):
        self.last_click = (x, y)
        pt = Point()
        pt.x = float(x)
        pt.y = float(y)
        self.pixel_pub.publish(pt)
        self.get_logger().info(f'Clicked pixel: u={x}, v={y}')

    def publish_user_input(self, text: str):
        msg = String()
        msg.data = text
        self.user_input_pub.publish(msg)
        self.get_logger().info(f"User input: {text}")


# ══════════════════════════════════════════════════════════════════════════════
# 主視窗
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self, ros_node, signal_bridge,
                 controller_url="http://10.0.0.30:11000",
                 model_path="PME033541/vla13"):
        super().__init__()
        self.ros_node = ros_node
        self.signal_bridge = signal_bridge
        self.controller_url = controller_url
        self.model_path = model_path
        self.setWindowTitle("Vision–Language Robotics Control Panel [Standalone / Offline Test]")
        self.setGeometry(100, 100, 1200, 1000)

        self.llm_worker = None
        self.current_image = None          # numpy BGR
        self.llm_points = []
        self.conversation_state = None

        if default_conversation is not None:
            self.conversation_state = default_conversation.copy()

        self.voice_thread = None
        self.voice_capture_duration = 4.0
        self.voice_available, self.voice_unavailable_reason = voice_support_status()
        self.is_processing_voice = False
        self.selection_phase = "select_first_point"

        self.setup_ui()
        self.connect_signals()

        QTimer.singleShot(2000, self.check_service_status)

    # ──────────────────────────────────────────────────────────────────────────
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ── 左側：影像區域 ───────────────────────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # 影像顯示（點擊以打開照片或選擇點）
        self.image_label = ImageLabel()
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setSizePolicy(
            self.image_label.sizePolicy().horizontalPolicy(),
            self.image_label.sizePolicy().verticalPolicy()
        )
        left_layout.addWidget(self.image_label, stretch=1)

        # 服務狀態列（LLM 連線狀態）
        self.service_status = QLabel("Controller: Not Connected | Worker: Not Connected")
        self.service_status.setAlignment(Qt.AlignCenter)
        self.service_status.setFont(QFont("Arial", 11, QFont.Bold))
        self.service_status.setStyleSheet(
            "padding: 8px; background-color: #f0f0f0; border-radius: 4px;"
        )
        left_layout.addWidget(self.service_status)

        main_layout.addWidget(left_widget, stretch=1)

        # ── 右側：文字區域 ───────────────────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        title_label = QLabel("Model Interaction")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(title_label)

        output_label = QLabel("Output:")
        output_label.setFont(QFont("Arial", 14, QFont.Bold))
        right_layout.addWidget(output_label)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Monospace", 20))
        self.output_text.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 2px solid #ccc;
                border-radius: 5px;
                padding: 8px;
            }
        """)
        right_layout.addWidget(self.output_text, stretch=3)

        input_label = QLabel("Input:")
        input_label.setFont(QFont("Arial", 14, QFont.Bold))
        right_layout.addWidget(input_label)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type message and press Enter...")
        self.input_field.setFont(QFont("Arial", 13))
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: 2px solid #ccc;
                border-radius: 5px;
                padding: 10px;
            }
            QLineEdit:focus { border: 2px solid #4CAF50; }
        """)
        right_layout.addWidget(self.input_field)

        self.send_button = QPushButton("Send")
        self.send_button.setFont(QFont("Arial", 13, QFont.Bold))
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; color: white;
                border: none; border-radius: 5px; padding: 12px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #3d8b40; }
        """)
        right_layout.addWidget(self.send_button)

        self.voice_button = QPushButton("Voice Input")
        self.voice_button.setFont(QFont("Arial", 13, QFont.Bold))
        self.voice_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; color: white;
                border: none; border-radius: 5px; padding: 12px;
            }
            QPushButton:hover { background-color: #0b7dda; }
            QPushButton:pressed { background-color: #0869c7; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        right_layout.addWidget(self.voice_button)

        main_layout.addWidget(right_widget, stretch=1)

    # ──────────────────────────────────────────────────────────────────────────
    def connect_signals(self):
        self.signal_bridge.image_signal.connect(self.update_image)
        self.signal_bridge.model_output_signal.connect(self.add_model_output)
        self.signal_bridge.point3d_signal.connect(self.update_3d_coord)
        self.signal_bridge.selection_phase_signal.connect(self.update_selection_phase)

        self.image_label.file_selection_requested.connect(self._open_photo)
        self.send_button.clicked.connect(self.send_user_input)
        self.input_field.returnPressed.connect(self.send_user_input)
        self.voice_button.clicked.connect(self.handle_voice_button)

        self._update_voice_button_state()

    # ──────────────────────────────────────────────────────────────────────────
    # Open local photo (triggered by clicking image label when no image is loaded)
    # ──────────────────────────────────────────────────────────────────────────
    def _open_photo(self):
        """Open a file dialog so the user can select a local image for testing."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Test Photo",
            "/workspace/Outputs/model_inputs/data",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.tif);;All Files (*)",
        )
        if not file_path:
            return  # user cancelled

        img = cv2.imread(file_path)
        if img is None:
            QMessageBox.warning(self, "Cannot Load", f"Cannot read image:\n{file_path}")
            return

        # Update image display
        self.update_image(img)

        # self.output_text.append(
        #     f"<b style='color: #FF9800;'>Loaded:</b> {os.path.basename(file_path)}"
        # )

    # ──────────────────────────────────────────────────────────────────────────
    def update_image(self, cv_image):
        self.current_image = cv_image.copy()
        self.image_label.set_image(cv_image)

    def add_model_output(self, text: str):
        pass  # 原版相同：不直接顯示原始 ROS model output

    def update_3d_coord(self, x: float, y: float, z: float):
        pass



    def phase_to_text(self, phase: str):
        mapping = {
            "select_first_point":     "請選第一個點（點選後會詢問確認）",
            "moving_to_approach":     "前往準備點中（暫不接受點選）",
            "waiting_second_point":   "已到準備點，請選第二個點",
            "moving_to_target":       "前往目標點中（暫不接受點選）",
        }
        return mapping.get(phase, f"未知階段：{phase}")

    def update_selection_phase(self, phase: str):
        if not phase:
            return
        self.selection_phase = phase

    # ──────────────────────────────────────────────────────────────────────────
    def append_point_preview(self, pixel_points):
        if not pixel_points or self.current_image is None:
            return
        try:
            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            annotated = visualize_2d(pil_image, pixel_points, [], scale=1.0)
            annotated_cv = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
            h_img, w_img = annotated_cv.shape[:2]
            rgb_qt = cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB)
            q_image = QImage(rgb_qt.data, w_img, h_img, 3 * w_img, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            max_width = self.output_text.width() - 40
            if pixmap.width() > max_width:
                pixmap = pixmap.scaled(
                    max_width,
                    int(pixmap.height() * max_width / pixmap.width()),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
            cursor = self.output_text.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText("\n")
            cursor.insertImage(pixmap.toImage())
            self.output_text.setTextCursor(cursor)
            self.output_text.verticalScrollBar().setValue(
                self.output_text.verticalScrollBar().maximum()
            )
        except Exception as e:
            print(f"Failed to create point preview: {e}", file=sys.stderr)

    def append_manual_selection_output(self, x: int, y: int):
        self.output_text.append(
            f"<b style='color: #2196F3;'>Selected:</b> ({int(x)}, {int(y)})"
        )
        self.append_point_preview([(int(x), int(y))])
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum()
        )

    def confirm_and_publish_pixel(self, x: int, y: int, source: str = "manual"):
        allowed_phases = {"select_first_point", "waiting_second_point"}
        if self.selection_phase not in allowed_phases:
            self.output_text.append(
                f"<b style='color: #FF9800;'>提示:</b> 目前階段為「{self.phase_to_text(self.selection_phase)}」，暫不接受新點。"
            )
            return

        is_second = (self.selection_phase == "waiting_second_point")
        ordinal = "第二次" if is_second else "第一次"

        if source == "manual":
            self.append_manual_selection_output(x, y)

        reply = QMessageBox.question(
            self, "確認選點",
            f"座標: ({x}, {y})\n\n要選擇這個{ordinal}點嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.ros_node.publish_pixel(x, y)

    # ──────────────────────────────────────────────────────────────────────────
    def send_user_input(self):
        text = self.input_field.text().strip()
        if not text:
            return

        self.output_text.append(f"<b style='color: #4CAF50;'>You:</b> {text}")
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum()
        )
        self.ros_node.publish_user_input(text)
        self.input_field.clear()

        if self.current_image is None:
            self.output_text.append(
                "<b style='color: #F44336;'>錯誤:</b> 請先載入照片（點擊「開啟照片」按鈕）"
            )
            return

        try:
            rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            image = PILImage.fromarray(rgb)
        except Exception as e:
            self.output_text.append(f"<b style='color: #F44336;'>影像轉換失敗:</b> {e}")
            return

        self.send_to_llm(text, image)

    # ──────────────────────────────────────────────────────────────────────────
    def send_to_llm(self, text: str, image: PILImage.Image):
        try:
            if default_conversation is None:
                self.output_text.append(
                    "<b style='color: #F44336;'>錯誤:</b> conversation 模組未載入"
                )
                return

            if self.llm_worker is not None and self.llm_worker.isRunning():
                self.llm_worker.wait()

            self.llm_worker = LLMWorkerThread(self.controller_url)
            self.llm_worker.response_received.connect(self.handle_llm_response)
            self.llm_worker.error_occurred.connect(self.handle_llm_error)
            self.llm_worker.finished_request.connect(self.llm_request_finished)
            self.llm_worker.controller_connected.connect(self.on_controller_connected)
            self.llm_worker.worker_processing.connect(self.on_worker_processing)

            self.service_status.setText("Controller: Connecting... | Worker: Waiting...")

            self.conversation_state = default_conversation.copy()

            if '<image>' not in text:
                text = '<image>\n' + text

            content = (text, image, 'Original')
            self.conversation_state.append_message(self.conversation_state.roles[0], content)
            self.conversation_state.append_message(self.conversation_state.roles[1], None)

            template_name = 'vicuna_v1'
            if len(self.conversation_state.messages) == 2:
                new_state = conv_templates[template_name].copy()
                new_state.append_message(
                    new_state.roles[0], self.conversation_state.messages[-2][1]
                )
                new_state.append_message(new_state.roles[1], None)
                self.conversation_state = new_state

            prompt = self.conversation_state.get_prompt()
            pil_images, images, transforms = self.conversation_state.get_images()

            request_data = {
                'model': model_name_from_path(self.model_path),
                'prompt': prompt,
                'temperature': 0,
                'top_p': 1.0,
                'max_new_tokens': 256,
                'use_cache': False,
                'stop': (
                    self.conversation_state.sep
                    if self.conversation_state.sep_style in [
                        SeparatorStyle.SINGLE, SeparatorStyle.MPT
                    ]
                    else self.conversation_state.sep2
                ),
                'images': images,
            }
            self.llm_worker.set_request_data(request_data)
            self.llm_worker.start()

            self.send_button.setEnabled(False)
            self.send_button.setText("Processing...")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.output_text.append(f"<b style='color: #F44336;'>Error:</b> {str(e)}")
            self.service_status.setText("Controller: Error | Worker: Error")

    # ──────────────────────────────────────────────────────────────────────────
    def handle_llm_response(self, response: str):
        display_response, snapped_points = split_worker_response(response)
        vectors = snapped_points if snapped_points else find_vectors(display_response)
        vectors_2d = [vec for vec in vectors if len(vec) == 2]

        if vectors_2d and self.current_image is not None:
            h, w = self.current_image.shape[:2]
            self.llm_points = []
            pixel_points = []
            norm_points = []
            for x, y in vectors_2d:
                x = float(x)
                y = float(y)
                if isinstance(x, float) and x <= 1:
                    px, py = int(x * w), int(y * h)
                    norm_points.append((x, y))
                else:
                    px, py = int(x), int(y)
                    norm_points.append(None)
                self.llm_points.append((px, py))
                pixel_points.append((px, py))

            # 顯示 LLM 解析出的點及標註圖片
            if any(point is not None for point in norm_points):
                point_info = display_response.replace('\n', ' ').strip()
            else:
                point_info = ", ".join([f"({px}, {py})" for px, py in pixel_points])
            self.output_text.append(
                f"<b style='color: #2196F3;'>LLM:</b> {point_info}"
            )
            self.append_point_preview(pixel_points)

            # 單點直接用，多點在 pixel 階段選最右下角
            if len(self.llm_points) == 1:
                px, py = self.llm_points[0]
                self.confirm_and_publish_pixel(int(px), int(py), source="llm")
            elif len(self.llm_points) > 1:
                best_px, best_py = self._select_best_pixel(self.llm_points)
                self.confirm_and_publish_pixel(int(best_px), int(best_py), source="llm")
        else:
            clean = display_response.replace('\n', ' ').strip()
            self.output_text.append(
                f"<b style='color: #2196F3;'>LLM:</b> {clean}"
            )

    def _select_best_pixel(self, pixel_points):
        """
        直接在旋轉後影像座標中選點，依據：
          1. px + py 最大（最靠右下）
          2. py 最大（若總和相同，偏下優先）
          3. px 最大（若仍相同，偏右優先）
        """
        if not pixel_points:
            self.ros_node.get_logger().warn('select_best_pixel: no candidates')
            return (0, 0)

        best = max(
            pixel_points,
            key=lambda p: (float(p[0]) + float(p[1]), float(p[1]), float(p[0])),
        )
        self.ros_node.get_logger().info(
            f'select_best_pixel: best=({best[0]:.0f},{best[1]:.0f}) '
            f'pixel_sum={float(best[0]) + float(best[1]):.0f} '
            f'from {len(pixel_points)} candidates'
        )
        return best

    def on_controller_connected(self, worker_addr: str):
        self.service_status.setText(f"Controller: ✓ Connected | Worker: {worker_addr}")
        self.service_status.setStyleSheet(
            "padding: 8px; background-color: #4CAF50; color: white; border-radius: 4px; font-weight: bold;"
        )

    def on_worker_processing(self):
        current = self.service_status.text()
        if "Worker:" in current:
            parts = current.split("|")
            self.service_status.setText(f"{parts[0]}| Worker: ⏳ Processing...")

    def handle_llm_error(self, error: str):
        self.output_text.append(f"<b style='color: #F44336;'>LLM Error:</b> {error}")
        self.service_status.setText("Controller: ✗ Error | Worker: ✗ Error")
        self.service_status.setStyleSheet(
            "padding: 8px; background-color: #F44336; color: white; border-radius: 4px; font-weight: bold;"
        )
        self.send_button.setEnabled(True)
        self.send_button.setText("Send")

    def llm_request_finished(self):
        current = self.service_status.text()
        if "Controller:" in current:
            parts = current.split("|")
            self.service_status.setText(f"{parts[0]}| Worker: ✓ Done")
            self.service_status.setStyleSheet(
                "padding: 8px; background-color: #4CAF50; color: white; border-radius: 4px; font-weight: bold;"
            )
        self.send_button.setEnabled(True)
        self.send_button.setText("Send")

    # ── 語音 ──────────────────────────────────────────────────────────────────
    def handle_voice_button(self):
        if not self.voice_available:
            self.output_text.append(
                f"<b style='color: #F44336;'>Voice input unavailable:</b> "
                f"{self.voice_unavailable_reason}"
            )
            return
        if self.is_processing_voice:
            return
        self.is_processing_voice = True
        self.voice_button.setEnabled(False)
        self.voice_button.setText("Recording...")
        self.voice_thread = VoiceRecognitionThread(duration_sec=self.voice_capture_duration)
        self.voice_thread.result_ready.connect(self.on_voice_result)
        self.voice_thread.error.connect(self.on_voice_error)
        self.voice_thread.status_changed.connect(self.on_voice_status_changed)
        self.voice_thread.finished.connect(self.on_voice_finished)
        self.voice_thread.start()

    def on_voice_result(self, text: str):
        self.input_field.setText(text)
        QTimer.singleShot(100, self.send_user_input)

    def on_voice_error(self, error_msg: str):
        self.output_text.append(f"<b style='color: #F44336;'>Voice Error:</b> {error_msg}")

    def on_voice_status_changed(self, status: str):
        self.voice_button.setText(status)

    def on_voice_finished(self):
        self.is_processing_voice = False
        self.voice_thread = None
        self._update_voice_button_state()

    def _update_voice_button_state(self):
        if not self.voice_available:
            self.voice_button.setEnabled(False)
            self.voice_button.setText("Voice Unavailable")
            self.voice_button.setToolTip(self.voice_unavailable_reason)
        else:
            self.voice_button.setEnabled(True)
            self.voice_button.setText("Voice Input")
            self.voice_button.setToolTip("Click to start voice input (about 4 sec)")

    # ── 服務狀態檢查 ──────────────────────────────────────────────────────────
    def check_service_status(self):
        """Periodically check LLM controller connection status."""
        try:
            ret = requests.post(
                self.controller_url + "/list_models", json={}, timeout=2
            )
            if ret.status_code == 200:
                models = ret.json().get("models", [])
                if models:
                    self.service_status.setText(
                        f"Controller: ✓ Connected | Worker: ✓ Ready ({len(models)} models)"
                    )
                    self.service_status.setStyleSheet(
                        "padding: 8px; background-color: #4CAF50; color: white; border-radius: 4px; font-weight: bold;"
                    )
                    # Check again in 10 seconds
                    QTimer.singleShot(10000, self.check_service_status)
                else:
                    self.service_status.setText(
                        "Controller: ✓ Connected | Worker: ⏳ Loading..."
                    )
                    self.service_status.setStyleSheet(
                        "padding: 8px; background-color: #FF9800; color: white; border-radius: 4px; font-weight: bold;"
                    )
                    QTimer.singleShot(10000, self.check_service_status)
            else:
                self.service_status.setText(
                    "Controller: ✓ Connected | Worker: ✗ Error"
                )
                self.service_status.setStyleSheet(
                    "padding: 8px; background-color: #F44336; color: white; border-radius: 4px; font-weight: bold;"
                )
                QTimer.singleShot(5000, self.check_service_status)
        except Exception as e:
            self.service_status.setText(
                f"Controller: ✗ Cannot Connect ({self.controller_url})"
            )
            self.service_status.setStyleSheet(
                "padding: 8px; background-color: #F44336; color: white; border-radius: 4px; font-weight: bold;"
            )
            QTimer.singleShot(5000, self.check_service_status)

    def closeEvent(self, event):
        self.ros_node.get_logger().info("Shutting down standalone GUI...")
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════
def main(args=None):
    parser = argparse.ArgumentParser(
        description="White Point GUI — Standalone / Offline Test Mode"
    )
    parser.add_argument("--controller-url", type=str, default="http://10.0.0.30:11000",
                        help="LLM controller URL")
    parser.add_argument("--model-path", type=str, default="PME033541/vla13",
                        help="Model path")
    parser.add_argument("--ros-args", nargs=argparse.REMAINDER, help="ROS arguments")

    custom_args = []
    ros_args_list = []
    if '--ros-args' in sys.argv:
        idx = sys.argv.index('--ros-args')
        custom_args = sys.argv[1:idx]
        ros_args_list = sys.argv[idx + 1:]
    else:
        custom_args = sys.argv[1:]

    parsed_args = parser.parse_args(custom_args)

    rclpy.init(args=ros_args_list if ros_args_list else None)

    app = QApplication([sys.argv[0]])

    signal_bridge = ROSSignalBridge()
    ros_node = WhitePointGUIStandalone(signal_bridge)

    window = MainWindow(
        ros_node, signal_bridge,
        controller_url=parsed_args.controller_url,
        model_path=parsed_args.model_path,
    )
    window.show()

    ros_thread = ROS2Thread(ros_node)
    ros_thread.start()

    exit_code = app.exec_()

    ros_thread.stop()
    ros_thread.wait()
    ros_node.destroy_node()
    rclpy.shutdown()

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
