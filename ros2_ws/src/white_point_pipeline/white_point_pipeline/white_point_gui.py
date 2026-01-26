#!/usr/bin/env python3
import sys
import os
import ctypes
import io
import wave

# Silence ALSA warnings when initializing PyAudio
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
            _alsa_handle = ctypes.cdll.LoadLibrary("libasound.so")
            _alsa_handle.snd_lib_error_set_handler(_alsa_handler)
        except OSError:
            _alsa_handle = None
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

# 最激進的 Qt 插件清理 - 必須在所有導入之前
# 完全移除 cv2 的 Qt 插件路徑
def clean_qt_environment():
    # 先設定正確的系統 Qt 路徑
    for p in [
        '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms',
        '/usr/lib/x86_64-linux-gnu/qt/plugins/platforms',
        '/usr/lib/qt/plugins/platforms',
    ]:
        if os.path.isdir(p):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = p
            os.environ['QT_PLUGIN_PATH'] = os.path.dirname(p)
            break
    
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    
    # 清除所有可能的 cv2 Qt 路徑
    qt_keys_to_clean = ['QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH']
    for key in qt_keys_to_clean:
        if key in os.environ:
            val = os.environ[key]
            # 移除任何包含 cv2 的路徑
            if 'cv2' in val or 'opencv' in val.lower():
                del os.environ[key]

clean_qt_environment()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge
import subprocess
import sys

# 導入 cv2 但設定為不使用 GUI
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
import cv2

# cv2 導入後再次強制清理
clean_qt_environment()

import numpy as np
import threading
import re
import requests
import json
import hashlib
from PIL import Image as PILImage, ImageDraw

# 現在才導入 PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QLineEdit, QPushButton, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont

# 導入 RoboPoint conversation 模組
try:
    from point.conversation import default_conversation, conv_templates, SeparatorStyle
except ImportError:
    print("Warning: Could not import point.conversation, LLM features may be limited", file=sys.stderr)
    default_conversation = None
    conv_templates = None
    SeparatorStyle = None

VOICE_CLIENT = None
VOICE_INIT_ERROR = None
WIT_TOKEN_PATH = os.getenv("WIT_TOKEN_PATH", "/workspace/wit_token.txt")


def get_wit_client():
    """Lazily initialize and cache the Wit.ai client."""
    global VOICE_CLIENT, VOICE_INIT_ERROR
    if VOICE_CLIENT is not None:
        return VOICE_CLIENT
    if Wit is None:
        VOICE_INIT_ERROR = "wit package not installed."
        return None
    token_path = WIT_TOKEN_PATH
    try:
        with open(token_path, "r") as token_file:
            token = token_file.read().strip()
        if not token:
            VOICE_INIT_ERROR = f"Wit token file '{token_path}' is empty."
            return None
        VOICE_CLIENT = Wit(token)
        VOICE_INIT_ERROR = None
        return VOICE_CLIENT
    except FileNotFoundError:
        VOICE_INIT_ERROR = f"Wit token file '{token_path}' not found."
    except Exception as exc:
        VOICE_INIT_ERROR = f"Failed to initialize Wit client: {exc}"
    return None


def voice_support_status():
    """Return (available, reason) for voice support."""
    if pyaudio is None:
        return False, "PyAudio is not available. Install PyAudio to enable voice input."
    client = get_wit_client()
    if client is None:
        return False, VOICE_INIT_ERROR or "Wit.ai client is not available."
    return True, ""


def record_audio_pyaudio(duration=3, rate=16000, chunk=1024):
    """Capture audio from the default microphone and return WAV bytes."""
    if pyaudio is None:
        return None
    audio = pyaudio.PyAudio()
    stream = None
    try:
        # Find a working input device (prefer USB devices like webcam)
        input_device_index = None
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                # Prefer USB Audio devices (webcams, etc.)
                if 'USB' in info.get('name', '').upper() or 'WEBCAM' in info.get('name', '').upper():
                    input_device_index = i
                    break
                # Fallback to first input device found
                if input_device_index is None:
                    input_device_index = i
        
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=chunk,
        )
        frames = []
        frame_count = int(rate / float(chunk) * float(duration))
        for _ in range(max(1, frame_count)):
            frames.append(stream.read(chunk, exception_on_overflow=False))
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(rate)
            wav_file.writeframes(b"".join(frames))
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as exc:
        print(f"Voice capture failed: {exc}", file=sys.stderr)
        return None
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        try:
            audio.terminate()
        except Exception:
            pass


def recognize_speech_with_wit(audio_bytes, client=None):
    """Send audio bytes to Wit.ai and return recognized text."""
    if not audio_bytes:
        return None
    if client is None:
        client = get_wit_client()
    if client is None:
        return None
    try:
        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.name = "voice.wav"
        audio_buffer.seek(0)
        response = client.speech(audio_buffer, {"Content-Type": "audio/wav"})
        if isinstance(response, dict):
            text = response.get("text", "")
            return text.strip() or None
    except Exception as exc:
        print(f"Wit.ai speech recognition failed: {exc}", file=sys.stderr)
    return None


def rotate_img_90(img):
    """旋轉影像 90 度（順時針）"""
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def find_vectors(text):
    """從文字中找出座標向量"""
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    vectors = []
    for match in matches:
        vector = [float(num) if '.' in num else int(num) for num in match.split(',')]
        vectors.append(vector)
    return vectors


def visualize_2d(img, points, bboxes, scale=1.0, cross_size=9, cross_width=4):
    """在影像上繪製點和邊界框"""
    if isinstance(img, np.ndarray):
        # Convert numpy array to PIL Image
        img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    draw = ImageDraw.Draw(img)
    size = int(cross_size * scale)
    width = int(cross_width * scale)
    
    # Draw each point as a red X
    for x, y in points:
        draw.line((x - size, y - size, x + size, y + size), fill='red', width=width)
        draw.line((x - size, y + size, x + size, y - size), fill='red', width=width)
    
    # Draw each bounding box
    for x1, y1, x2, y2 in bboxes:
        draw.rectangle([x1, y1, x2, y2], outline='red', width=width)
    
    img = img.convert('RGB')
    return img


class VoiceRecognitionThread(QThread):
    """Thread for voice recognition using Wit.ai"""
    result_ready = pyqtSignal(str)
    error = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(self, duration_sec=4.0):
        super().__init__()
        self.duration_sec = duration_sec

    def run(self):
        """Record and recognize speech."""
        try:
            self.status_changed.emit("Recording...")
            audio_bytes = record_audio_pyaudio(duration=self.duration_sec)
            if not audio_bytes:
                self.error.emit("Failed to record audio")
                return
            
            self.status_changed.emit("Recognizing...")
            client = get_wit_client()
            text = recognize_speech_with_wit(audio_bytes, client=client)
            if text:
                self.result_ready.emit(text)
            else:
                self.error.emit("Failed to recognize speech")
        except Exception as e:
            self.error.emit(f"Voice recognition error: {str(e)}")


class ServerProcess:
    """管理 Controller 和 Model Worker 進程"""
    def __init__(self):
        self.controller_process = None
        self.model_worker_process = None
        self.controller_url = "http://10.0.0.1:11000"

    def start_controller(self, host="0.0.0.0", port=11000):
        """啟動 Controller"""
        cmd = [sys.executable, "-m", "point.serve.controller", "--host", host, "--port", str(port)]
        print(f"Starting controller: {' '.join(cmd)}")
        self.controller_process = subprocess.Popen(cmd)
        self.controller_url = f"http://{('10.0.0.1' if host in ['0.0.0.0', '::'] else host)}:{port}"

    def start_model_worker(self, host="0.0.0.0", controller_url="http://10.0.0.1:11000",
                           port=22000, worker_url="http://10.0.0.1:22000",
                           model_path="PME033541/vla2.7", load_4bit=True):
        """啟動 Model Worker"""
        cmd = [
            sys.executable, "-m", "point.serve.model_worker",
            "--host", host,
            "--port", str(port),
            "--controller-address", controller_url,
            "--worker-address", worker_url,
            "--model-path", model_path
        ]
        if load_4bit:
            cmd.append("--load-4bit")
        print(f"Starting model worker: {' '.join(cmd)}")
        self.model_worker_process = subprocess.Popen(cmd)

    def stop_all(self):
        """停止所有服務"""
        if self.controller_process:
            self.controller_process.terminate()
            self.controller_process.wait()
        if self.model_worker_process:
            self.model_worker_process.terminate()
            self.model_worker_process.wait()


class LLMWorkerThread(QThread):
    """處理 LLM 請求的執行緒"""
    response_received = pyqtSignal(str)  # 完整回應（只發送一次）
    error_occurred = pyqtSignal(str)
    finished_request = pyqtSignal()
    controller_connected = pyqtSignal(str)  # 發送 worker 地址
    worker_processing = pyqtSignal()  # Worker 開始處理
    
    def __init__(self, controller_url="http://10.0.0.1:11000"):
        super().__init__()
        self.controller_url = controller_url
        self.request_data = None
        
    def set_request_data(self, data):
        self.request_data = data
        
    def run(self):
        try:
            # Get worker address
            ret = requests.post(
                self.controller_url + "/get_worker_address",
                json={"model": self.request_data["model"]},
                timeout=5
            )
            worker_addr = ret.json()["address"]
            
            if worker_addr == "":
                self.error_occurred.emit("No available worker")
                return
            
            # 發送 controller 連接成功和 worker 地址
            self.controller_connected.emit(worker_addr)
            
            # 開始處理
            self.worker_processing.emit()
            
            # Stream output（累積完整回應）
            response = requests.post(
                worker_addr + "/worker_generate_stream",
                headers={"User-Agent": "White Point GUI Client"},
                json=self.request_data,
                stream=True,
                timeout=30
            )
            
            full_output = ""
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if not chunk:
                    continue
                data = json.loads(chunk.decode())
                if data.get("error_code", 1) == 0:
                    # 累積回應文字
                    full_output = data["text"][len(self.request_data["prompt"]):].strip()
                else:
                    self.error_occurred.emit(f"Error: {data.get('text','')} (code: {data.get('error_code')})")
                    return
            
            # 只在最後發送完整回應一次
            if full_output:
                self.response_received.emit(full_output)
            
            self.finished_request.emit()
            
        except Exception as e:
            self.error_occurred.emit(f"Request failed: {str(e)}")


class ROSSignalBridge(QObject):
    """Qt 信號橋接器"""
    image_signal = pyqtSignal(object)
    model_output_signal = pyqtSignal(str)
    point3d_signal = pyqtSignal(float, float, float)
    
    def __init__(self):
        super().__init__()


class ROS2Thread(QThread):
    """在獨立執行緒中運行 ROS2 spin"""
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.running = True

    def run(self):
        while self.running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def stop(self):
        self.running = False


class ImageLabel(QLabel):
    """可點擊的影像標籤"""
    clicked = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid #ccc;")
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 轉換顯示座標到原始影像座標
            x = int((event.x() - self.offset_x) / self.scale_x)
            y = int((event.y() - self.offset_y) / self.scale_y)
            self.clicked.emit(x, y)

    def set_image(self, cv_image):
        """設定並顯示 OpenCV 影像"""
        h, w = cv_image.shape[:2]
        
        # 轉換 BGR 到 RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # 轉換為 QImage
        bytes_per_line = 3 * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 縮放以符合標籤大小
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # 計算縮放比例和偏移
        self.scale_x = scaled_pixmap.width() / w
        self.scale_y = scaled_pixmap.height() / h
        self.offset_x = (self.width() - scaled_pixmap.width()) / 2
        self.offset_y = (self.height() - scaled_pixmap.height()) / 2
        
        self.setPixmap(scaled_pixmap)


class WhitePointGUI(Node):
    """ROS2 Node for White Point GUI"""
    
    def __init__(self, signal_bridge):
        super().__init__('white_point_gui')
        self.bridge = CvBridge()
        self.signal_bridge = signal_bridge

        # 訂閱顏色影像
        self.image_sub = self.create_subscription(
            Image,
            '/d435i/color/image_raw',
            self.image_callback,
            10
        )

        # 發布點擊像素
        self.pixel_pub = self.create_publisher(Point, '/white_point_pixel', 10)

        # 訂閱 3D 座標（base_link frame）
        self.point3d_sub = self.create_subscription(
            PointStamped,
            '/white_point_base',
            self.point3d_callback,
            10
        )

        # 訂閱模型輸出
        self.model_output_sub = self.create_subscription(
            String,
            '/model_output',
            self.model_output_callback,
            10
        )

        # 發布使用者輸入
        self.user_input_pub = self.create_publisher(String, '/user_input', 10)

        self.last_click = None  # (u, v)
        self.last_3d = None     # (x, y, z)

    # ----------------------------------------------------------
    # ROS2 Callbacks
    # ----------------------------------------------------------
    def image_callback(self, msg: Image):
        """接收影像並發送信號到 Qt GUI"""
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = rotate_img_90(cv_image)
        
        # 如果有點擊位置，繪製標記
        # if self.last_click is not None:
        #     u, v = self.last_click
        #     cv2.circle(cv_image, (u, v), 6, (255, 255, 255), -1)
        #     cv2.circle(cv_image, (u, v), 8, (0, 255, 0), 2)
            
        #     # 如果有 3D 座標，顯示文字
        #     if self.last_3d is not None:
        #         x, y, z = self.last_3d
        #         text = f"X={x:.2f}, Y={y:.2f}, Z={z:.2f}"
        #         cv2.putText(cv_image, text, (u + 15, v - 10),
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        self.signal_bridge.image_signal.emit(cv_image)

    def point3d_callback(self, msg: PointStamped):
        """接收 3D 座標"""
        self.last_3d = (msg.point.x, msg.point.y, msg.point.z)
        self.signal_bridge.point3d_signal.emit(msg.point.x, msg.point.y, msg.point.z)
        self.get_logger().info(
            f"3D updated: X={msg.point.x:.3f}, Y={msg.point.y:.3f}, Z={msg.point.z:.3f}"
        )

    def model_output_callback(self, msg: String):
        """接收模型輸出"""
        self.signal_bridge.model_output_signal.emit(msg.data)
        self.get_logger().info(f"Model output: {msg.data}")

    def publish_pixel(self, x: int, y: int):
        """發布點擊的像素座標"""
        self.last_click = (x, y)
        pt = Point()
        pt.x = float(x)
        pt.y = float(y)
        self.pixel_pub.publish(pt)
        self.get_logger().info(f'Clicked pixel: u={x}, v={y}')

    def publish_user_input(self, text: str):
        """發布使用者輸入"""
        msg = String()
        msg.data = text
        self.user_input_pub.publish(msg)
        self.get_logger().info(f"User input: {text}")


class MainWindow(QMainWindow):
    """主視窗"""
    def __init__(self, ros_node, signal_bridge, controller_url="http://10.0.0.1:11000", 
                 model_path="PME033541/vla2.7"):
        super().__init__()
        self.ros_node = ros_node
        self.signal_bridge = signal_bridge
        self.controller_url = controller_url
        self.model_path = model_path
        self.setWindowTitle("Vision–Language Robotics Control Panel")
        self.setGeometry(100, 100, 1200, 1000)
        
        # LLM 相關狀態
        self.llm_worker = None  # 每次請求時創建新的 worker
        self.current_image = None
        self.llm_points = []  # 儲存 LLM 解析出的點
        self.conversation_state = None  # 對話狀態（用於管理影像和提示）
        
        # 初始化 conversation state
        if default_conversation is not None:
            self.conversation_state = default_conversation.copy()
        
        # 語音辨識相關狀態
        self.voice_thread = None
        self.voice_capture_duration = 4.0
        self.voice_available, self.voice_unavailable_reason = voice_support_status()
        self.is_processing_voice = False
        
        self.setup_ui()
        self.connect_signals()
        
        # 檢查遠端 LLM 服務連接（延遲檢查以避免阻塞 UI）
        QTimer.singleShot(2000, self.check_service_status)

    def setup_ui(self):
        """設定 UI 元件"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # 左側：影像顯示
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.image_label = ImageLabel()
        self.image_label.setMinimumSize(400, 300)
        left_layout.addWidget(self.image_label)
        
        # 服務狀態顯示
        self.service_status = QLabel("Controller: Not Connected | Worker: Not Connected")
        self.service_status.setAlignment(Qt.AlignCenter)
        self.service_status.setFont(QFont("Arial", 12, QFont.Bold))
        self.service_status.setStyleSheet("padding: 8px; background-color: #f0f0f0; font-weight: bold;")
        left_layout.addWidget(self.service_status)
        
        main_layout.addWidget(left_widget, stretch=1)
        
        # 右側：文字輸入/輸出
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 標題
        title_label = QLabel("Model Interaction")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(title_label)
        
        # 輸出區域（白色背景）
        output_label = QLabel("Output:")
        output_label.setFont(QFont("Arial", 14, QFont.Bold))
        right_layout.addWidget(output_label)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Monospace", 28))
        self.output_text.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 2px solid #ccc;
                border-radius: 5px;
                padding: 8px;
            }
        """)
        right_layout.addWidget(self.output_text, stretch=3)
        
        # 輸入區域（白色背景）
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
            QLineEdit:focus {
                border: 2px solid #4CAF50;
            }
        """)
        right_layout.addWidget(self.input_field)
        
        # 發送按鈕
        self.send_button = QPushButton("Send")
        self.send_button.setFont(QFont("Arial", 13, QFont.Bold))
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        right_layout.addWidget(self.send_button)
        
        # 語音輸入按鈕
        self.voice_button = QPushButton("Voice Input")
        self.voice_button.setFont(QFont("Arial", 13, QFont.Bold))
        self.voice_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:pressed {
                background-color: #0869c7;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        right_layout.addWidget(self.voice_button)
        
        main_layout.addWidget(right_widget, stretch=1)

    def connect_signals(self):
        """連接信號與槽"""
        # ROS2 信號（透過 signal_bridge）
        self.signal_bridge.image_signal.connect(self.update_image)
        self.signal_bridge.model_output_signal.connect(self.add_model_output)
        self.signal_bridge.point3d_signal.connect(self.update_3d_coord)
        
        # Qt 信號
        self.image_label.clicked.connect(self.on_image_clicked)
        self.send_button.clicked.connect(self.send_user_input)
        self.input_field.returnPressed.connect(self.send_user_input)
        self.voice_button.clicked.connect(self.handle_voice_button)
        
        # 更新語音按鈕狀態
        self._update_voice_button_state()
        
        # LLM Worker 信號會在 send_to_llm 中動態連接（每次請求創建新 worker）

    def update_image(self, cv_image):
        """更新顯示的影像，並保存用於 LLM"""
        # 保存當前影像供 LLM 使用
        self.current_image = cv_image.copy()
        
        # 不在即時畫面上標註 LLM 的點，只顯示原始影像
        self.image_label.set_image(cv_image)

    def add_model_output(self, text: str):
        """添加模型輸出到文字區域"""
        self.output_text.append(f"<b style='color: #2196F3;'>Model:</b> {text}")
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum()
        )

    def update_3d_coord(self, x: float, y: float, z: float):
        """更新 3D 座標顯示（不顯示在 UI）"""
        # 只記錄到日誌，不更新 UI
        pass

    def on_image_clicked(self, x: int, y: int):
        """處理影像點擊事件"""
        self.ros_node.publish_pixel(x, y)
        # 不顯示點擊訊息

    def send_user_input(self):
        """發送使用者輸入並查詢 LLM"""
        text = self.input_field.text().strip()
        if not text:
            return
        
        # 顯示使用者輸入（不含隱藏的格式指示）
        self.output_text.append(f"<b style='color: #4CAF50;'>You:</b> {text}")
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum()
        )
        
        # 發布到 ROS2
        self.ros_node.publish_user_input(text)
        
        # 清空輸入欄位
        self.input_field.clear()
        
        # 取得當前影像（使用按下 Enter 時的影像）
        image = None
        if self.current_image is not None:
            try:
                # 轉換 BGR 到 RGB 並建立 PIL Image
                rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                image = PILImage.fromarray(rgb)
            except Exception as e:
                print(f"Failed to convert image: {e}", file=sys.stderr)
                return
        
        if image is None:
            self.output_text.append("<b style='color: #F44336;'>錯誤:</b> 沒有可用的影像")
            return
        
        # 添加隱藏的格式指示（不顯示給使用者）
        prompt_with_instruction = text + (
            " Your answer should be formatted as a list of tuples, "
            "i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the "
            "x and y coordinates of a point satisfying the conditions above. "
            "The coordinates should be between 0 and 1, indicating the "
            "normalized pixel locations of the points in the image."
        )
        
        # 送給 LLM 處理
        self.send_to_llm(prompt_with_instruction, image)
    
    def send_to_llm(self, text: str, image: PILImage.Image):
        """將文字和影像送到 LLM（使用 conversation_state 系統）"""
        try:
            if default_conversation is None:
                self.output_text.append("<b style='color: #F44336;'>錯誤:</b> conversation 模組未載入")
                return
            
            # 如果之前的 worker 還在運行，先停止它
            if self.llm_worker is not None and self.llm_worker.isRunning():
                self.llm_worker.wait()
            
            # 創建新的 LLM worker
            self.llm_worker = LLMWorkerThread(self.controller_url)
            
            # 連接信號
            self.llm_worker.response_received.connect(self.handle_llm_response)
            self.llm_worker.error_occurred.connect(self.handle_llm_error)
            self.llm_worker.finished_request.connect(self.llm_request_finished)
            self.llm_worker.controller_connected.connect(self.on_controller_connected)
            self.llm_worker.worker_processing.connect(self.on_worker_processing)
            
            # 更新狀態
            self.service_status.setText("Controller: Connecting... | Worker: Waiting...")
            
            # 建立 conversation state（像 vlservoing.py 一樣）
            self.conversation_state = default_conversation.copy()
            
            # 添加影像標記
            if '<image>' not in text:
                text = '<image>\n' + text
            
            # 建立內容：(text, image, mode)
            content = (text, image, 'Pad')  # 使用 Pad 模式
            
            # 添加使用者訊息
            self.conversation_state.append_message(self.conversation_state.roles[0], content)
            self.conversation_state.append_message(self.conversation_state.roles[1], None)
            
            # 決定對話模板（基於模型名稱）
            model_name = "vla2.7"  # 使用註冊在 Controller 的模型名稱
            template_name = 'vicuna_v1'  # Vicuna 模型使用 vicuna_v1 模板
            
            # 如果是新對話，使用適當的模板
            if len(self.conversation_state.messages) == 2:
                new_state = conv_templates[template_name].copy()
                new_state.append_message(new_state.roles[0], self.conversation_state.messages[-2][1])
                new_state.append_message(new_state.roles[1], None)
                self.conversation_state = new_state
            
            # 取得 prompt 和影像
            prompt = self.conversation_state.get_prompt()
            pil_images, images, transforms = self.conversation_state.get_images()
            
            # 準備請求資料
            request_data = {
                'model': model_name,
                'prompt': prompt,
                'temperature': 1.0,
                'top_p': 0.7,
                'max_new_tokens': 512,
                'stop': self.conversation_state.sep if self.conversation_state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else self.conversation_state.sep2,
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
    
    def handle_llm_response(self, response: str):
        """處理 LLM 回應（只顯示一次完整回應並附上視覺化圖片）"""
        # 顯示文字回應（移除多餘換行）
        clean_response = response.replace('\n', ' ').strip()
        self.output_text.append(f"<b style='color: #2196F3;'>LLM:</b> {clean_response}")
        
        # 解析座標
        vectors = find_vectors(response)
        vectors_2d = [vec for vec in vectors if len(vec) == 2]
        
        if vectors_2d and self.current_image is not None:
            # 轉換標準化座標到像素座標
            h, w = self.current_image.shape[:2]
            self.llm_points = []
            pixel_points = []
            
            for x, y in vectors_2d:
                if isinstance(x, float) and x <= 1:
                    px = int(x * w)
                    py = int(y * h)
                else:
                    px, py = int(x), int(y)
                self.llm_points.append((px, py))
                pixel_points.append((px, py))
            
            # 建立視覺化圖片
            try:
                # 轉換 BGR 到 RGB
                rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                pil_image = PILImage.fromarray(rgb_image)
                
                # 在圖片上繪製點
                annotated_image = visualize_2d(pil_image, pixel_points, [], scale=1.0)
                
                # 轉換回 cv2 格式並顯示
                annotated_cv = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
                
                # 轉換為 QPixmap 並顯示在文字區域
                h_img, w_img = annotated_cv.shape[:2]
                bytes_per_line = 3 * w_img
                rgb_for_qt = cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB)
                q_image = QImage(rgb_for_qt.data, w_img, h_img, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                
                # 縮放圖片以適應文字區域寬度
                max_width = self.output_text.width() - 40
                if pixmap.width() > max_width:
                    pixmap = pixmap.scaled(max_width, int(pixmap.height() * max_width / pixmap.width()), 
                                         Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                # 插入圖片到文字區域（減少前後空白）
                cursor = self.output_text.textCursor()
                cursor.movePosition(cursor.End)
                cursor.insertText("\n")  # 只加一個換行
                cursor.insertImage(pixmap.toImage())
                self.output_text.setTextCursor(cursor)
                
                # 不顯示「找到 X 個點」的訊息
                
            except Exception as e:
                print(f"Failed to create visualization: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
            
            # 如果只有一個點，自動發布
            if len(self.llm_points) == 1:
                px, py = self.llm_points[0]
                self.ros_node.publish_pixel(int(px), int(py))
    
    def on_controller_connected(self, worker_addr: str):
        """Controller 連接成功"""
        self.service_status.setText(f"Controller: ✓ Connected | Worker: {worker_addr}")
    
    def on_worker_processing(self):
        """Worker 開始處理"""
        current = self.service_status.text()
        if "Worker:" in current:
            parts = current.split("|")
            self.service_status.setText(f"{parts[0]}| Worker: Processing...")
    
    def handle_llm_error(self, error: str):
        """處理 LLM 錯誤"""
        self.output_text.append(f"<b style='color: #F44336;'>LLM Error:</b> {error}")
        self.service_status.setText("Controller: ✗ Error | Worker: ✗ Error")
        self.send_button.setEnabled(True)
        self.send_button.setText("Send")
    
    def llm_request_finished(self):
        """LLM 請求完成"""
        current = self.service_status.text()
        if "Controller:" in current:
            parts = current.split("|")
            self.service_status.setText(f"{parts[0]}| Worker: ✓ Done")
        self.send_button.setEnabled(True)
        self.send_button.setText("Send")
    
    def handle_voice_button(self):
        """處理語音按鈕點擊"""
        if not self.voice_available:
            self.output_text.append(f"<b style='color: #F44336;'>Voice input unavailable:</b> {self.voice_unavailable_reason}")
            return
        
        if self.is_processing_voice:
            return
        
        self.is_processing_voice = True
        self.voice_button.setEnabled(False)
        self.voice_button.setText("Recording...")
        
        # 創建新的語音辨識執行緒
        self.voice_thread = VoiceRecognitionThread(duration_sec=self.voice_capture_duration)
        self.voice_thread.result_ready.connect(self.on_voice_result)
        self.voice_thread.error.connect(self.on_voice_error)
        self.voice_thread.status_changed.connect(self.on_voice_status_changed)
        self.voice_thread.finished.connect(self.on_voice_finished)
        self.voice_thread.start()
    
    def on_voice_result(self, text: str):
        """處理語音辨識結果"""
        self.input_field.setText(text)
        # 移除語音輸入的輸出訊息,直接自動發送
        QTimer.singleShot(100, self.send_user_input)
    
    def on_voice_error(self, error_msg: str):
        """處理語音辨識錯誤"""
        self.output_text.append(f"<b style='color: #F44336;'>Voice Error:</b> {error_msg}")
    
    def on_voice_status_changed(self, status: str):
        """更新語音辨識狀態"""
        self.voice_button.setText(status)
    
    def on_voice_finished(self):
        """語音辨識完成"""
        self.is_processing_voice = False
        self.voice_thread = None
        self._update_voice_button_state()
    
    def _update_voice_button_state(self):
        """更新語音按鈕的啟用狀態和文字"""
        if not self.voice_available:
            self.voice_button.setEnabled(False)
            self.voice_button.setText("Voice Unavailable")
            self.voice_button.setToolTip(self.voice_unavailable_reason)
        else:
            self.voice_button.setEnabled(True)
            self.voice_button.setText("Voice Input")
            self.voice_button.setToolTip("Click to start voice input (about 4 sec)")

    def check_service_status(self):
        """檢查遠端 LLM 服務狀態"""
        try:
            # 使用 POST 請求獲取 worker 列表
            ret = requests.post(self.controller_url + "/list_models", json={}, timeout=2)
            if ret.status_code == 200:
                models = ret.json().get("models", [])
                if models:
                    self.service_status.setText(f"Controller: ✓ Connected | Worker: ✓ Ready ({len(models)} models)")
                    self.service_status.setStyleSheet("padding: 5px; background-color: #4CAF50; color: white; font-weight: bold;")
                    print(f"✓ Remote LLM service ready! Available models: {models}")
                else:
                    self.service_status.setText("Controller: ✓ Connected | Worker: ⏳ Loading...")
                    # 再次檢查
                    QTimer.singleShot(10000, self.check_service_status)
            else:
                self.service_status.setText("Controller: ✓ Connected | Worker: ✗ Error")
        except Exception as e:
            self.service_status.setText(f"Controller: ✗ Cannot Connect ({self.controller_url})")
            print(f"Cannot connect to remote LLM service: {e}")
            print(f"Please confirm Controller is running at {self.controller_url}")
            # 重試
            QTimer.singleShot(5000, self.check_service_status)

    def closeEvent(self, event):
        """關閉視窗時的處理"""
        self.ros_node.get_logger().info("Shutting down GUI...")
        event.accept()


def main(args=None):
    import argparse
    
    # 解析命令列參數
    parser = argparse.ArgumentParser(description="White Point GUI with LLM")
    parser.add_argument("--controller-url", type=str, default="http://10.0.0.1:11000",
                       help="LLM controller URL")
    parser.add_argument("--model-path", type=str, default="PME033541/vla2.7",
                       help="Model path to load in the GUI")
    parser.add_argument("--ros-args", nargs=argparse.REMAINDER, help="ROS arguments")
    
    # 分離 ROS 參數和自定義參數
    import sys
    custom_args = []
    ros_args_list = []
    if '--ros-args' in sys.argv:
        idx = sys.argv.index('--ros-args')
        custom_args = sys.argv[1:idx]
        ros_args_list = sys.argv[idx+1:]
    else:
        custom_args = sys.argv[1:]
    
    parsed_args = parser.parse_args(custom_args)
    
    # 初始化 ROS2
    rclpy.init(args=ros_args_list if ros_args_list else None)
    
    # 創建 Qt 應用（必須在主執行緒）
    app = QApplication([sys.argv[0]])
    
    # 創建信號橋接器
    signal_bridge = ROSSignalBridge()
    
    # 創建 ROS2 節點
    ros_node = WhitePointGUI(signal_bridge)
    
    # 創建主視窗（傳入控制器 URL 和模型路徑）
    window = MainWindow(ros_node, signal_bridge, 
                       controller_url=parsed_args.controller_url,
                       model_path=parsed_args.model_path)
    window.show()
    
    # 在獨立執行緒中運行 ROS2 spin
    ros_thread = ROS2Thread(ros_node)
    ros_thread.start()
    
    # 運行 Qt 事件迴圈
    exit_code = app.exec_()
    
    # 清理
    ros_thread.stop()
    ros_thread.wait()
    ros_node.destroy_node()
    rclpy.shutdown()
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
