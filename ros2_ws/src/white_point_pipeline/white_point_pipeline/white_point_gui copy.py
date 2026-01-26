#!/usr/bin/env python3
import sys
import os

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
                           model_path="wentao-yuan/robopoint-v1-vicuna-v1.5-13b", load_4bit=True):
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
    response_received = pyqtSignal(str)
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
            
            # Stream output
            response = requests.post(
                worker_addr + "/worker_generate_stream",
                headers={"User-Agent": "White Point GUI Client"},
                json=self.request_data,
                stream=True,
                timeout=30
            )
            
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if not chunk:
                    continue
                data = json.loads(chunk.decode())
                if data.get("error_code", 1) == 0:
                    # 移除 prompt 部分，只保留模型的回應
                    output = data["text"][len(self.request_data["prompt"]):].strip()
                    self.response_received.emit(output)
                else:
                    self.error_occurred.emit(f"Error: {data.get('text','')} (code: {data.get('error_code')})")
                    return
            
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
        if self.last_click is not None:
            u, v = self.last_click
            cv2.circle(cv_image, (u, v), 6, (255, 255, 255), -1)
            cv2.circle(cv_image, (u, v), 8, (0, 255, 0), 2)
            
            # 如果有 3D 座標，顯示文字
            if self.last_3d is not None:
                x, y, z = self.last_3d
                text = f"X={x:.2f}, Y={y:.2f}, Z={z:.2f}"
                cv2.putText(cv_image, text, (u + 15, v - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
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
                 model_path="wentao-yuan/robopoint-v1-vicuna-v1.5-13b"):
        super().__init__()
        self.ros_node = ros_node
        self.signal_bridge = signal_bridge
        self.controller_url = controller_url
        self.model_path = model_path
        self.setWindowTitle("White Point GUI - ROS2 + LLM")
        self.setGeometry(100, 100, 900, 600)
        
        # LLM 相關狀態
        self.llm_worker = None  # 每次請求時創建新的 worker
        self.current_image = None
        self.llm_points = []  # 儲存 LLM 解析出的點
        self.conversation_state = None  # 對話狀態（用於管理影像和提示）
        
        # 初始化 conversation state
        if default_conversation is not None:
            self.conversation_state = default_conversation.copy()
        
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
        self.service_status = QLabel("Controller: 未連接 | Worker: 未連接")
        self.service_status.setAlignment(Qt.AlignCenter)
        self.service_status.setStyleSheet("padding: 5px; background-color: #f0f0f0; font-weight: bold;")
        left_layout.addWidget(self.service_status)
        
        main_layout.addWidget(left_widget, stretch=1)
        
        # 右側：文字輸入/輸出
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 標題
        title_label = QLabel("Model Interaction")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(title_label)
        
        # 輸出區域（白色背景）
        output_label = QLabel("Output:")
        output_label.setFont(QFont("Arial", 10))
        right_layout.addWidget(output_label)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Monospace", 9))
        self.output_text.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 2px solid #ccc;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        right_layout.addWidget(self.output_text, stretch=3)
        
        # 輸入區域（白色背景）
        input_label = QLabel("Input:")
        input_label.setFont(QFont("Arial", 10))
        right_layout.addWidget(input_label)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("輸入訊息並按 Enter...")
        self.input_field.setFont(QFont("Arial", 10))
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: 2px solid #ccc;
                border-radius: 5px;
                padding: 8px;
            }
            QLineEdit:focus {
                border: 2px solid #4CAF50;
            }
        """)
        right_layout.addWidget(self.input_field)
        
        # 發送按鈕
        self.send_button = QPushButton("發送")
        self.send_button.setFont(QFont("Arial", 10))
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        right_layout.addWidget(self.send_button)
        
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
        
        # LLM Worker 信號會在 send_to_llm 中動態連接（每次請求創建新 worker）

    def update_image(self, cv_image):
        """更新顯示的影像，並保存用於 LLM"""
        # 保存當前影像供 LLM 使用
        self.current_image = cv_image.copy()
        
        # 如果有 LLM 解析的點，繪製在影像上
        if self.llm_points:
            for px, py in self.llm_points:
                cv2.circle(cv_image, (int(px), int(py)), 8, (255, 0, 0), 2)  # 藍色圓圈
                cv2.circle(cv_image, (int(px), int(py)), 3, (255, 255, 255), -1)  # 白色中心
        
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
            self.service_status.setText("Controller: 連接中... | Worker: 等待中...")
            
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
            model_name = "robopoint-v1-vicuna-v1.5-13b"  # 預設模型
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
            self.send_button.setText("處理中...")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.output_text.append(f"<b style='color: #F44336;'>Error:</b> {str(e)}")
            self.service_status.setText("Controller: 錯誤 | Worker: 錯誤")
    
    def handle_llm_response(self, response: str):
        """處理 LLM 回應"""
        # 顯示回應
        self.output_text.append(f"<b style='color: #2196F3;'>LLM:</b> {response}")
        
        # 解析座標
        vectors = find_vectors(response)
        vectors_2d = [vec for vec in vectors if len(vec) == 2]
        
        if vectors_2d:
            # 轉換標準化座標到像素座標
            if self.current_image is not None:
                h, w = self.current_image.shape[:2]
                self.llm_points = []
                for x, y in vectors_2d:
                    if isinstance(x, float) and x <= 1:
                        px = int(x * w)
                        py = int(y * h)
                    else:
                        px, py = int(x), int(y)
                    self.llm_points.append((px, py))
                
                self.output_text.append(
                    f"<b style='color: #9C27B0;'>Found {len(self.llm_points)} points</b>"
                )
                
                # 如果只有一個點，自動發布
                if len(self.llm_points) == 1:
                    px, py = self.llm_points[0]
                    self.ros_node.publish_pixel(int(px), int(py))
                    self.output_text.append(
                        f"<b style='color: #FF9800;'>Auto-selected:</b> ({px}, {py})"
                    )
    
    def on_controller_connected(self, worker_addr: str):
        """Controller 連接成功"""
        self.service_status.setText(f"Controller: ✓ 已連接 | Worker: {worker_addr}")
    
    def on_worker_processing(self):
        """Worker 開始處理"""
        current = self.service_status.text()
        if "Worker:" in current:
            parts = current.split("|")
            self.service_status.setText(f"{parts[0]}| Worker: 處理中...")
    
    def handle_llm_error(self, error: str):
        """處理 LLM 錯誤"""
        self.output_text.append(f"<b style='color: #F44336;'>LLM Error:</b> {error}")
        self.service_status.setText("Controller: ✗ 錯誤 | Worker: ✗ 錯誤")
        self.send_button.setEnabled(True)
        self.send_button.setText("發送")
    
    def llm_request_finished(self):
        """LLM 請求完成"""
        current = self.service_status.text()
        if "Controller:" in current:
            parts = current.split("|")
            self.service_status.setText(f"{parts[0]}| Worker: ✓ 完成")
        self.send_button.setEnabled(True)
        self.send_button.setText("發送")

    def check_service_status(self):
        """檢查遠端 LLM 服務狀態"""
        try:
            # 使用 POST 請求獲取 worker 列表
            ret = requests.post(self.controller_url + "/list_models", json={}, timeout=2)
            if ret.status_code == 200:
                models = ret.json().get("models", [])
                if models:
                    self.service_status.setText(f"Controller: ✓ 連接 | Worker: ✓ 就緒 ({len(models)} 模型)")
                    self.service_status.setStyleSheet("padding: 5px; background-color: #4CAF50; color: white; font-weight: bold;")
                    print(f"✓ 遠端 LLM 服務就緒！可用模型: {models}")
                else:
                    self.service_status.setText("Controller: ✓ 連接 | Worker: ⏳ 載入中...")
                    # 再次檢查
                    QTimer.singleShot(10000, self.check_service_status)
            else:
                self.service_status.setText("Controller: ✓ 連接 | Worker: ✗ 錯誤")
        except Exception as e:
            self.service_status.setText(f"Controller: ✗ 無法連接 ({self.controller_url})")
            print(f"無法連接到遠端 LLM 服務: {e}")
            print(f"請確認 Controller 在 {self.controller_url} 上運行")
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
    parser.add_argument("--model-path", type=str, default="wentao-yuan/robopoint-v1-vicuna-v1.5-13b",
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
