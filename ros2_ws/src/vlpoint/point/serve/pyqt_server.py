#!/usr/bin/env python3
"""
PyQt-based GUI server for VLPoint that replaces Gradio interface.
This server integrates controller, model_worker, and GUI into one application.
"""

import sys
import os
import subprocess
import threading
import time
import argparse
import datetime
import hashlib
import json
import numpy as np
import re
import requests
from PIL import Image, ImageDraw, ImageQt

# Work around OpenCV's Qt plugin overriding PyQt5's platform plugins.
# Must run BEFORE importing PyQt5.
try:
    if 'QT_PLUGIN_PATH' in os.environ and 'cv2/qt/plugins' in os.environ.get('QT_PLUGIN_PATH', ''):
        os.environ.pop('QT_PLUGIN_PATH', None)
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
    qt5_platforms = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms'
    if os.path.isdir(qt5_platforms):
        os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', qt5_platforms)
except Exception:
    pass

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox, QSlider,
    QGroupBox, QScrollArea, QSplitter, QTabWidget, QSpinBox,
    QCheckBox, QFileDialog, QMessageBox, QProgressBar, QFrame, QDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QPixmap, QFont, QTextCursor, QPalette, QColor, QImage

# Import VLPoint modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from point.conversation import default_conversation, conv_templates, SeparatorStyle
from point.constants import LOGDIR
from point.utils import build_logger, server_error_msg, violates_moderation, moderation_msg


logger = build_logger("pyqt_server", "pyqt_server.log")


def pil_to_qimage(img: Image.Image) -> QImage:
    """Convert a PIL.Image to QImage.
    Tries Pillow's ImageQt.toqimage if available; otherwise falls back to manual conversion.
    Always returns a deep-copied QImage whose data is owned by Qt.
    """
    # Try Pillow's helper if present and functional
    try:
        if hasattr(ImageQt, "toqimage"):
            qimg = ImageQt.toqimage(img)
            # Some Pillow builds may return a proxy; ensure we own the data
            return qimg.copy()
    except Exception:
        pass

    # Fallback: manual conversion via bytes
    pil_img = img
    if pil_img.mode not in ("RGB", "RGBA"):
        pil_img = pil_img.convert("RGB")
    if pil_img.mode == "RGBA":
        data = pil_img.tobytes("raw", "RGBA")
        qimg = QImage(data, pil_img.width, pil_img.height, pil_img.width * 4, QImage.Format_RGBA8888)
    else:
        data = pil_img.tobytes("raw", "RGB")
        qimg = QImage(data, pil_img.width, pil_img.height, pil_img.width * 3, QImage.Format_RGB888)
    return qimg.copy()


def qpixmap_from_pil(img: Image.Image) -> QPixmap:
    """Convert a PIL.Image to QPixmap in a version-robust way."""
    return QPixmap.fromImage(pil_to_qimage(img))


class ServerProcess:
    """Manages controller and model worker processes"""
    
    def __init__(self):
        self.controller_process = None
        self.model_worker_process = None
        self.controller_url = "http://localhost:10000"
        
    def start_controller(self, host="0.0.0.0", port=10000):
        """Start the controller process"""
        cmd = [
            sys.executable, "-m", "point.serve.controller",
            "--host", host, "--port", str(port)
        ]
        logger.info(f"Starting controller: {' '.join(cmd)}")
        self.controller_process = subprocess.Popen(cmd)
        # Remember effective controller URL for later API calls
        self.controller_url = f"http://{('localhost' if host in ['0.0.0.0', '::'] else host)}:{port}"
        
    def start_model_worker(self, host="0.0.0.0", controller_url="http://localhost:10000", 
                          port=20000, worker_url="http://localhost:20000",
                          model_path="/workspace/models/vla13", load_4bit=True):
        """Start the model worker process"""
        # Note: model_worker.py expects --controller-address and --worker-address
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
            
        logger.info(f"Starting model worker: {' '.join(cmd)}")
        self.model_worker_process = subprocess.Popen(cmd)
        
    def stop_all(self):
        """Stop all processes"""
        if self.controller_process:
            self.controller_process.terminate()
            self.controller_process = None
        if self.model_worker_process:
            self.model_worker_process.terminate()
            self.model_worker_process = None


class ModelWorkerThread(QThread):
    """Thread for handling HTTP requests to model worker"""
    
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished_request = pyqtSignal()
    
    def __init__(self, controller_url="http://localhost:10000"):
        super().__init__()
        self.controller_url = controller_url
        self.request_data = None
        
    def set_request_data(self, data):
        self.request_data = data
        
    def run(self):
        try:
            # Get worker address
            ret = requests.post(self.controller_url + "/get_worker_address",
                              json={"model": self.request_data["model"]})
            worker_addr = ret.json()["address"]
            
            if worker_addr == "":
                self.error_occurred.emit("No available worker")
                return
                
            # Make request to worker
            response = requests.post(worker_addr + "/worker_generate_stream",
                                   headers={"User-Agent": "VLPoint Client"}, 
                                   json=self.request_data, stream=True, timeout=30)
                                   
            full_response = ""
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data["error_code"] == 0:
                        output = data["text"][len(self.request_data["prompt"]):].strip()
                        full_response = output
                        self.response_received.emit(output)
                    else:
                        self.error_occurred.emit(f"Error: {data['text']} (code: {data['error_code']})")
                        return
                        
            self.finished_request.emit()
            
        except Exception as e:
            self.error_occurred.emit(f"Request failed: {str(e)}")


class ImageLabel(QLabel):
    """Custom QLabel for displaying images with click handling"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 2px dashed #aaa;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("Click to upload image or drag & drop")
        self.setAcceptDrops(True)
        self.image_path = None
        self.original_image = None
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.select_image()
            
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
            
    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files and files[0].lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            self.load_image(files[0])
            
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_path:
            self.load_image(file_path)
            
    def load_image(self, path):
        try:
            self.image_path = path
            self.original_image = Image.open(path)
            
            # Convert PIL image to QPixmap and display (Pillow 11+ API)
            pixmap = qpixmap_from_pil(self.original_image)
            
            # Scale image to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load image: {str(e)}")
            
    def get_image(self):
        return self.original_image

    def load_pil(self, pil_image):
        """Load a PIL.Image directly into the label"""
        self.image_path = None
        self.original_image = pil_image
        pixmap = qpixmap_from_pil(self.original_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)


class CameraDialog(QDialog):
    """Simple camera preview dialog with capture support"""
    def __init__(self, parent=None, camera_index=0):
        super().__init__(parent)
        self.setWindowTitle("Camera Preview")
        self.resize(800, 600)
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.captured_frame = None

        # UI
        layout = QVBoxLayout(self)
        self.preview = QLabel("Starting camera...")
        self.preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview)

        btns = QHBoxLayout()
        self.capture_btn = QPushButton("Capture")
        self.close_btn = QPushButton("Close")
        btns.addWidget(self.capture_btn)
        btns.addWidget(self.close_btn)
        layout.addLayout(btns)

        self.capture_btn.clicked.connect(self._capture)
        self.close_btn.clicked.connect(self.reject)

        # Start camera
        self._open_camera(camera_index)

    def _open_camera(self, index):
        import cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", f"Failed to open camera index {index}")
            self.reject()
            return
        self.timer.start(30)

    def _update_frame(self):
        import cv2
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        pixmap = qpixmap_from_pil(Image.fromarray(frame_rgb))
        # Fit to label
        scaled = pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)
        self._last_frame_rgb = frame_rgb

    def _capture(self):
        # Keep the last frame for return
        if hasattr(self, '_last_frame_rgb'):
            self.captured_frame = self._last_frame_rgb.copy()
            self.accept()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        super().closeEvent(event)


class ChatWidget(QTextEdit):
    """Custom text widget for displaying chat history"""
    
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMinimumHeight(400)
        
        # Set styling
        font = QFont("Arial", 10)
        self.setFont(font)
        
    def add_message(self, sender, content, is_image=False):
        """Add a message to the chat"""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Add sender name
        cursor.insertText(f"\n{sender}:\n")
        
        if is_image and isinstance(content, tuple):
            # Handle image + text content
            text, image, _ = content
            cursor.insertText(f"{text}\n")
            
            # Convert PIL image to QImage and insert inline (robust across Pillow versions)
            if image is not None:
                qimage = pil_to_qimage(image)
                max_w = 600
                if qimage.width() > max_w:
                    qimage = qimage.scaled(max_w, int(qimage.height() * (max_w / qimage.width())), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                cursor.insertImage(qimage)
                cursor.insertText("\n")
        else:
            cursor.insertText(f"{content}\n")
            
        # Scroll to bottom
        self.moveCursor(QTextCursor.End)


class VLPointMainWindow(QMainWindow):
    """Main window for VLPoint PyQt interface"""

    def __init__(self, controller_url=None, autostart=True, model_path="/workspace/models/vla13", load_4bit=True):
        super().__init__()
        self.setWindowTitle("VLPoint - Vision-Language Model for Spatial Affordance Prediction")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize components
        self.server_process = ServerProcess()
        self.worker_thread = ModelWorkerThread()
        # Persist user preferences
        self.model_path = model_path
        self.load_4bit = load_4bit
        self.autostart = autostart
        self.conversation_state = default_conversation.copy()
        self.models = []
        self._pending_response_text = None
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        
        # Configure controller URL if provided (external controller mode)
        if controller_url:
            self.server_process.controller_url = controller_url
            self.worker_thread.controller_url = controller_url
        
        # Start background services or connect to existing controller
        if autostart:
            self.start_services()
        else:
            # Just connect and populate model list
            QTimer.singleShot(1000, self.refresh_model_list)
        
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        self.model_selector = QComboBox()
        model_layout.addWidget(self.model_selector)
        left_layout.addWidget(model_group)
        
        # Image upload area
        image_group = QGroupBox("Image Upload")
        image_layout = QVBoxLayout(image_group)
        self.image_label = ImageLabel()
        image_layout.addWidget(self.image_label)
        # Camera controls
        cam_ctrl_row = QHBoxLayout()
        self.camera_selector = QComboBox()
        self.refresh_cameras_btn = QPushButton("Refresh")
        cam_ctrl_row.addWidget(self.camera_selector, 1)
        cam_ctrl_row.addWidget(self.refresh_cameras_btn)
        image_layout.addLayout(cam_ctrl_row)
        # Open camera button
        self.camera_button = QPushButton("Open Camera…")
        image_layout.addWidget(self.camera_button)
        
        # Image processing mode
        self.image_mode_combo = QComboBox()
        self.image_mode_combo.addItems(["Crop", "Pad"])
        self.image_mode_combo.setCurrentText("Pad")
        image_layout.addWidget(QLabel("Image Processing:"))
        image_layout.addWidget(self.image_mode_combo)
        
        left_layout.addWidget(image_group)
        
        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)
        
        # Temperature
        params_layout.addWidget(QLabel("Temperature:"))
        self.temperature_slider = QSlider(Qt.Horizontal)
        self.temperature_slider.setRange(0, 100)
        self.temperature_slider.setValue(100)
        self.temperature_label = QLabel("1.0")
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(self.temperature_slider)
        temp_layout.addWidget(self.temperature_label)
        params_layout.addLayout(temp_layout)
        
        # Top P
        params_layout.addWidget(QLabel("Top P:"))
        self.top_p_slider = QSlider(Qt.Horizontal)
        self.top_p_slider.setRange(0, 100)
        self.top_p_slider.setValue(70)
        self.top_p_label = QLabel("0.7")
        top_p_layout = QHBoxLayout()
        top_p_layout.addWidget(self.top_p_slider)
        top_p_layout.addWidget(self.top_p_label)
        params_layout.addLayout(top_p_layout)
        
        # Max tokens
        params_layout.addWidget(QLabel("Max Output Tokens:"))
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(0, 1024)
        self.max_tokens_spin.setValue(512)
        params_layout.addWidget(self.max_tokens_spin)
        
        left_layout.addWidget(params_group)
        
        # Service status
        status_group = QGroupBox("Service Status")
        status_layout = QVBoxLayout(status_group)
        self.controller_status = QLabel("Controller: Not Started")
        self.worker_status = QLabel("Model Worker: Not Started")
        status_layout.addWidget(self.controller_status)
        status_layout.addWidget(self.worker_status)
        
        # Service control buttons
        self.start_services_btn = QPushButton("Start Services")
        self.stop_services_btn = QPushButton("Stop Services")
        status_layout.addWidget(self.start_services_btn)
        status_layout.addWidget(self.stop_services_btn)
        
        left_layout.addWidget(status_group)
        left_layout.addStretch()
        
        # Right panel for chat
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Chat area
        self.chat_widget = ChatWidget()
        right_layout.addWidget(self.chat_widget)
        
        # Input area
        input_layout = QHBoxLayout()
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter your question about the image...")
        self.send_button = QPushButton("Send")
        self.send_button.setEnabled(False)
        
        input_layout.addWidget(self.text_input)
        input_layout.addWidget(self.send_button)
        right_layout.addLayout(input_layout)
        
        # Action buttons
        action_layout = QHBoxLayout()
        self.clear_button = QPushButton("Clear Chat")
        self.regenerate_button = QPushButton("Regenerate")
        action_layout.addWidget(self.clear_button)
        action_layout.addWidget(self.regenerate_button)
        action_layout.addStretch()
        right_layout.addLayout(action_layout)
        
        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        main_layout.addWidget(splitter)

        # If running in connect-only mode, disable service control buttons
        if not self.autostart:
            self.start_services_btn.setEnabled(False)
            self.stop_services_btn.setEnabled(False)

        # Populate camera list initially
        self.refresh_cameras()
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        # UI connections
        self.send_button.clicked.connect(self.send_message)
        self.text_input.returnPressed.connect(self.send_message)
        self.clear_button.clicked.connect(self.clear_chat)
        self.regenerate_button.clicked.connect(self.regenerate_response)
        self.camera_button.clicked.connect(self.open_camera)
        self.refresh_cameras_btn.clicked.connect(self.refresh_cameras)
        
        # Service control
        self.start_services_btn.clicked.connect(self.start_services)
        self.stop_services_btn.clicked.connect(self.stop_services)
        
        # Parameter updates
        self.temperature_slider.valueChanged.connect(self.update_temperature)
        self.top_p_slider.valueChanged.connect(self.update_top_p)
        
        # Worker thread connections
        self.worker_thread.response_received.connect(self.handle_response)
        self.worker_thread.error_occurred.connect(self.handle_error)
        self.worker_thread.finished_request.connect(self.request_finished)
        
        # Timer for checking service status
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_service_status)
        self.status_timer.start(5000)  # Check every 5 seconds
        
    def start_services(self):
        """Start controller and model worker services"""
        try:
            self.controller_status.setText("Controller: Starting...")
            self.worker_status.setText("Model Worker: Starting...")
            
            # Start controller
            self.server_process.start_controller()
            time.sleep(3)  # Give controller time to start
            
            # Ensure worker thread points to the running controller URL
            self.worker_thread.controller_url = self.server_process.controller_url

            # Start model worker (use the detected controller URL)
            self.server_process.start_model_worker(
                controller_url=self.server_process.controller_url,
                model_path=self.model_path,
                load_4bit=self.load_4bit,
            )
            
            # Update UI
            self.start_services_btn.setEnabled(False)
            self.stop_services_btn.setEnabled(True)
            
            # Wait a bit then refresh model list
            QTimer.singleShot(10000, self.refresh_model_list)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start services: {str(e)}")
            
    def stop_services(self):
        """Stop all services"""
        self.server_process.stop_all()
        self.controller_status.setText("Controller: Stopped")
        self.worker_status.setText("Model Worker: Stopped")
        self.start_services_btn.setEnabled(True)
        self.stop_services_btn.setEnabled(False)
        self.send_button.setEnabled(False)
        
    def check_service_status(self):
        """Check if services are running"""
        try:
            # Check controller by listing models
            response = requests.post(f"{self.server_process.controller_url}/list_models", timeout=2)
            if response.status_code == 200:
                self.controller_status.setText("Controller: Running")
            else:
                self.controller_status.setText("Controller: Error")
        except:
            if self.server_process.controller_process and self.server_process.controller_process.poll() is None:
                self.controller_status.setText("Controller: Starting...")
            else:
                self.controller_status.setText("Controller: Not Running")
                
        # Enable/disable send button based on service status
        services_running = "Running" in self.controller_status.text()
        self.send_button.setEnabled(services_running and len(self.models) > 0)
        
    def refresh_model_list(self):
        """Refresh the list of available models"""
        try:
            base = self.server_process.controller_url
            ret = requests.post(f"{base}/refresh_all_workers", timeout=5)
            if ret.status_code == 200:
                ret = requests.post(f"{base}/list_models", timeout=5)
                if ret.status_code == 200:
                    self.models = ret.json()["models"]
                    self.model_selector.clear()
                    self.model_selector.addItems(self.models)
                    if self.models:
                        self.worker_status.setText("Model Worker: Running")
                        logger.info(f"Loaded models: {self.models}")
                    else:
                        self.worker_status.setText("Model Worker: No Models")
        except Exception as e:
            logger.error(f"Failed to refresh model list: {e}")
            
    def update_temperature(self, value):
        """Update temperature display"""
        temp = value / 100.0
        self.temperature_label.setText(f"{temp:.1f}")
        
    def update_top_p(self, value):
        """Update top_p display"""
        top_p = value / 100.0
        self.top_p_label.setText(f"{top_p:.1f}")
        
    def send_message(self):
        """Send message to model"""
        text = self.text_input.text().strip()
        image = self.image_label.get_image()
        
        if not text and not image:
            return
            
        if not self.models:
            QMessageBox.warning(self, "Warning", "No models available. Please wait for services to start.")
            return
            
        # Add prompt suffix
        text += " Your answer should be formatted as a list of tuples, " \
                "i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the " \
                "x and y coordinates of a point satisfying the conditions above." \
                " The coordinates should be between 0 and 1, indicating the " \
                "normalized pixel locations of the points in the image."
        
        # Prepare conversation state
        if image:
            if '<image>' not in text:
                text = '<image>\n' + text
            content = (text, image, self.image_mode_combo.currentText())
            self.conversation_state = default_conversation.copy()
        else:
            content = text
            
        self.conversation_state.append_message(self.conversation_state.roles[0], content)
        self.conversation_state.append_message(self.conversation_state.roles[1], None)
        
        # Add user message to chat
        self.chat_widget.add_message("User", text)
        
        # Clear input
        self.text_input.clear()
        
        # Disable send button during processing
        self.send_button.setEnabled(False)
        self.send_button.setText("Processing...")
        
        # Send request
        self.send_request_to_worker()
        
    def send_request_to_worker(self):
        """Send request to model worker"""
        try:
            # Determine conversation template
            model_name = self.model_selector.currentText()
            if 'vicuna' in model_name.lower():
                template_name = "vicuna_v1"
            elif "llama" in model_name.lower():
                template_name = "llava_llama_2"
            elif "mistral" in model_name.lower():
                template_name = "mistral_instruct"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                template_name = "llava_v1"
                
            if len(self.conversation_state.messages) == 2:
                new_state = conv_templates[template_name].copy()
                new_state.append_message(new_state.roles[0], self.conversation_state.messages[-2][1])
                new_state.append_message(new_state.roles[1], None)
                self.conversation_state = new_state
                
            prompt = self.conversation_state.get_prompt()
            pil_images, images, transforms = self.conversation_state.get_images()
            
            # Prepare request data
            request_data = {
                "model": model_name,
                "prompt": prompt,
                "temperature": self.temperature_slider.value() / 100.0,
                "top_p": self.top_p_slider.value() / 100.0,
                "max_new_tokens": min(self.max_tokens_spin.value(), 1536),
                "stop": self.conversation_state.sep if self.conversation_state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else self.conversation_state.sep2,
                "images": images,
            }
            
            # Start worker thread
            self.worker_thread.set_request_data(request_data)
            self.worker_thread.start()
            
        except Exception as e:
            self.handle_error(f"Failed to send request: {str(e)}")
            
    def handle_response(self, response):
        """Buffer streaming response; render once when finished"""
        self._pending_response_text = response
            
    def process_response_visualization(self, response):
        """Process response to add visualizations"""
        try:
            # Find vectors in response
            vectors = self.find_vectors(response)
            vectors_2d = [vec for vec in vectors if len(vec) == 2]
            vectors_bbox = [vec for vec in vectors if len(vec) == 4]
            
            if not vectors_2d and not vectors_bbox:
                return response
                
            # Get the image and transform
            pil_images, images, transforms = self.conversation_state.get_images()
            if not pil_images:
                return response
                
            image = pil_images[-1].copy()
            transform = transforms[-1]
            
            # Transform coordinates
            ref_w, ref_h = 640, 480
            
            # Process 2D points
            new_vectors = []
            for x, y in vectors_2d:
                if isinstance(x, float) and x <= 1:
                    x = x * ref_w
                    y = y * ref_h
                x, y, _ = (transform @ np.array([[x], [y], [1]])).ravel()
                new_vectors.append((x, y))
                
            # Process bounding boxes
            new_bbox = []
            for x1, y1, x2, y2 in vectors_bbox:
                if isinstance(x1, float) and x1 <= 1:
                    x1, y1, x2, y2 = x1 * ref_w, y1 * ref_h, x2 * ref_w, y2 * ref_h
                x1, y1, _ = (transform @ np.array([[x1], [y1], [1]])).ravel()
                x2, y2, _ = (transform @ np.array([[x2], [y2], [1]])).ravel()
                new_bbox.append((x1, y1, x2, y2))
                
            # Visualize on image
            annotated_image = self.visualize_2d(image, new_vectors, new_bbox, transform[0][0])
            
            return (response, annotated_image, self.image_mode_combo.currentText())
            
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return response
            
    def find_vectors(self, text):
        """Find coordinate vectors in text"""
        pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
        matches = re.findall(pattern, text)
        
        vectors = []
        for match in matches:
            vector = [float(num) if '.' in num else int(num) for num in match.split(',')]
            vectors.append(vector)
            
        return vectors
        
    def visualize_2d(self, img, points, bounding_boxes, scale, cross_size=9, cross_width=4):
        """Visualize points and bounding boxes on image"""
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
            
        draw = ImageDraw.Draw(img)
        size = int(cross_size * scale)
        width = int(cross_width * scale)
        
        # Draw points as red X
        for x, y in points:
            draw.line((x - size, y - size, x + size, y + size), fill='red', width=width)
            draw.line((x - size, y + size, x + size, y - size), fill='red', width=width)
            
        # Draw bounding boxes
        for x1, y1, x2, y2 in bounding_boxes:
            draw.rectangle([x1, y1, x2, y2], outline='red', width=width)
            
        img = img.convert('RGB')
        return img
        
    def handle_error(self, error_message):
        """Handle error from model worker"""
        self.chat_widget.add_message("System", f"Error: {error_message}")
        self._pending_response_text = None
        self.request_finished()
        
    def request_finished(self):
        """Called when request is finished"""
        if self._pending_response_text is not None:
            # Update conversation state and render final output once
            self.conversation_state.messages[-1] = [self.conversation_state.roles[1], self._pending_response_text]
            processed_response = self.process_response_visualization(self._pending_response_text)
            if isinstance(processed_response, tuple):
                self.chat_widget.add_message("Assistant", processed_response, is_image=True)
            else:
                self.chat_widget.add_message("Assistant", processed_response)
            self._pending_response_text = None

        self.send_button.setEnabled(True)
        self.send_button.setText("Send")
        
    def clear_chat(self):
        """Clear chat history"""
        self.chat_widget.clear()
        self.conversation_state = default_conversation.copy()

    def detect_cameras(self, max_index=12):
        """Return first available camera index, or None"""
        import cv2
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                return i
        return None

    def refresh_cameras(self, max_index=12):
        """Enumerate available /dev/video* devices and fill the selector."""
        import cv2
        self.camera_selector.clear()
        found = False
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                # Try to read a friendly name from sysfs
                name = None
                try:
                    sysfs = f"/sys/class/video4linux/video{i}/name"
                    if os.path.exists(sysfs):
                        with open(sysfs, 'r') as f:
                            name = f.read().strip()
                except Exception:
                    pass
                label = f"/dev/video{i}" + (f" - {name}" if name else "")
                self.camera_selector.addItem(label, i)
                found = True
        if not found:
            self.camera_selector.addItem("No camera found", -1)
        else:
            self.camera_selector.setCurrentIndex(0)

    def open_camera(self):
        """Open a simple camera preview and capture one frame into the image area"""
        # Prefer selected index from the combo box
        idx = self.camera_selector.currentData()
        if idx is None or idx == -1:
            idx = self.detect_cameras()
        if idx is None:
            QMessageBox.warning(self, "Camera", "No available camera detected.")
            return
        dlg = CameraDialog(self, camera_index=idx)
        if dlg.exec_() == QDialog.Accepted and dlg.captured_frame is not None:
            # Convert numpy RGB -> PIL
            pil = Image.fromarray(dlg.captured_frame)
            self.image_label.load_pil(pil)
        
    def regenerate_response(self):
        """Regenerate last response"""
        if len(self.conversation_state.messages) >= 2:
            self.conversation_state.messages[-1][-1] = None
            self.send_request_to_worker()
            
    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_services()
        event.accept()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="VLPoint PyQt Server")
    parser.add_argument("--controller-url", type=str, default="http://localhost:10000",
                       help="Controller URL")
    parser.add_argument("--model-path", type=str, 
                       default="/workspace/models/vla13",
                       help="Model path")
    parser.add_argument("--load-4bit", action="store_true", default=True,
                       help="Load model in 4-bit mode")
    
    args = parser.parse_args()
    
    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("VLPoint")
    
    # Create and show main window
    window = VLPointMainWindow(
        controller_url=args.controller_url,
        autostart=False,
        model_path=args.model_path,
        load_4bit=args.load_4bit,
    )
    window.show()
    
    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
