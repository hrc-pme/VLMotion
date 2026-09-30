#!/usr/bin/env python3
"""
PyQt-based GUI server for RoboPoint that replaces Gradio interface.
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

"""Qt plugin environment hardening
Avoid OpenCV's bundled Qt plugins taking precedence over system Qt plugins.
This must execute BEFORE importing PyQt5.
"""
try:
    # Always clear QT_PLUGIN_PATH; OpenCV wheels often set it to cv2/qt/plugins.
    os.environ.pop('QT_PLUGIN_PATH', None)

    # Force xcb unless user overrides.
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

    # Find a valid platforms dir and set QT_QPA_PLATFORM_PLUGIN_PATH
    candidates = [
        '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms',
        '/usr/lib/x86_64-linux-gnu/qt/plugins/platforms',
        '/usr/lib/qt/plugins/platforms',
    ]
    for p in candidates:
        if os.path.isdir(p):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = p
            # Also set plugin root for good measure
            root = os.path.dirname(p)
            os.environ['QT_PLUGIN_PATH'] = root
            break
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

# Import RoboPoint modules: add repo root (two levels up) to sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.append(repo_root)
from robopoint.conversation import default_conversation, conv_templates, SeparatorStyle
from robopoint.constants import LOGDIR
from robopoint.utils import build_logger, server_error_msg, violates_moderation, moderation_msg

# Networking for D405 stream (align with recv_and_yolo_d405_images)
import zmq
import stretch_visual_servoing.yolo_networking as yn
import yaml
from yaml.loader import SafeLoader


logger = build_logger("pyqt_server", "pyqt_server.log")

class YoloGraspProcess:
    """Manage the recv_and_yolo_d405_images process (local by default)."""
    def __init__(self):
        self.proc = None

    def start(self, use_remote: bool = False):
        if self.proc and self.proc.poll() is None:
            return  # already running
        cmd = [
            sys.executable, "-m",
            "stretch_visual_servoing.recv_and_yolo_d405_images"
        ]
        if use_remote:
            cmd.append("-r")
        logger.info(f"Starting YOLO Grasping: {' '.join(cmd)}")
        # NOTE: If your script needs args, append here, e.g.:
        # cmd += ["--remote-ip", yn.remote_computer_ip, "--d405-port", str(yn.d405_port)]
        # Ensure the module and its sibling imports are discoverable regardless of CWD.
        # - Set cwd to repo_root so `-m stretch_visual_servoing....` resolves
        # - Extend PYTHONPATH with both repo_root and the stretch_visual_servoing dir
        env = os.environ.copy()
        repo = repo_root
        pkg_dir = os.path.join(repo, 'stretch_visual_servoing')
        sep = os.pathsep
        existing = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f"{repo}{sep}{pkg_dir}{(sep + existing) if existing else ''}"
        # Run headless to avoid Qt HighGUI thread warnings from cv2/matplotlib
        env.setdefault('MPLBACKEND', 'Agg')
        env.setdefault('QT_QPA_PLATFORM', 'offscreen')
        # Ensure external YOLO module doesn't open any OpenCV windows
        env.setdefault('YOLO_DISPLAY', '0')
        env.setdefault('YOLO_DEBUG_RESULTS', '0')
        env.setdefault('YOLO_DEBUG_RECEIVED', '0')
        # Use the package dir as CWD so scripts that open local files
        # like 'aruco_marker_info.yaml' via relative paths can find them.
        self.proc = subprocess.Popen(cmd, cwd=pkg_dir, env=env)
        return self.proc

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None


class VisualServoingProcess:
    """Manage the visual_servoing_demo grasp routine as a subprocess."""
    def __init__(self):
        self.proc = None

    def start(self, use_remote: bool = False):
        if self.proc and self.proc.poll() is None:
            return
        # Run: python -m stretch_visual_servoing.visual_servoing_demo -y [ -r ]
        # Try to free any stale processes holding the Stretch robot first.
        try:
            subprocess.run(["stretch_free_robot_process.py"], timeout=5)
            time.sleep(0.5)
        except Exception:
            pass
        cmd = [
            sys.executable, "-m", "stretch_visual_servoing.visual_servoing_demo",
            "-y",
        ]
        if use_remote:
            cmd.append("-r")
        logger.info(f"Starting Visual Servoing Demo: {' '.join(cmd)}")

        env = os.environ.copy()
        repo = repo_root
        pkg_dir = os.path.join(repo, 'stretch_visual_servoing')
        sep = os.pathsep
        existing = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f"{repo}{sep}{pkg_dir}{(sep + existing) if existing else ''}"
        # Ensure headless safe if any OpenCV windows attempt to open
        env.setdefault('MPLBACKEND', 'Agg')
        env.setdefault('QT_QPA_PLATFORM', 'offscreen')
        # Also suppress YOLO display windows spawned from perception
        env.setdefault('YOLO_DISPLAY', '0')
        env.setdefault('YOLO_DEBUG_RESULTS', '0')
        env.setdefault('YOLO_DEBUG_RECEIVED', '0')
        # Start in package dir for relative resource opens
        self.proc = subprocess.Popen(cmd, cwd=pkg_dir, env=env)
        return self.proc

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None


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
            sys.executable, "-m", "robopoint.serve.controller",
            "--host", host, "--port", str(port)
        ]
        logger.info(f"Starting controller: {' '.join(cmd)}")
        self.controller_process = subprocess.Popen(cmd)
        # Remember effective controller URL for later API calls
        self.controller_url = f"http://{('localhost' if host in ['0.0.0.0', '::'] else host)}:{port}"
        
    def start_model_worker(self, host="0.0.0.0", controller_url="http://localhost:10000", 
                          port=20000, worker_url="http://localhost:20000",
                          model_path="wentao-yuan/robopoint-v1-vicuna-v1.5-13b", load_4bit=True):
        """Start the model worker process"""
        # Note: model_worker.py expects --controller-address and --worker-address
        cmd = [
            sys.executable, "-m", "robopoint.serve.model_worker",
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
                                   headers={"User-Agent": "RoboPoint Client"}, 
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
        self.setText("Waiting for D405 frames…")
        self.setAcceptDrops(False)
        self.image_path = None
        self.original_image = None
        self.enable_interaction = False  # disable click/drag interactions when receiving via network
        
    def mousePressEvent(self, event):
        if self.enable_interaction and event.button() == Qt.LeftButton:
            self.select_image()
            
    def dragEnterEvent(self, event):
        if self.enable_interaction and event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
            
    def dropEvent(self, event):
        if not self.enable_interaction:
            return
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


class D405ReceiverThread(QThread):
    """Background thread mirroring recv_and_yolo_d405_images subscription to receive D405 frames."""
    frame_received = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, use_remote=False):
        super().__init__()
        self.use_remote = use_remote
        self._running = True
        self._socket = None
        self._context = None

    def run(self):
        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.setsockopt(zmq.SUBSCRIBE, b'')
            self._socket.setsockopt(zmq.SNDHWM, 1)
            self._socket.setsockopt(zmq.RCVHWM, 1)
            self._socket.setsockopt(zmq.CONFLATE, 1)
            if self.use_remote:
                address = 'tcp://' + yn.robot_ip + ':' + str(yn.d405_port)
            else:
                address = 'tcp://' + '127.0.0.1' + ':' + str(yn.d405_port)
            self._socket.connect(address)
            self.status_changed.emit(f"Connected ({address})")

            while self._running:
                try:
                    d405_output = self._socket.recv_pyobj(flags=0)
                except zmq.error.ZMQError:
                    break
                if not self._running:
                    break
                if isinstance(d405_output, dict):
                    self.frame_received.emit(d405_output)
        except Exception as e:
            self.status_changed.emit(f"Error: {e}")
        finally:
            try:
                if self._socket is not None:
                    self._socket.close(0)
                if self._context is not None:
                    self._context.term()
            except Exception:
                pass
            self.status_changed.emit("Disconnected")

    def stop(self):
        self._running = False
        try:
            if self._socket is not None:
                self._socket.close(0)
        except Exception:
            pass


class YoloResultSubscriberThread(QThread):
    """Background thread subscribing to YOLO result PUB stream (yn.yolo_port).
    Mirrors SUB socket options used elsewhere: SUBSCRIBE '', HWM=1, CONFLATE=1.
    Emits the latest send_dict to the GUI thread.
    """

    result_received = pyqtSignal(dict)
    status_changed = pyqtSignal(str)

    def __init__(self, use_remote=True):
        super().__init__()
        self.use_remote = use_remote
        self._context = None
        self._socket = None
        self._running = True

    def run(self):
        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.setsockopt(zmq.SUBSCRIBE, b'')
            self._socket.setsockopt(zmq.SNDHWM, 1)
            self._socket.setsockopt(zmq.RCVHWM, 1)
            self._socket.setsockopt(zmq.CONFLATE, 1)

            # When using a remote machine for YOLO (off-robot GPU), connect to that IP.
            # Otherwise, subscribe locally.
            if self.use_remote:
                address = 'tcp://' + yn.remote_computer_ip + ':' + str(yn.yolo_port)
            else:
                address = 'tcp://' + '127.0.0.1' + ':' + str(yn.yolo_port)
            self._socket.connect(address)
            self.status_changed.emit(f"YOLO Connected ({address})")

            while self._running:
                try:
                    send_dict = self._socket.recv_pyobj(flags=0)
                except zmq.error.ZMQError:
                    break
                if not self._running:
                    break
                if isinstance(send_dict, dict):
                    self.result_received.emit(send_dict)
        except Exception as e:
            self.status_changed.emit(f"YOLO Error: {e}")
        finally:
            try:
                if self._socket is not None:
                    self._socket.close(0)
                if self._context is not None:
                    self._context.term()
            except Exception:
                pass
            self.status_changed.emit("YOLO Disconnected")

    def stop(self):
        self._running = False
        try:
            if self._socket is not None:
                self._socket.close(0)
        except Exception:
            pass

# 1) New: Inline YOLO inference thread (no external windows)
class YoloInlineThread(QThread):
    """Run YOLO perception inline by subscribing to D405 frames and emitting results.

    Mirrors recv_and_yolo_d405_images.py but returns results via Qt signals
    instead of publishing over ZMQ. Avoids any external GUI windows.
    """

    result_ready = pyqtSignal(dict)
    status_changed = pyqtSignal(str)

    def __init__(self, use_remote=True, parent=None):
        super().__init__(parent)
        self.use_remote = use_remote
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            import zmq  # local import to keep module load light
            # Import YOLO pipeline and networking via absolute paths
            import stretch_visual_servoing.yolo_servo_perception as yp
            import stretch_visual_servoing.yolo_networking as yn
            import stretch_visual_servoing.loop_timer as lt
            import cv2
            import os

            # Ensure any cv2.imshow calls in dependencies are no-ops
            try:
                cv2.imshow = lambda *args, **kwargs: None
                cv2.waitKey = lambda *args, **kwargs: -1
                cv2.namedWindow = lambda *args, **kwargs: None
                cv2.destroyAllWindows = lambda *args, **kwargs: None
            except Exception:
                pass

            # ZMQ SUB to D405 stream (aligned with existing networking options)
            ctx = zmq.Context()
            sub = ctx.socket(zmq.SUB)
            sub.setsockopt(zmq.SUBSCRIBE, b'')
            sub.setsockopt(zmq.SNDHWM, 1)
            sub.setsockopt(zmq.RCVHWM, 1)
            sub.setsockopt(zmq.CONFLATE, 1)
            if self.use_remote:
                d405_addr = 'tcp://' + yn.robot_ip + ':' + str(yn.d405_port)
                model_name = yn.yolo_model_on_remote_computer
            else:
                d405_addr = 'tcp://' + '127.0.0.1' + ':' + str(yn.d405_port)
                model_name = yn.yolo_model_on_robot
            sub.connect(d405_addr)
            self.status_changed.emit(f"YOLO Inline Connected ({d405_addr})")

            # Prepare YOLO perception module; ensure relative resources resolve
            # (yolo_servo_perception opens 'aruco_marker_info.yaml' relatively)
            cwd_prev = os.getcwd()
            try:
                pkg_dir = os.path.dirname(os.path.abspath(__file__))
                os.chdir(pkg_dir)
                ysp = yp.YoloServoPerception(model_name=model_name)
            finally:
                os.chdir(cwd_prev)

            loop = lt.LoopTimer()
            first_frame = True

            while self._running:
                loop.start_of_iteration()
                try:
                    d405_output = sub.recv_pyobj(flags=zmq.NOBLOCK)
                except zmq.Again:
                    self.msleep(1)
                    continue

                color_image = d405_output.get('color_image')
                depth_image = d405_output.get('depth_image')
                depth_camera_info = d405_output.get('depth_camera_info')
                depth_scale = d405_output.get('depth_scale')

                if first_frame and (depth_camera_info is not None) and (depth_scale is not None):
                    ysp.set_camera_parameters(depth_camera_info, depth_scale)
                    first_frame = False

                try:
                    send_dict = ysp.apply(color_image, depth_image)
                    self.result_ready.emit(send_dict)
                except Exception as e:
                    # Surface perception errors non-fatally to the UI
                    self.status_changed.emit(f"YOLO Inline Inference Error: {e}")

                loop.end_of_iteration()
                # Optional: loop.pretty_print()

        except Exception as e:
            self.status_changed.emit(f"YOLO Inline Error: {e}")
        finally:
            try:
                sub.close(0)
                ctx.term()
            except Exception:
                pass
            self.status_changed.emit("YOLO Inline Disconnected")

class RoboPointMainWindow(QMainWindow):
    """Main window for RoboPoint PyQt interface"""

    # Emitted when a tennis target is detected via YOLO results.
    # Payload is a dict with keys: 'pixel' (u,v), 'depth_m', 'camera_xyz'
    tennis_target_detected = pyqtSignal(dict)

    def __init__(self, controller_url=None, autostart=True, model_path="wentao-yuan/robopoint-v1-vicuna-v1.5-13b", load_4bit=True, use_remote_d405=False):
        super().__init__()
        self.setWindowTitle("RoboPoint - Vision-Language Model for Spatial Affordance Prediction")
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
        self.use_remote_d405 = use_remote_d405
        # Assume YOLO runs on the remote computer when D405 is remote.
        self.use_remote_yolo = use_remote_d405
        self.d405_thread = None
        self.yolo_thread = None
        # YOLO grasping external process
        self.yolo_grasp_proc = YoloGraspProcess()
        # Visual servoing process (auto-grasp routine)
        self.visual_servo_proc = VisualServoingProcess()
        # Inline YOLO thread (preferred)
        self.yolo_inline_thread = None

        # Perception state caches
        self.latest_fingertips = None
        self.latest_yolo_send_dict = None
        self.latest_camera_info = None
        self.latest_depth_scale = None
        self.latest_depth_image = None
        self.latest_llm_points = []  # list of (x,y) pixel coords from LLM annotations
        
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

        # Initialize ArUco detection pipeline (import locally to avoid cv2 import before Qt init)
        try:
            from stretch_visual_servoing import aruco_detector as ad
            from stretch_visual_servoing import aruco_to_fingertips as af
            from stretch_visual_servoing import d405_helpers_without_pyrealsense as dh
            self.dh = dh
            with open(os.path.join(os.path.dirname(__file__), 'aruco_marker_info.yaml')) as f:
                marker_info = yaml.load(f, Loader=SafeLoader)
        except Exception:
            # Fallback to current working dir if relative open fails
            try:
                with open('aruco_marker_info.yaml') as f:
                    marker_info = yaml.load(f, Loader=SafeLoader)
            except Exception:
                marker_info = {}
        self.aruco_detector = ad.ArucoDetector(marker_info=marker_info, show_debug_images=False, use_apriltag_refinement=False, brighten_images=False)
        self.fingertip_part = 'cup_top'
        self.aruco_to_fingertips = af.ArucoToFingertips(default_height_above_mounting_surface=af.suctioncup_height[self.fingertip_part])
        
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
        
        # D405 stream area (replaces upload/camera controls)
        image_group = QGroupBox("D405 Stream")
        image_layout = QVBoxLayout(image_group)
        self.image_label = ImageLabel()
        image_layout.addWidget(self.image_label)
        # Status label for stream
        self.d405_status = QLabel("Status: Disconnected")
        image_layout.addWidget(self.d405_status)

        # Overlay toggle for detections
        self.overlay_checkbox = QCheckBox("Overlay detections")
        self.overlay_checkbox.setChecked(True)
        image_layout.addWidget(self.overlay_checkbox)
        
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

        # YOLO Grasping control
        yolo_group = QGroupBox("YOLO Grasping")
        yolo_layout = QVBoxLayout(yolo_group)
        self.btn_start_yolo = QPushButton("Start YOLO Grasping")
        self.btn_stop_yolo = QPushButton("Stop YOLO Grasping")
        self.btn_stop_yolo.setEnabled(False)
        self.yolo_status = QLabel("YOLO Grasping: Not Running")
        # Option to let GUI manage the servoing demo process; off by default
        self.manage_demo_checkbox = QCheckBox("Manage Visual Servoing process")
        # Allow env override: YOLO_MANAGE_DEMO=1
        try:
            self.manage_demo_checkbox.setChecked(bool(int(os.getenv('YOLO_MANAGE_DEMO', '0'))))
        except Exception:
            self.manage_demo_checkbox.setChecked(False)
        yolo_layout.addWidget(self.btn_start_yolo)
        yolo_layout.addWidget(self.btn_stop_yolo)
        yolo_layout.addWidget(self.yolo_status)
        yolo_layout.addWidget(self.manage_demo_checkbox)
        left_layout.addWidget(yolo_group)
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
        
        # Start D405 receiver thread
        self.start_d405_receiver()
        # Start YOLO result subscriber
        self.start_yolo_subscriber()
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        # UI connections
        self.send_button.clicked.connect(self.send_message)
        self.text_input.returnPressed.connect(self.send_message)
        self.clear_button.clicked.connect(self.clear_chat)
        self.regenerate_button.clicked.connect(self.regenerate_response)

        # Service control
        self.start_services_btn.clicked.connect(self.start_services)
        self.stop_services_btn.clicked.connect(self.stop_services)

        # YOLO grasping buttons
        self.btn_start_yolo.clicked.connect(self.start_yolo_grasping)
        self.btn_stop_yolo.clicked.connect(self.stop_yolo_grasping)
        
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
        
    def start_d405_receiver(self):
        """Start background thread to receive D405 frames via ZMQ and update preview."""
        if self.d405_thread is not None:
            return
        self.d405_thread = D405ReceiverThread(use_remote=self.use_remote_d405)
        self.d405_thread.frame_received.connect(self.on_d405_frame)
        # Also feed frames to ArUco detection slot
        self.d405_thread.frame_received.connect(self.process_aruco_from_d405)
        self.d405_thread.status_changed.connect(self.on_d405_status)
        self.d405_thread.start()

    def start_yolo_grasping(self):
        """Start YOLO publisher and visual servoing grasp routine; show overlays in GUI."""
        try:
            # 1) Start YOLO publisher (recv_and_yolo_d405_images) to produce results on yn.yolo_port
            self.yolo_grasp_proc.start(use_remote=self.use_remote_yolo)
            # 2) Ensure our GUI subscriber is running (it is started in setup_ui)
            # 3) Optionally start visual servoing routine (exclusive control of robot)
            if self.manage_demo_checkbox.isChecked():
                self.visual_servo_proc.start(use_remote=self.use_remote_yolo)

            # UI updates
            self.btn_start_yolo.setEnabled(False)
            self.btn_stop_yolo.setEnabled(True)
            self.overlay_checkbox.setChecked(True)
            self.yolo_status.setText("YOLO Grasping: Running")
        except Exception as e:
            QMessageBox.critical(self, "YOLO Grasping", f"Failed to start: {e}")

    def stop_yolo_grasping(self):
        """Stop YOLO publisher and visual servoing routine; clear overlays."""
        try:
            # Stop external processes (only stop demo if we started/managing it)
            if self.manage_demo_checkbox.isChecked():
                self.visual_servo_proc.stop()
            self.yolo_grasp_proc.stop()
            # Clear latest YOLO results so overlays disappear
            self.latest_yolo_send_dict = None
        finally:
            self.btn_start_yolo.setEnabled(True)
            self.btn_stop_yolo.setEnabled(False)
            self.yolo_status.setText("YOLO Grasping: Not Running")

    def start_yolo_subscriber(self):
        """Start subscriber to YOLO results (send_dict)."""
        if self.yolo_thread is not None:
            return
        self.yolo_thread = YoloResultSubscriberThread(use_remote=self.use_remote_yolo)
        self.yolo_thread.result_received.connect(self.on_yolo_result)
        # Optional: could connect to a status label if added
        self.yolo_thread.start()

    def on_d405_status(self, text):
        self.d405_status.setText(f"Status: {text}")

    def on_d405_frame(self, d405_output):
        """Handle incoming D405 frame dict: display color image in the preview label."""
        try:
            color_image = d405_output.get('color_image', None)
            depth_image = d405_output.get('depth_image', None)
            if color_image is None:
                return
            # Cache camera parameters for downstream computations
            self.latest_camera_info = d405_output.get('depth_camera_info', d405_output.get('color_camera_info', None))
            self.latest_depth_scale = d405_output.get('depth_scale', None)
            self.latest_depth_image = depth_image

            # Prepare RGB image for overlay and display (input likely BGR)
            if isinstance(color_image, np.ndarray) and color_image.ndim == 3 and color_image.shape[2] == 3:
                import cv2
                rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            else:
                rgb = color_image if isinstance(color_image, np.ndarray) else np.array(color_image)

            vis = np.copy(rgb)

            if self.overlay_checkbox.isChecked():
                # Overlay ArUco fingertips if available
                if (self.latest_fingertips is not None) and (self.latest_camera_info is not None):
                    try:
                        # Draw directly on vis (RGB array); drawing uses OpenCV colorspace but fine for visualization
                        self.aruco_to_fingertips.draw_fingertip_frames(
                            self.latest_fingertips,
                            vis,
                            self.latest_camera_info,
                            axis_length_in_m=0.02,
                            draw_origins=True,
                            write_coordinates=True
                        )
                    except Exception:
                        pass

                # Overlay YOLO detection (prefer segmentation mask over bbox)
                if (self.latest_yolo_send_dict is not None) and (self.latest_camera_info is not None):
                    try:
                        yolo_list = self.latest_yolo_send_dict.get('yolo', [])
                        for det in yolo_list:
                            # If a polygon mask is provided, overlay it as a translucent region (white)
                            import cv2
                            poly = det.get('mask', None)
                            if poly is not None:
                                try:
                                    poly = np.array(poly, dtype=np.int32)
                                    mask_bin = np.zeros(vis.shape[:2], dtype=np.uint8)
                                    cv2.fillPoly(mask_bin, [poly], 255, lineType=cv2.LINE_AA)
                                    overlay_color = np.array([255, 255, 255], dtype=np.float32)
                                    alpha = 0.40
                                    region = vis[mask_bin == 255].astype(np.float32)
                                    blended = (1.0 - alpha) * region + alpha * overlay_color
                                    vis[mask_bin == 255] = np.clip(blended, 0, 255).astype(np.uint8)
                                    # Minimal info: draw center point and centered width/XYZ text above
                                    center_xyz = det.get('grasp_center_xyz', None)
                                    width_m = det.get('width_m', None)
                                    if center_xyz is not None and self.latest_camera_info is not None:
                                        try:
                                            center_xyz_np = np.array(center_xyz, dtype=np.float32)
                                            uv = self.dh.pixel_from_3d(center_xyz_np, self.latest_camera_info)
                                            cx, cy = int(uv[0]), int(uv[1])
                                            # Draw center point
                                            cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                                            # Prepare centered text lines
                                            lines = []
                                            if width_m is not None:
                                                lines.append(f"{width_m*100.0:.1f} cm wide")
                                            x_cm, y_cm, z_cm = center_xyz_np * 100.0
                                            lines.append(f"{x_cm:.1f}, {y_cm:.1f}, {z_cm:.1f} cm")
                                            # Anchor position slightly above the center point
                                            base_x, base_y = cx, cy - 55
                                            font = cv2.FONT_HERSHEY_SIMPLEX
                                            font_scale = 0.6
                                            thickness_fg = 1
                                            thickness_bg = 4
                                            for i, line in enumerate(lines):
                                                (tw, th), tb = cv2.getTextSize(line, font, font_scale, thickness_fg)
                                                lx = int(base_x - tw / 2)
                                                ly = int(base_y + i * (1.7 * th))
                                                cv2.putText(vis, line, (lx, ly), font, font_scale, (0,0,0), thickness_bg, cv2.LINE_AA)
                                                cv2.putText(vis, line, (lx, ly), font, font_scale, (255,255,255), thickness_fg, cv2.LINE_AA)
                                        except Exception:
                                            pass
                                    continue
                                except Exception:
                                    pass

                            # No mask available: skip drawing boxes/labels
                            # (User prefers masked visualization only.)
                            continue
                    except Exception:
                        pass

                # Overlay LLM central points with XYZ text (like tennis ball)
                if self.latest_llm_points and (self.latest_depth_image is not None) and (self.latest_camera_info is not None) and (self.latest_depth_scale is not None):
                    try:
                        import cv2
                        h, w = vis.shape[:2]
                        cx0, cy0 = w // 2, h // 2
                        # scale cached points (stored at 640x480) to current frame size
                        sx = w / 640.0
                        sy = h / 480.0
                        scaled_pts = [(int(round(px * sx)), int(round(py * sy))) for (px, py) in self.latest_llm_points]
                        # sort by distance to image center
                        dists = [(float(np.hypot(px - cx0, py - cy0)), (int(px), int(py))) for (px, py) in scaled_pts]
                        if dists:
                            dists.sort(key=lambda t: t[0])
                            min_d = dists[0][0]
                            selected = [pt for (d, pt) in dists if d <= min_d + 20.0]
                            selected = selected[:6]
                            for (px, py) in selected:
                                # Robust depth estimate using small neighborhood
                                k = 3
                                y0 = max(0, py - k)
                                y1 = min(self.latest_depth_image.shape[0], py + k + 1)
                                x0 = max(0, px - k)
                                x1 = min(self.latest_depth_image.shape[1], px + k + 1)
                                patch = self.latest_depth_image[y0:y1, x0:x1]
                                depth_m = None
                                if patch is not None and patch.size > 0:
                                    vals = patch.reshape(-1)
                                    # remove invalid
                                    if np.issubdtype(vals.dtype, np.floating):
                                        vals = vals[np.isfinite(vals)]
                                    vals = vals[vals > 0]
                                    if vals.size > 0:
                                        med = float(np.median(vals))
                                        depth_m = med if np.issubdtype(patch.dtype, np.floating) else med * float(self.latest_depth_scale)
                                if depth_m is None or depth_m <= 0:
                                    continue
                                center_xyz = self.dh.pixel_to_3d(np.array([px, py], dtype=np.float32), depth_m, self.latest_camera_info)
                                # draw center dot
                                cv2.circle(vis, (px, py), 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                                # centered text above
                                x_cm, y_cm, z_cm = (center_xyz * 100.0).astype(np.float32)
                                line = f"{x_cm:.1f}, {y_cm:.1f}, {z_cm:.1f} cm"
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.6
                                t_fg, t_bg = 1, 4
                                (tw, th), _ = cv2.getTextSize(line, font, font_scale, t_fg)
                                tx = int(px - tw / 2)
                                ty = int(py - 55)
                                cv2.putText(vis, line, (tx, ty), font, font_scale, (0, 0, 0), t_bg, cv2.LINE_AA)
                                cv2.putText(vis, line, (tx, ty), font, font_scale, (255, 255, 255), t_fg, cv2.LINE_AA)
                    except Exception:
                        pass

            # Convert to PIL and display
            pil = Image.fromarray(vis)
            self.image_label.load_pil(pil)
        except Exception as e:
            logger.error(f"Failed to handle D405 frame: {e}")

    def process_aruco_from_d405(self, d405_output):
        """Run ArUco detection from incoming D405 frame and cache fingertips for overlay."""
        try:
            color_image = d405_output.get('color_image', None)
            if color_image is None:
                return
            camera_info = d405_output.get('depth_camera_info', d405_output.get('color_camera_info', None))
            if camera_info is None:
                return
            # Convert to RGB for detection
            if isinstance(color_image, np.ndarray) and color_image.ndim == 3 and color_image.shape[2] == 3:
                import cv2
                rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            else:
                rgb = color_image if isinstance(color_image, np.ndarray) else np.array(color_image)

            # Update detector and cache fingertips
            self.aruco_detector.update(rgb, camera_info)
            markers = self.aruco_detector.get_detected_marker_dict()
            self.latest_fingertips = self.aruco_to_fingertips.get_fingertips(markers)
        except Exception as e:
            logger.error(f"AruCo processing error: {e}")

    def on_yolo_result(self, send_dict):
        """Receive latest YOLO results and optionally emit tennis target signal."""
        self.latest_yolo_send_dict = send_dict
        try:
            if self.latest_camera_info is None:
                return
            yolo_list = send_dict.get('yolo', [])
            if not yolo_list:
                return
            det = yolo_list[0]
            name = det.get('name', '')
            # Treat common variants as tennis ball
            if name in ['tennis', 'sports ball', '網球']:
                center_xyz = np.array(det.get('grasp_center_xyz', None)) if det.get('grasp_center_xyz', None) is not None else None
                if center_xyz is None:
                    return
                center_uv = self.dh.pixel_from_3d(center_xyz, self.latest_camera_info)
                depth_m = float(det.get('estimated_z_m', 0.0)) if det.get('estimated_z_m', None) is not None else 0.0
                payload = {
                    'pixel': (float(center_uv[0]), float(center_uv[1])),
                    'depth_m': depth_m,
                    'camera_xyz': (float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2]))
                }
                self.tennis_target_detected.emit(payload)
        except Exception:
            pass
        
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
                
            # Cache points for live overlay on D405 frames
            try:
                w_vis, h_vis = 640, 480
                pts = []
                for (x, y) in new_vectors:
                    xi = int(round(x))
                    yi = int(round(y))
                    if 0 <= xi < w_vis and 0 <= yi < h_vis:
                        pts.append((xi, yi))
                self.latest_llm_points = pts
            except Exception:
                pass

            # Visualize on chat image
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

    # Camera-related functions removed; replaced by D405 stream receiver
        
    def regenerate_response(self):
        """Regenerate last response"""
        if len(self.conversation_state.messages) >= 2:
            self.conversation_state.messages[-1][-1] = None
            self.send_request_to_worker()
            
    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_services()
        try:
            # Ensure YOLO grasping process is stopped
            self.visual_servo_proc.stop()
            self.yolo_grasp_proc.stop()
            if self.yolo_inline_thread is not None:
                self.yolo_inline_thread.stop()
                self.yolo_inline_thread.wait(1000)
            if self.d405_thread is not None:
                self.d405_thread.stop()
                self.d405_thread.wait(1000)
            if self.yolo_thread is not None:
                self.yolo_thread.stop()
                self.yolo_thread.wait(1000)
        except Exception:
            pass
        event.accept()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RoboPoint PyQt Server")
    parser.add_argument("--controller-url", type=str, default="http://localhost:10000",
                       help="Controller URL")
    parser.add_argument("--model-path", type=str, 
                       default="wentao-yuan/robopoint-v1-vicuna-v1.5-13b",
                       help="Model path")
    parser.add_argument("--load-4bit", action="store_true", default=True,
                       help="Load model in 4-bit mode")
    parser.add_argument('-r', '--remote', action='store_true', help='Use when receiving D405 images from a remote robot. Mirrors recv_and_yolo_d405_images.')
    
    args = parser.parse_args()
    
    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("RoboPoint")
    
    # Create and show main window
    window = RoboPointMainWindow(
        controller_url=args.controller_url,
        autostart=False,
        model_path=args.model_path,
        load_4bit=args.load_4bit,
        use_remote_d405=args.remote,
    )
    window.show()
    
    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
