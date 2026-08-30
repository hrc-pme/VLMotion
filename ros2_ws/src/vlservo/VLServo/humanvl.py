#!/usr/bin/env python3
"""
RoboPoint Visual Servoing GUI

This combines the functionality of recv_and_robopoint_d405_images and
recv_and_robopoint_d435i_images, and adds a GUI control to start either
send_d405_images or send_d435i_images directly from the app.
"""

import sys
import os
import subprocess
import time
import argparse
import json
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageQt
import cv2
try:
    import stretch_body.robot as _rb
except Exception:
    _rb = None

# Qt plugin environment hardening (must run before importing PyQt5)
try:
    os.environ.pop('QT_PLUGIN_PATH', None)
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
    for p in [
        '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms',
        '/usr/lib/x86_64-linux-gnu/qt/plugins/platforms',
        '/usr/lib/qt/plugins/platforms',
    ]:
        if os.path.isdir(p):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = p
            root = os.path.dirname(p)
            os.environ['QT_PLUGIN_PATH'] = root
            break
except Exception:
    pass

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox, QSlider,
    QGroupBox, QSplitter, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QFont, QTextCursor, QImage

# Import RoboPoint modules: add repo root to sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.append(repo_root)
# Use local VLPoint package instead of RoboPoint
from point.conversation import default_conversation, conv_templates, SeparatorStyle
from point.utils import build_logger

import zmq
# Use local package networking settings
from . import yolo_networking as yn
import yaml
from yaml.loader import SafeLoader

logger = build_logger("pyqt_server", "pyqt_server.log")

# Prefer custom pose utils (unified homing, head/gripper lock); fallback to default
try:
    from VLServo.pose_utils_custom import (
        go_to_start_pose as go_to_start_pose_custom,
        set_head_tilt_deg as set_head_tilt_deg_custom,
        point_head_to_gripper as point_head_to_gripper_custom,
        lock_head_and_gripper as lock_head_and_gripper_custom,
        unlock_head_and_gripper as unlock_head_and_gripper_custom,
        nudge_head_pan_tilt as nudge_head_pan_tilt_custom,
        get_head_pan_tilt as get_head_pan_tilt_custom,
        get_gripper_forward_xyz as get_gripper_forward_xyz_custom,
        TILT_BIAS as HEAD_TILT_BIAS_DEFAULT,
    )
except Exception:
    try:
        from VLServo.pose_utils import (
            go_to_start_pose as go_to_start_pose_custom,
            set_head_tilt_deg as set_head_tilt_deg_custom,
        )
        point_head_to_gripper_custom = None
        lock_head_and_gripper_custom = None
        unlock_head_and_gripper_custom = None
        nudge_head_pan_tilt_custom = None
        get_head_pan_tilt_custom = None
        get_gripper_forward_xyz_custom = None
        HEAD_TILT_BIAS_DEFAULT = 0.0
    except Exception:
        go_to_start_pose_custom = None
        set_head_tilt_deg_custom = None
        point_head_to_gripper_custom = None
        lock_head_and_gripper_custom = None
        unlock_head_and_gripper_custom = None
        nudge_head_pan_tilt_custom = None
        get_head_pan_tilt_custom = None
        get_gripper_forward_xyz_custom = None
        HEAD_TILT_BIAS_DEFAULT = 0.0


class YoloGraspProcess:
    """Manage the recv_and_yolo_d405_images process (local by default)."""
    def __init__(self):
        self.proc = None

    def start(self, use_remote: bool = False):
        if self.proc and self.proc.poll() is None:
            return  # already running
        # Prefer the legacy receiver if available; otherwise fall back to sending D405 frames
        # so a remote YOLO can process them.
        env = os.environ.copy()
        sep = os.pathsep
        existing = env.get('PYTHONPATH', '')
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(pkg_dir)
        env['PYTHONPATH'] = f"{parent_dir}{sep}{pkg_dir}{(sep + existing) if existing else ''}"
        env.setdefault('MPLBACKEND', 'Agg')
        env.setdefault('QT_QPA_PLATFORM', 'offscreen')
        env.setdefault('YOLO_DISPLAY', '0')
        env.setdefault('YOLO_DEBUG_RESULTS', '0')
        env.setdefault('YOLO_DEBUG_RECEIVED', '0')

        # Try to run a local YOLO receiver if present
        try:
            import importlib.util as _il
            if _il.find_spec('VLServo.recv_and_yolo_d405_images') is not None:
                cmd = [sys.executable, "-m", "VLServo.recv_and_yolo_d405_images"]
            else:
                # Fallback: start D405 sender, typically used with remote YOLO
                cmd = [sys.executable, "-m", "VLServo.send_d405_images"]
        except Exception:
            cmd = [sys.executable, "-m", "VLServo.send_d405_images"]

        if use_remote:
            cmd.append("-r")
        logger.info(f"Starting YOLO Grasping: {' '.join(cmd)}")
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
        try:
            subprocess.run(["stretch_free_robot_process.py"], timeout=5)
            # Give the OS a moment to release serial ports
            time.sleep(0.5)
        except Exception:
            pass
        # Preflight: if running locally, verify ports are available to avoid noisy crashes
        if (not use_remote) and (_rb is not None):
            try:
                r = _rb.Robot()
                ok = False
                for _ in range(2):
                    try:
                        r.startup()
                        ok = True
                        break
                    except Exception:
                        time.sleep(0.5)
                try:
                    r.stop()
                except Exception:
                    pass
                if not ok:
                    raise RuntimeError("Robot serial ports are busy. Close other Stretch Body processes and try Reset/Calibrate.")
            except Exception as e:
                # Bubble up to caller: GUI will show an error dialog
                raise
        cmd = None
        # Try to run a demo reset routine if available; otherwise skip
        try:
            import importlib.util as _il
            # Prefer LLM-friendly demo wrapper if available; fall back to original
            if _il.find_spec('VLServo.visual_servoing_demo_llm') is not None:
                cmd = [sys.executable, "-m", "VLServo.visual_servoing_demo_llm", "-y"]
            elif _il.find_spec('VLServo.visual_servoing_demo') is not None:
                cmd = [sys.executable, "-m", "VLServo.visual_servoing_demo", "-y"]
        except Exception:
            cmd = None
        if use_remote:
            if cmd:
                cmd.append("-r")
        if not cmd:
            logger.info("No visual_servoing_demo available; skipping demo process.")
            return None
        logger.info(f"Starting Visual Servoing Demo: {' '.join(cmd)}")

        env = os.environ.copy()
        sep = os.pathsep
        existing = env.get('PYTHONPATH', '')
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(pkg_dir)
        env['PYTHONPATH'] = f"{parent_dir}{sep}{pkg_dir}{(sep + existing) if existing else ''}"
        env.setdefault('MPLBACKEND', 'Agg')
        env.setdefault('QT_QPA_PLATFORM', 'offscreen')
        env.setdefault('YOLO_DISPLAY', '0')
        env.setdefault('YOLO_DEBUG_RESULTS', '0')
        env.setdefault('YOLO_DEBUG_RECEIVED', '0')
        # Force: do NOT home inside the demo wrapper to avoid port contention,
        # regardless of any inherited environment variable.
        # Users can home explicitly via the GUI Reset/Calibrate button.
        env['VLSERVO_ALLOW_HOME'] = '0'
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


class WhitePointGraspProcess:
    """Manage the white_point publisher process (local by default)."""
    def __init__(self):
        self.proc = None

    def start(self, *args, **kwargs):
        # Accept both positional and keyword for robustness across callers
        if self.proc and self.proc.poll() is None:
            return  # already running
        # Parse args
        if args and len(args) >= 2:
            x, y = int(args[0]), int(args[1])
            use_remote = bool(args[2]) if len(args) >= 3 else False
            radius_m = float(args[3]) if len(args) >= 4 else 0.0
            extra_push_m = float(args[4]) if len(args) >= 5 else None
        else:
            x = int(kwargs.get('x'))
            y = int(kwargs.get('y'))
            use_remote = bool(kwargs.get('use_remote', False))
            radius_m = float(kwargs.get('radius_m', 0.0))
            extra_push_m = kwargs.get('extra_push_m', None)
        cmd = [
            sys.executable, "-m",
            # Use D435i-based white-point publisher
            "VLServo.white_point_d435i",
            "-x", str(x), "-y", str(y),
        ]
        if radius_m and radius_m > 0:
            cmd += ["--radius-m", str(radius_m)]
        if extra_push_m is not None:
            try:
                epm = float(extra_push_m)
                if epm >= 0.0:
                    cmd += ["--extra-push-m", str(epm)]
            except Exception:
                pass
        if use_remote:
            cmd.append("-r")
        logger.info(f"Starting LLM (White-Point) Grasping: {' '.join(cmd)}")
        env = os.environ.copy()
        sep = os.pathsep
        existing = env.get('PYTHONPATH', '')
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(pkg_dir)
        env['PYTHONPATH'] = f"{parent_dir}{sep}{pkg_dir}{(sep + existing) if existing else ''}"
        env.setdefault('MPLBACKEND', 'Agg')
        env.setdefault('QT_QPA_PLATFORM', 'offscreen')
        env.setdefault('YOLO_DISPLAY', '0')
        env.setdefault('YOLO_DEBUG_RESULTS', '0')
        env.setdefault('YOLO_DEBUG_RECEIVED', '0')
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


class BaseNavigationProcess:
    """Manage the base_motion navigation process (local by default)."""
    def __init__(self):
        self.proc = None

    def start(self, *args, **kwargs):
        if self.proc and self.proc.poll() is None:
            return  # already running
        # Parse args (x, y, use_remote, stop_dist_m, tilt_only, tilt_down_negative, invert_yaw)
        if args and len(args) >= 2:
            x, y = int(args[0]), int(args[1])
            use_remote = bool(args[2]) if len(args) >= 3 else False
            stop_dist_m = float(args[3]) if len(args) >= 4 else 0.5
            tilt_only = bool(args[4]) if len(args) >= 5 else True
            tilt_down_negative = bool(args[5]) if len(args) >= 6 else True
            invert_yaw = bool(args[6]) if len(args) >= 7 else True
        else:
            x = int(kwargs.get('x'))
            y = int(kwargs.get('y'))
            use_remote = bool(kwargs.get('use_remote', False))
            stop_dist_m = float(kwargs.get('stop_dist_m', 0.5))
            tilt_only = bool(kwargs.get('tilt_only', True))
            tilt_down_negative = bool(kwargs.get('tilt_down_negative', True))
            invert_yaw = bool(kwargs.get('invert_yaw', True))
        cmd = [
            sys.executable, "-m",
            "VLServo.base_motion",
            "-x", str(x), "-y", str(y),
            "--stop-dist-m", str(stop_dist_m)
        ]
        if use_remote:
            cmd.append("-r")
        if tilt_only:
            cmd.append("--tilt-only")
        cmd.append("--tilt-down-negative" if tilt_down_negative else "--tilt-down-positive")
        if invert_yaw:
            cmd.append("--invert-yaw")
        logger.info(f"Starting LLM Navigation: {' '.join(cmd)}")
        env = os.environ.copy()
        sep = os.pathsep
        existing = env.get('PYTHONPATH', '')
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(pkg_dir)
        env['PYTHONPATH'] = f"{parent_dir}{sep}{pkg_dir}{(sep + existing) if existing else ''}"
        env.setdefault('MPLBACKEND', 'Agg')
        env.setdefault('QT_QPA_PLATFORM', 'offscreen')
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


class ArmGraspProcess:
    """Manage the D435i-based arm grasp process (visual_servoing_demo_llm)."""
    def __init__(self):
        self.proc = None

    def start(self, *args, **kwargs):
        if self.proc and self.proc.poll() is None:
            return  # already running
        # Parse args (x, y, use_remote, stop_z_m, grasp_z_m)
        if args and len(args) >= 2:
            x, y = int(args[0]), int(args[1])
            use_remote = bool(args[2]) if len(args) >= 3 else False
            stop_z_m = float(args[3]) if len(args) >= 4 else 0.38
            grasp_z_m = float(args[4]) if len(args) >= 5 else 0.28
        else:
            x = int(kwargs.get('x'))
            y = int(kwargs.get('y'))
            use_remote = bool(kwargs.get('use_remote', False))
            stop_z_m = float(kwargs.get('stop_z_m', 0.38))
            grasp_z_m = float(kwargs.get('grasp_z_m', 0.28))
        cmd = [
            sys.executable, "-m",
            "VLServo.visual_servoing_demo_llm",
            "-x", str(x), "-y", str(y),
            "--stop-z-m", str(stop_z_m),
            "--grasp-z-m", str(grasp_z_m)
        ]
        if use_remote:
            cmd.append("-r")
        logger.info(f"Starting LLM Arm Grasp: {' '.join(cmd)}")
        env = os.environ.copy()
        sep = os.pathsep
        existing = env.get('PYTHONPATH', '')
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(pkg_dir)
        env['PYTHONPATH'] = f"{parent_dir}{sep}{pkg_dir}{(sep + existing) if existing else ''}"
        env.setdefault('MPLBACKEND', 'Agg')
        env.setdefault('QT_QPA_PLATFORM', 'offscreen')
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
    try:
        if hasattr(ImageQt, "toqimage"):
            qimg = ImageQt.toqimage(img)
            return qimg.copy()
    except Exception:
        pass
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
    return QPixmap.fromImage(pil_to_qimage(img))


class ChatWidget(QTextEdit):
    """Chat area that can render text and inline images (Pillow -> QImage)."""
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMinimumHeight(400)
        self.setFont(QFont("Arial", 10))

    def add_message(self, sender, content, is_image=False):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(f"\n{sender}:\n")
        if is_image and isinstance(content, tuple):
            text, image, _ = content
            cursor.insertText(f"{text}\n")
            if image is not None:
                qimage = pil_to_qimage(image)
                max_w = 600
                if qimage.width() > max_w:
                    qimage = qimage.scaled(max_w, int(qimage.height() * (max_w / qimage.width())), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                cursor.insertImage(qimage)
                cursor.insertText("\n")
        else:
            cursor.insertText(f"{content}\n")
        self.moveCursor(QTextCursor.End)


class ServerProcess:
    def __init__(self):
        self.controller_process = None
        self.model_worker_process = None
        self.controller_url = "http://10.0.0.1:11000"

    def start_controller(self, host="0.0.0.0", port=11000):
        cmd = [sys.executable, "-m", "point.serve.controller", "--host", host, "--port", str(port)]
        logger.info(f"Starting controller: {' '.join(cmd)}")
        self.controller_process = subprocess.Popen(cmd)
        self.controller_url = f"http://{('10.0.0.1' if host in ['0.0.0.0', '::'] else host)}:{port}"

    def start_model_worker(self, host="0.0.0.0", controller_url="http://10.0.0.1:11000",
                           port=22000, worker_url="http://10.0.0.1:22000",
                           model_path="wentao-yuan/robopoint-v1-vicuna-v1.5-13b", load_4bit=True):
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
        if self.controller_process:
            self.controller_process.terminate()
            self.controller_process = None
        if self.model_worker_process:
            self.model_worker_process.terminate()
            self.model_worker_process = None


class ModelWorkerThread(QThread):
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished_request = pyqtSignal()

    def __init__(self, controller_url="http://10.0.0.1:11000"):
        super().__init__()
        self.controller_url = controller_url
        self.request_data = None

    def set_request_data(self, data):
        self.request_data = data

    def run(self):
        try:
            ret = requests.post(self.controller_url + "/get_worker_address", json={"model": self.request_data["model"]})
            worker_addr = ret.json()["address"]
            if worker_addr == "":
                self.error_occurred.emit("No available worker")
                return
            response = requests.post(worker_addr + "/worker_generate_stream",
                                     headers={"User-Agent": "RoboPoint Client"},
                                     json=self.request_data, stream=True, timeout=30)
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if not chunk:
                    continue
                data = json.loads(chunk.decode())
                if data.get("error_code", 1) == 0:
                    output = data["text"][len(self.request_data["prompt"]):].strip()
                    self.response_received.emit(output)
                else:
                    self.error_occurred.emit(f"Error: {data.get('text','')} (code: {data.get('error_code')})")
                    return
            self.finished_request.emit()
        except Exception as e:
            self.error_occurred.emit(f"Request failed: {str(e)}")


class ImageLabel(QLabel):
    # Add signal for mouse click events
    mouse_clicked = pyqtSignal(int, int)  # x, y coordinates
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 2px dashed #aaa;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("Waiting for camera frames…")
        self.setAcceptDrops(False)
        self.image_path = None
        self.original_image = None
        self.scale_factor = 1.0  # Track scaling for coordinate conversion
        # Track source frame dimensions for numpy-based frames
        self.source_w = None
        self.source_h = None

    def load_pil(self, pil_image):
        self.image_path = None
        self.original_image = pil_image
        # Set source dimensions for click mapping
        try:
            self.source_w, self.source_h = pil_image.width, pil_image.height
        except Exception:
            self.source_w, self.source_h = None, None
        pixmap = qpixmap_from_pil(self.original_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Calculate scale factor for coordinate conversion
        if self.original_image and scaled_pixmap.width() > 0 and scaled_pixmap.height() > 0:
            self.scale_factor = min(
                self.original_image.width / scaled_pixmap.width(),
                self.original_image.height / scaled_pixmap.height()
            )
        else:
            self.scale_factor = 1.0
            
        self.setPixmap(scaled_pixmap)

    def load_rgb_np(self, np_rgb):
        """Fast path: draw a numpy RGB array directly without PIL conversions."""
        try:
            if np_rgb is None:
                return
            h, w = np_rgb.shape[:2]
            # Update source dimensions for click mapping
            self.source_w, self.source_h = int(w), int(h)
            self.original_image = None  # Make it clear we are using numpy-based source
            # Ensure 3-channel uint8
            if np_rgb.dtype != np.uint8 or np_rgb.ndim != 3 or np_rgb.shape[2] != 3:
                return self.load_pil(Image.fromarray(np_rgb))
            bytes_per_line = 3 * w
            qimg = QImage(np_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            qimg = qimg.copy()  # detach from numpy buffer
            pixmap = QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # Track scale factor for click mapping similar to load_pil
            if scaled_pixmap.width() > 0 and scaled_pixmap.height() > 0:
                self.scale_factor = min(w / scaled_pixmap.width(), h / scaled_pixmap.height())
            else:
                self.scale_factor = 1.0
            self.setPixmap(scaled_pixmap)
        except Exception:
            # Fallback to PIL path on any error
            try:
                self.load_pil(Image.fromarray(np_rgb))
            except Exception:
                pass

    def mousePressEvent(self, event):
        """Handle mouse click events and convert to original image coordinates."""
        if event.button() == Qt.LeftButton:
            # Get click position relative to the widget
            click_pos = event.pos()
            
            # Get the actual pixmap position within the label
            pixmap = self.pixmap()
            if pixmap and (self.source_w is not None) and (self.source_h is not None):
                # Calculate the offset to center the pixmap in the label
                label_size = self.size()
                pixmap_size = pixmap.size()
                
                offset_x = (label_size.width() - pixmap_size.width()) // 2
                offset_y = (label_size.height() - pixmap_size.height()) // 2
                
                # Adjust click position relative to pixmap
                pixmap_x = click_pos.x() - offset_x
                pixmap_y = click_pos.y() - offset_y
                
                # Check if click is within pixmap bounds
                if (0 <= pixmap_x < pixmap_size.width() and 
                    0 <= pixmap_y < pixmap_size.height()):
                    
                    # Convert to original image coordinates
                    original_x = int(pixmap_x * self.scale_factor)
                    original_y = int(pixmap_y * self.scale_factor)
                    
                    # Clamp to image bounds
                    original_x = max(0, min(original_x, self.source_w - 1))
                    original_y = max(0, min(original_y, self.source_h - 1))
                    
                    # Emit the signal with original image coordinates
                    self.mouse_clicked.emit(original_x, original_y)
        
        super().mousePressEvent(event)


class SimplePointTracker:
    """Template-matching based point tracker to keep LLM white-point attached to the object.

    Initializes a template around the provided pixel and, on update, finds the
    best match within a local window to produce a new pixel location.
    """
    def __init__(self, template_size: int = 41, search_radius: int = 40, min_score: float = 0.4):
        self.template_size = max(11, int(template_size) | 1)
        self.search_radius = max(8, int(search_radius))
        self.min_score = float(min_score)
        self.template = None
        self.px = None
        self.py = None
        self.initialized = False

    def _extract_patch(self, gray, cx, cy, size):
        h, w = gray.shape[:2]
        half = size // 2
        x0 = max(0, int(cx) - half); x1 = min(w, int(cx) + half + 1)
        y0 = max(0, int(cy) - half); y1 = min(h, int(cy) + half + 1)
        patch = gray[y0:y1, x0:x1]
        if patch.shape[0] != size or patch.shape[1] != size:
            pad_t = size - patch.shape[0]
            pad_l = size - patch.shape[1]
            patch = cv2.copyMakeBorder(patch, 0, pad_t, 0, pad_l, cv2.BORDER_REPLICATE)
        return patch

    def initialize(self, color_image, px, py):
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY) if color_image.ndim == 3 else color_image
        self.template = self._extract_patch(gray, px, py, self.template_size)
        self.px, self.py = int(px), int(py)
        self.initialized = True

    def update(self, color_image):
        if not self.initialized:
            raise RuntimeError('Tracker not initialized')
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY) if color_image.ndim == 3 else color_image
        h, w = gray.shape[:2]
        half_t = self.template_size // 2
        sr = self.search_radius
        x0 = max(0, self.px - sr - half_t); x1 = min(w, self.px + sr + half_t + 1)
        y0 = max(0, self.py - sr - half_t); y1 = min(h, self.py + sr + half_t + 1)
        search = gray[y0:y1, x0:x1]
        if search.shape[0] < self.template_size or search.shape[1] < self.template_size:
            return self.px, self.py
        res = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        # If quality too low, keep last location
        if max_val < self.min_score:
            return self.px, self.py
        cx = x0 + max_loc[0] + half_t
        cy = y0 + max_loc[1] + half_t
        self.px, self.py = int(cx), int(cy)
        return self.px, self.py

    def reset(self):
        self.template = None
        self.initialized = False
        self.px = None
        self.py = None


class CameraReceiverThread(QThread):
    frame_received = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, port, use_remote=False):
        super().__init__()
        self.port = port
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
                address = 'tcp://' + yn.robot_ip + ':' + str(self.port)
            else:
                address = 'tcp://' + '127.0.0.1' + ':' + str(self.port)
            self._socket.connect(address)
            self.status_changed.emit(f"Connected ({address})")
            while self._running:
                try:
                    out = self._socket.recv_pyobj(flags=0)
                except zmq.error.ZMQError:
                    break
                if not self._running:
                    break
                if isinstance(out, dict):
                    self.frame_received.emit(out)
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


class DirectCameraThread(QThread):
    frame_received = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, camera: str = 'D405', exposure: str = 'low'):
        super().__init__()
        self.camera = camera
        self.exposure = exposure
        self._running = True
        self._pipeline = None
        self._profile = None
        self._ctx = None
        self._pub = None
        self._pub_every_n = 3
        self._frame_idx = 0

    def stop(self):
        self._running = False

    def run(self):
        try:
            import zmq
            from VLServo import yolo_networking as yn
            # Lazy import pyrealsense helpers based on selection
            if self.camera == 'D405':
                from VLServo import d405_helpers as dh
                pipeline, profile = dh.start_d405(self.exposure)
                port = yn.d405_port
            else:
                from VLServo import d435i_helpers as dh
                pipeline, profile = dh.start_d435i(self.exposure)
                port = yn.d435i_port

            self._pipeline, self._profile = pipeline, profile
            # Setup local ZMQ PUB to mirror send_* scripts so YOLO subscribers can work
            try:
                self._ctx = zmq.Context()
                self._pub = self._ctx.socket(zmq.PUB)
                self._pub.setsockopt(zmq.SNDHWM, 1)
                self._pub.setsockopt(zmq.RCVHWM, 1)
                addr = 'tcp://' + '127.0.0.1' + ':' + str(port)
                self._pub.bind(addr)
                self.status_changed.emit(f"Camera started ({self.camera}, exposure={self.exposure}).\nPublishing {addr}")
            except Exception as e:
                self.status_changed.emit(f"Camera started ({self.camera}, exposure={self.exposure}).\nPublish error: {e}")

            import numpy as np
            import time
            first_frame = True
            depth_camera_info = None
            color_camera_info = None
            depth_scale = None
            consecutive_errors = 0

            while self._running:
                self._frame_idx += 1
                try:
                    # Prefer non-blocking poll to avoid wait timeout exceptions
                    frames = pipeline.poll_for_frames()
                    if not frames:
                        time.sleep(0.002)
                        continue
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    if (not depth_frame) or (not color_frame):
                        continue

                    if first_frame:
                        depth_scale = dh.get_depth_scale(profile)
                        depth_camera_info = dh.get_camera_info(depth_frame)
                        color_camera_info = dh.get_camera_info(color_frame)
                        # Trim distortion_model for downstream dict consistency
                        try:
                            if isinstance(depth_camera_info, dict):
                                depth_camera_info.pop('distortion_model', None)
                            if isinstance(color_camera_info, dict):
                                color_camera_info.pop('distortion_model', None)
                        except Exception:
                            pass
                        first_frame = False

                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())

                    out = {
                        'depth_camera_info': depth_camera_info,
                        'color_camera_info': color_camera_info,
                        'depth_scale': depth_scale,
                        'color_image': color_image,
                        'depth_image': depth_image,
                    }
                    self.frame_received.emit(out)
                    # Also publish over ZMQ for subscribers (e.g., YOLO)
                    try:
                        if self._pub is not None and (self._frame_idx % max(1, int(self._pub_every_n))) == 0:
                            # Reduce publish frequency to limit pickling overhead and GIL stalls
                            self._pub.send_pyobj(out)
                    except Exception:
                        pass
                    # Reset error counter on success
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    # Surface the error without stopping the thread
                    try:
                        self.status_changed.emit(f"Error: {e}")
                    except Exception:
                        pass
                    # Attempt lightweight recovery: short sleep, then possibly reinitialize pipeline
                    time.sleep(0.02)
                    if consecutive_errors >= 5 and self._running:
                        # Try to reinitialize the camera pipeline
                        try:
                            if self._pipeline is not None:
                                try:
                                    self._pipeline.stop()
                                except Exception:
                                    pass
                            if self.camera == 'D405':
                                pipeline, profile = dh.start_d405(self.exposure)
                            else:
                                pipeline, profile = dh.start_d435i(self.exposure)
                            self._pipeline, self._profile = pipeline, profile
                            first_frame = True
                            consecutive_errors = 0
                            self.status_changed.emit("Camera recovered")
                        except Exception as e2:
                            # Could not recover yet; keep looping
                            try:
                                self.status_changed.emit(f"Recover failed: {e2}")
                            except Exception:
                                pass
                            time.sleep(0.05)

        except Exception as e:
            self.status_changed.emit(f"Error: {e}")
        finally:
            try:
                if self._pipeline is not None:
                    self._pipeline.stop()
            except Exception:
                pass
            try:
                if self._pub is not None:
                    self._pub.close(0)
                if self._ctx is not None:
                    self._ctx.term()
            except Exception:
                pass
            self.status_changed.emit("Camera stopped")


class RoboPointMainWindow(QMainWindow):
    def __init__(self, controller_url=None, autostart=False, model_path="wentao-yuan/robopoint-v1-vicuna-v1.5-13b",
                 load_4bit=True, use_remote_stream=False):
        super().__init__()
        self.setWindowTitle("RoboPoint - Visual Servoing")
        self.setGeometry(100, 100, 1200, 800)

        self.server_process = ServerProcess()
        self.worker_thread = ModelWorkerThread()
        self.model_path = model_path
        self.load_4bit = load_4bit
        self.autostart = autostart
        self.conversation_state = default_conversation.copy()
        self.models = []
        self._pending_response_text = None

        # Camera/YOLO state
        self.use_remote_stream = use_remote_stream
        self.use_remote_yolo = use_remote_stream
        self.camera_thread = None
        self.yolo_thread = None
        self.latest_camera_info = None
        self.latest_depth_scale = None
        self.latest_depth_image = None
        self.last_color_frame_bgr = None
        self.latest_yolo_send_dict = None
        self.latest_fingertips = None
        self.latest_aruco_markers = {}
        self.latest_d405_pixel = None
        self.latest_d405_frame_idx = -1
        self.d405_display_x_ratio = 0.50  # Center horizontally (rotated view) for D405 marker
        self.d405_display_y_ratio = 0.50  # Center vertically (rotated view) for D405 marker
        self.latest_llm_points = []  # sensor-frame (x,y) pixels derived from LLM annotations
        self.latest_llm_points_size = None  # (width, height) of image used for those coords
        self.llm_tracker = None
        self.llm_tracked_px = None
        self.llm_tracked_py = None
        self.llm_clusters = None
        self.llm_cluster_idx = -1
        self.user_clicked_position = False  # Flag to track if user manually clicked
        # Track whether an LLM grasping session is active (awaiting/using white point)
        self.llm_session_active = False
        # Track whether a navigation session is active
        self.nav_session_active = False
        # UI frame update limiter to reduce heavy redraw stutter
        self.ui_fps_limit = 15.0
        self._last_ui_frame_ts = 0.0
        self._frame_idx = 0
        self._aruco_every_n = 3  # do ArUco update every N frames to reduce UI stalls

        # Direct camera capture thread (no external process)
        self.direct_camera_thread = None

        # External processes for YOLO grasping
        self.yolo_grasp_proc = YoloGraspProcess()
        self.visual_servo_proc = VisualServoingProcess()
        # External process for LLM (white-point) grasping
        self.white_point_proc = WhitePointGraspProcess()
        # External process for LLM Navigation
        self.base_nav_proc = BaseNavigationProcess()
        # External process for Arm grasping (D435i + arm reach)
        self.arm_grasp_proc = ArmGraspProcess()
        # Track if we started the demo process to avoid camera conflicts
        self._llm_demo_started_here = False
        # Track current camera selection for display transforms (fixed to D435i)
        self.current_camera = 'D435i'

        # ArUco perception
        try:
            from VLServo import aruco_detector as ad
            from VLServo import aruco_to_fingertips as af
            from VLServo import d405_helpers_without_pyrealsense as dh
            self.dh = dh
            with open(os.path.join(os.path.dirname(__file__), 'aruco_marker_info.yaml')) as f:
                marker_info = yaml.load(f, Loader=SafeLoader)
        except Exception:
            try:
                with open('aruco_marker_info.yaml') as f:
                    marker_info = yaml.load(f, Loader=SafeLoader)
            except Exception:
                marker_info = {}
            from VLServo import aruco_detector as ad
            from VLServo import aruco_to_fingertips as af
        self.aruco_detector = ad.ArucoDetector(marker_info=marker_info, show_debug_images=False, use_apriltag_refinement=False, brighten_images=False)
        self.aruco_to_fingertips = af.ArucoToFingertips(default_height_above_mounting_surface=af.suctioncup_height['cup_top'])

        self.setup_ui()
        self.setup_connections()

        if controller_url:
            self.server_process.controller_url = controller_url
            self.worker_thread.controller_url = controller_url

        if autostart:
            self.start_services()
        else:
            QTimer.singleShot(1000, self.refresh_model_list)

        # Start YOLO subscriber (optional publisher)
        self.start_yolo_subscriber()
        # If subscribing to a remote robot's ZMQ stream, start receiver now
        if self.use_remote_stream:
            self.start_camera_receiver()

        # Head auto-tracking timer (keeps gripper centered during grasping)
        self.head_track_timer = QTimer(self)
        self.head_track_timer.setInterval(250)
        try:
            self.head_track_timer.timeout.connect(self._head_track_tick)
        except Exception:
            pass

    def setup_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left = QWidget(); left_layout = QVBoxLayout(left)
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        self.model_selector = QComboBox()
        model_layout.addWidget(self.model_selector)
        left_layout.addWidget(model_group)

        # Camera stream group
        stream_group = QGroupBox("Camera Stream")
        stream_layout = QVBoxLayout(stream_group)
        self.image_label = ImageLabel()
        stream_layout.addWidget(self.image_label)
        # Lightweight overlay toggle to draw white point without extra processing
        try:
            self.overlay_white_dot_cb = QCheckBox("Overlay white dot")
            self.overlay_white_dot_cb.setChecked(True)
            stream_layout.addWidget(self.overlay_white_dot_cb)
        except Exception:
            self.overlay_white_dot_cb = None
        self.camera_status = QLabel("Status: Disconnected")
        stream_layout.addWidget(self.camera_status)
        left_layout.addWidget(stream_group)

        # Parameters group (kept but hidden, so dependent code can read defaults)
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)
        params_layout.addWidget(QLabel("Temperature:"))
        self.temperature_slider = QSlider(Qt.Horizontal); self.temperature_slider.setRange(0, 100); self.temperature_slider.setValue(100)
        self.temperature_label = QLabel("1.0")
        trow = QHBoxLayout(); trow.addWidget(self.temperature_slider); trow.addWidget(self.temperature_label)
        params_layout.addLayout(trow)
        params_layout.addWidget(QLabel("Top P:"))
        self.top_p_slider = QSlider(Qt.Horizontal); self.top_p_slider.setRange(0, 100); self.top_p_slider.setValue(70)
        self.top_p_label = QLabel("0.7")
        prow = QHBoxLayout(); prow.addWidget(self.top_p_slider); prow.addWidget(self.top_p_label)
        params_layout.addLayout(prow)
        params_layout.addWidget(QLabel("Max Output Tokens:"))
        self.max_tokens_spin = QSpinBox(); self.max_tokens_spin.setRange(0, 1024); self.max_tokens_spin.setValue(512)
        params_layout.addWidget(self.max_tokens_spin)
        left_layout.addWidget(params_group)
        params_group.setVisible(False)

        # Service status group (moved to bottom, single-line status)
        status_group = QGroupBox("Service Status")
        status_layout = QVBoxLayout(status_group)
        self.service_status = QLabel("Controller: Not Started | Model Worker: Not Started")
        status_layout.addWidget(self.service_status)
        # Prepare buttons but do not add to layout (no buttons requested)
        self.start_services_btn = QPushButton("Start Services"); self.stop_services_btn = QPushButton("Stop Services")

        # Robot control: explicit Reset/Calibrate (Homing)
        robot_group = QGroupBox("Robot Control")
        robot_layout = QVBoxLayout(robot_group)
        self.btn_reset_calibrate = QPushButton("Reset / Calibrate (Home)")
        self.robot_status = QLabel("")
        robot_layout.addWidget(self.btn_reset_calibrate)
        robot_layout.addWidget(self.robot_status)

        # Camera control
        sender_group = QGroupBox("Camera")
        sender_layout = QVBoxLayout(sender_group)
        # Camera is fixed to D435i; keep an internal selector for compatibility but do not show it
        self.camera_selector = QComboBox(); self.camera_selector.addItems(["D405", "D435i"])
        try:
            self.camera_selector.setCurrentText("D435i")
        except Exception:
            pass
        row2 = QHBoxLayout(); row2.addWidget(QLabel("Exposure:"))
        self.exposure_selector = QComboBox(); self.exposure_selector.addItems(["low", "medium", "auto"]) ; row2.addWidget(self.exposure_selector)
        sender_layout.addLayout(row2)
        btnrow = QHBoxLayout();
        self.btn_start_sender = QPushButton("Start Camera"); self.btn_stop_sender = QPushButton("Stop Camera"); self.btn_stop_sender.setEnabled(False)
        btnrow.addWidget(self.btn_start_sender); btnrow.addWidget(self.btn_stop_sender)
        sender_layout.addLayout(btnrow)
        left_layout.addWidget(sender_group)
        left_layout.addWidget(robot_group)

        # LLM Grasping controls (white-point -> 3D -> publish)
        llm_group = QGroupBox("LLM Grasping")
        llm_layout = QVBoxLayout(llm_group)
        self.btn_start_llm = QPushButton("Start LLM Grasping"); self.btn_stop_llm = QPushButton("Stop LLM Grasping"); self.btn_stop_llm.setEnabled(False)
        self.btn_clean_llm = QPushButton("Clean Points")
        self.llm_status = QLabel("LLM Grasping: Not Running")
        # Add a dedicated checkbox for LLM flow to control running the grasp demo
        self.manage_llm_demo_checkbox = QCheckBox("Manage Visual Servoing process")
        # Default checked so LLM Grasping will move the robot toward the white point
        self.manage_llm_demo_checkbox.setChecked(True)
        # Grasp closeness control (extra push along viewing ray)
        self.extra_push_cm_spin = QDoubleSpinBox(); self.extra_push_cm_spin.setRange(0.0, 5.0); self.extra_push_cm_spin.setSingleStep(0.5); self.extra_push_cm_spin.setValue(3.0)
        self.extra_push_cm_spin.setSuffix(" cm")
        extra_row = QHBoxLayout(); extra_row.addWidget(QLabel("Grasp Offset (toward object):")); extra_row.addWidget(self.extra_push_cm_spin)
        # Place buttons side-by-side
        llmrow = QHBoxLayout(); llmrow.addWidget(self.btn_start_llm); llmrow.addWidget(self.btn_stop_llm)
        llm_layout.addLayout(llmrow)
        llm_layout.addLayout(extra_row)
        llm_layout.addWidget(self.btn_clean_llm)
        llm_layout.addWidget(self.llm_status)
        llm_layout.addWidget(self.manage_llm_demo_checkbox)
        left_layout.addWidget(llm_group)

        # YOLO Grasping controls (hidden in GUI per request)
        self.yolo_group = QGroupBox("YOLO Grasping", parent=left)
        yolo_layout = QVBoxLayout(self.yolo_group)
        self.btn_start_yolo = QPushButton("Start YOLO Grasping", parent=self.yolo_group)
        self.btn_stop_yolo = QPushButton("Stop YOLO Grasping", parent=self.yolo_group); self.btn_stop_yolo.setEnabled(False)
        self.yolo_status = QLabel("YOLO Grasping: Not Running", parent=self.yolo_group)
        self.manage_demo_checkbox = QCheckBox("Manage Visual Servoing process", parent=self.yolo_group)
        try:
            self.manage_demo_checkbox.setChecked(bool(int(os.getenv('YOLO_MANAGE_DEMO', '0'))))
        except Exception:
            self.manage_demo_checkbox.setChecked(False)
        yrow = QHBoxLayout(); yrow.addWidget(self.btn_start_yolo); yrow.addWidget(self.btn_stop_yolo)
        yolo_layout.addLayout(yrow)
        yolo_layout.addWidget(self.yolo_status); yolo_layout.addWidget(self.manage_demo_checkbox)
        # Keep the group alive but hidden and not added to the visible layout
        self.yolo_group.hide()
        
        # LLM Navigation controls
        nav_group = QGroupBox("LLM Navigation")
        nav_layout = QVBoxLayout(nav_group)
        self.nav_start_btn = QPushButton("Start LLM Navigation")
        self.nav_stop_btn = QPushButton("Stop LLM Navigation"); self.nav_stop_btn.setEnabled(False)
        nav_row = QHBoxLayout(); nav_row.addWidget(self.nav_start_btn); nav_row.addWidget(self.nav_stop_btn)
        nav_layout.addLayout(nav_row)
        # Replace stop distance and advanced options with a direct head tilt control (degrees)
        tilt_row = QHBoxLayout(); tilt_row.addWidget(QLabel("Head tilt (deg):"))
        # Allow tilting down to -90 degrees, clamp max to 0 (no upward tilt needed)
        self.head_tilt_deg = QDoubleSpinBox(); self.head_tilt_deg.setRange(-90.0, 0.0); self.head_tilt_deg.setSingleStep(2.0); self.head_tilt_deg.setValue(0.0)
        tilt_row.addWidget(self.head_tilt_deg)
        nav_layout.addLayout(tilt_row)
        self.nav_status = QLabel("LLM Navigation: Not Running")
        nav_layout.addWidget(self.nav_status)
        left_layout.addWidget(nav_group)
        left_layout.addStretch()
        # Place service status at very bottom
        left_layout.addWidget(status_group)

        # Right panel: chat
        right = QWidget(); right_layout = QVBoxLayout(right)
        self.chat_widget = ChatWidget()
        right_layout.addWidget(self.chat_widget)
        inrow = QHBoxLayout(); self.text_input = QLineEdit(); self.text_input.setPlaceholderText("Enter your question about the image...")
        self.send_button = QPushButton("Send"); self.send_button.setEnabled(False)
        inrow.addWidget(self.text_input); inrow.addWidget(self.send_button)
        right_layout.addLayout(inrow)
        actrow = QHBoxLayout(); self.clear_button = QPushButton("Clear Chat"); self.regenerate_button = QPushButton("Regenerate")
        actrow.addWidget(self.clear_button); actrow.addWidget(self.regenerate_button); actrow.addStretch(); right_layout.addLayout(actrow)

        splitter = QSplitter(Qt.Horizontal); splitter.addWidget(left); splitter.addWidget(right); splitter.setSizes([400, 800])
        main_layout.addWidget(splitter)

        # Enable manual service control by default; autostart remains off
        self.start_services_btn.setEnabled(True)
        self.stop_services_btn.setEnabled(False)

    def setup_connections(self):
        self.send_button.clicked.connect(self.send_message)
        self.text_input.returnPressed.connect(self.send_message)
        self.clear_button.clicked.connect(self.clear_chat)
        self.regenerate_button.clicked.connect(self.regenerate_response)
        self.start_services_btn.clicked.connect(self.start_services)
        self.stop_services_btn.clicked.connect(self.stop_services)
        self.temperature_slider.valueChanged.connect(self.update_temperature)
        self.top_p_slider.valueChanged.connect(self.update_top_p)
        self.btn_start_sender.clicked.connect(self.start_camera)
        self.btn_stop_sender.clicked.connect(self.stop_camera)
        # Camera is fixed to D435i, so ignore selection changes in the GUI
        # self.camera_selector.currentTextChanged.connect(lambda _: self.on_camera_selection_changed())
        self.btn_start_yolo.clicked.connect(self.start_yolo_grasping)
        self.btn_stop_yolo.clicked.connect(self.stop_yolo_grasping)
        self.btn_start_llm.clicked.connect(self.start_llm_grasping)
        self.btn_stop_llm.clicked.connect(self.stop_llm_grasping)
        self.btn_clean_llm.clicked.connect(self.clean_llm_points)
        self.btn_reset_calibrate.clicked.connect(self.reset_calibrate_robot)
        # Navigation buttons
        self.nav_start_btn.clicked.connect(self.start_llm_navigation)
        self.nav_stop_btn.clicked.connect(self.stop_llm_navigation)
        # Debounce head tilt updates to avoid GUI jitter
        self._tilt_apply_timer = QTimer(self); self._tilt_apply_timer.setSingleShot(True)
        self._tilt_apply_timer.setInterval(150)
        self._pending_tilt_deg = 0.0

    def reset_calibrate_robot(self):
        """Stop controller-like processes, free ports, and home the robot safely.

        - Stops visual servo, YOLO grasp, white-point publisher, and navigation processes.
        - Runs stretch_free_robot_process.py to ensure ports are released.
        - Runs stretch_robot_home.py with brief retries when ports report busy.
        - Updates a small status label and shows a dialog on failure.
        """
        btn = getattr(self, 'btn_reset_calibrate', None)
        if btn is not None:
            btn.setEnabled(False)
        try:
            # Stop external processes that might hold serial ports
            try:
                if hasattr(self, 'visual_servo_proc') and self.visual_servo_proc is not None:
                    self.visual_servo_proc.stop()
            except Exception:
                pass
            try:
                if hasattr(self, 'yolo_grasp_proc') and self.yolo_grasp_proc is not None:
                    self.yolo_grasp_proc.stop()
            except Exception:
                pass
            try:
                if hasattr(self, 'white_point_proc') and self.white_point_proc is not None:
                    self.white_point_proc.stop()
            except Exception:
                pass
            try:
                if hasattr(self, 'base_nav_proc') and self.base_nav_proc is not None:
                    self.base_nav_proc.stop()
            except Exception:
                pass

            # Free existing robot processes
            try:
                subprocess.run(["stretch_free_robot_process.py"], timeout=5)
            except Exception:
                pass
            time.sleep(0.5)

            # Try homing with short retries when ports are busy
            success = False
            last_err = None
            for i in range(3):
                try:
                    subprocess.run(["stretch_robot_home.py"], timeout=180, check=True)
                    success = True
                    break
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()
                    if any(k in msg for k in ["busy", "multiple access", "disconnected", "no data"]):
                        time.sleep(2.0)
                        continue
                    else:
                        break

            if success:
                try:
                    self.robot_status.setText("Robot homed successfully.")
                except Exception:
                    pass
            else:
                try:
                    self.robot_status.setText("Homing failed. See logs.")
                except Exception:
                    pass
                try:
                    QMessageBox.warning(self, "Reset / Calibrate", f"Homing did not complete: {last_err}")
                except Exception:
                    pass
        finally:
            if btn is not None:
                btn.setEnabled(True)
        try:
            self.head_tilt_deg.valueChanged.connect(self.on_head_tilt_changed)
            self._tilt_apply_timer.timeout.connect(self._apply_pending_tilt)
        except Exception:
            pass

        # Image label mouse click connection
        self.image_label.mouse_clicked.connect(self.on_image_clicked)

        # Worker thread connections (streaming output)
        self.worker_thread.response_received.connect(self.handle_response)
        self.worker_thread.error_occurred.connect(self.handle_error)
        self.worker_thread.finished_request.connect(self.request_finished)

        # Background status timer
        self.status_timer = QTimer(); self.status_timer.timeout.connect(self.check_service_status); self.status_timer.start(5000)

    def start_camera_receiver(self):
        if not self.use_remote_stream:
            return
        if self.camera_thread is not None:
            return
        # Fixed to D435i stream
        port = yn.d435i_port
        self.camera_thread = CameraReceiverThread(port=port, use_remote=True)
        self.camera_thread.frame_received.connect(self.on_camera_frame)
        self.camera_thread.status_changed.connect(self.on_camera_status)
        self.camera_thread.start()

    def restart_camera_receiver(self):
        if self.camera_thread is not None:
            try:
                self.camera_thread.stop(); self.camera_thread.wait(500)
            except Exception:
                pass
            self.camera_thread = None
        self.start_camera_receiver()

    def on_camera_selection_changed(self):
        # If subscribing to remote stream, switch the SUB port
        # Camera selection is fixed; already set to D435i
        try:
            self.current_camera = 'D435i'
        except Exception:
            pass
        if self.use_remote_stream:
            self.restart_camera_receiver()

    def start_yolo_subscriber(self):
        if self.yolo_thread is not None:
            return
        self.yolo_thread = YoloResultSubscriberThread(use_remote=self.use_remote_stream)
        self.yolo_thread.result_received.connect(self.on_yolo_result)
        self.yolo_thread.start()

    def start_yolo_grasping(self):
        try:
            # Assume YOLO runs on the remote computer when stream is remote.
            self.yolo_grasp_proc.start(use_remote=self.use_remote_yolo)
            if self.manage_demo_checkbox.isChecked():
                self.visual_servo_proc.start(use_remote=self.use_remote_yolo)
            self.btn_start_yolo.setEnabled(False)
            self.btn_stop_yolo.setEnabled(True)
            self.yolo_status.setText("YOLO Grasping: Running")
        except Exception as e:
            QMessageBox.critical(self, "YOLO Grasping", f"Failed to start: {e}")

    def stop_yolo_grasping(self):
        try:
            if self.manage_demo_checkbox.isChecked():
                self.visual_servo_proc.stop()
            self.yolo_grasp_proc.stop()
        finally:
            self.btn_start_yolo.setEnabled(True)
            self.btn_stop_yolo.setEnabled(False)
            self.yolo_status.setText("YOLO Grasping: Not Running")

    def start_llm_grasping(self):
        """Start or re-start the LLM grasping session.

        Behavior:
        - Always return the arm to its initial position (restart the demo process if managed).
        - Do not show popups.
        - If a white point exists, start tracking/publishing immediately; otherwise, wait idle
          at the start pose until a white point appears (from LLM or a user click).
        - Allow pressing Start multiple times to re-home the arm.
        """
        try:
            # Keep the current camera stream (D435i). Do not restart it here.

            self.llm_session_active = True
            # Center the gripper exactly: use zero tilt bias during LLM grasping
            try:
                self.llm_tilt_bias_rad = 0.0
            except Exception:
                pass

            # Free any existing Stretch process; avoid homing so current calibration/state is preserved.
            try:
                subprocess.run(["stretch_free_robot_process.py"], timeout=5)
            except Exception:
                pass

            # After homing, set a reasonable start pose and aim head once (before demo starts)
            try:
                if go_to_start_pose_custom is not None:
                    go_to_start_pose_custom(head_tilt_deg=None, head_pan_rad=None)
                if point_head_to_gripper_custom is not None:
                    point_head_to_gripper_custom()
            except Exception as e:
                logger.warning(f"Failed to set grasping start pose: {e}")

            # Ensure D435i stream is available: if not remote and not running, start it now
            try:
                if (not self.use_remote_stream) and (self.direct_camera_thread is None or not self.direct_camera_thread.isRunning()):
                    self.start_camera()
            except Exception:
                pass

            # Do not start visual_servoing_demo here. We will launch arm_motion instead.
            self._llm_demo_started_here = False

            # Aggressively align head so D405 ArUco is visible in the center BEFORE starting controller
            try:
                self._pre_align_head_to_d405(max_iters=12, pixel_tol=18.0)
            except Exception:
                pass
            # Do not lock head/gripper here; allow arm_motion to steer pan/tilt freely

            # Decide if we can start publishing a white point now
            px_py = None
            if (self.llm_tracked_px is not None) and (self.llm_tracked_py is not None):
                px_py = (int(self.llm_tracked_px), int(self.llm_tracked_py))
            else:
                # Use only explicit points (LLM crosses or a user click). Do NOT auto-detect.
                pts = list(self.latest_llm_points or [])
                if pts:
                    chosen = self._choose_point_from_crosses_by_density(pts, radius=8)
                    refined = self._select_point_from_crosses(self.last_color_frame_bgr, pts)
                    if refined is not None:
                        chosen = refined if (chosen is None) else chosen
                    if chosen is not None:
                        px, py = int(chosen[0]), int(chosen[1])
                        if self.last_color_frame_bgr is not None:
                            hsv = cv2.cvtColor(self.last_color_frame_bgr, cv2.COLOR_BGR2HSV)
                            px, py = self._nearest_non_white_black(hsv, px, py, radius=6)
                        self.llm_tracked_px, self.llm_tracked_py = int(px), int(py)
                        self.llm_tracker = None
                        px_py = (self.llm_tracked_px, self.llm_tracked_py)

            # Launch arm-motion controller if we have a target pixel
            running_arm = (self.arm_grasp_proc.proc is not None) and (self.arm_grasp_proc.proc.poll() is None)
            if px_py is not None:
                if running_arm:
                    try:
                        self.arm_grasp_proc.stop()
                        time.sleep(0.1)
                    except Exception:
                        pass
                # Start arm reach controller (uses D435i stream directly)
                self.arm_grasp_proc.start(px_py[0], px_py[1], self.use_remote_yolo)
                # Compose status with current head angles and approx gripper xyz
                status_extra = ""
                try:
                    if get_head_pan_tilt_custom is not None:
                        pan, tilt = get_head_pan_tilt_custom()
                        status_extra += f" | head_pan={pan:.2f} rad, head_tilt={tilt:.2f} rad"
                except Exception:
                    pass
                try:
                    if get_gripper_forward_xyz_custom is not None:
                        gx, gy, gz = get_gripper_forward_xyz_custom()
                        status_extra += f" | gripper_xyz=({gx:.2f},{gy:.2f},{gz:.2f}) m"
                except Exception:
                    pass
                self.llm_status.setText(f"LLM Grasping: Running at ({px_py[0]}, {px_py[1]}){status_extra}")
                self.btn_stop_llm.setEnabled(True)
            else:
                if running_arm:
                    try:
                        self.arm_grasp_proc.stop()
                    except Exception:
                        pass
                # Even if waiting, show current head angles to aid debugging
                status_extra = ""
                try:
                    if get_head_pan_tilt_custom is not None:
                        pan, tilt = get_head_pan_tilt_custom()
                        status_extra += f" | head_pan={pan:.2f} rad, head_tilt={tilt:.2f} rad"
                except Exception:
                    pass
                self.llm_status.setText(f"LLM Grasping: Waiting for white point{status_extra}")
                self.btn_stop_llm.setEnabled(True)

            # Keep Start enabled so user can re-home anytime
            self.btn_start_llm.setEnabled(True)
            # Do not start GUI head tracking; arm_motion handles view stabilization.
        except Exception as e:
            logger.error(f"LLM Grasping start error: {e}")
            self.llm_status.setText("LLM Grasping: Error starting")

    def stop_llm_grasping(self):
        try:
            # Stop arm grasping controller if running
            try:
                self.arm_grasp_proc.stop()
            except Exception:
                pass
            # Stop demo if any was started (defensive)
            if self._llm_demo_started_here:
                self.visual_servo_proc.stop()
                self._llm_demo_started_here = False
            # Try to unlock head/gripper if available
            try:
                if unlock_head_and_gripper_custom is not None:
                    unlock_head_and_gripper_custom()
            except Exception:
                pass
        finally:
            self.llm_session_active = False
            try:
                self.llm_tilt_bias_rad = HEAD_TILT_BIAS_DEFAULT
            except Exception:
                pass
            self.btn_start_llm.setEnabled(True)
            self.btn_stop_llm.setEnabled(False)
            self.llm_status.setText("LLM Grasping: Not Running")
            try:
                self.head_track_timer.stop()
            except Exception:
                pass

    def on_camera_status(self, text):
        self.camera_status.setText(f"Status: {text}")

    def on_camera_frame(self, output):
        try:
            # Throttle UI redraws to reduce choppiness from heavy image conversions
            try:
                now_ts = time.time()
                min_dt = 1.0 / max(1.0, float(self.ui_fps_limit))
                if (now_ts - self._last_ui_frame_ts) < min_dt:
                    return
            except Exception:
                pass
            self._frame_idx += 1
            color_image = output.get('color_image', None)
            depth_image = output.get('depth_image', None)
            if color_image is None:
                return
            # cache BGR frame for auto-detection on LLM start
            try:
                self.last_color_frame_bgr = color_image
            except Exception:
                pass
            self.latest_camera_info = output.get('depth_camera_info', output.get('color_camera_info', None))
            self.latest_depth_scale = output.get('depth_scale', None)
            self.latest_depth_image = depth_image
            if isinstance(color_image, np.ndarray) and color_image.ndim == 3 and color_image.shape[2] == 3:
                import cv2
                rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            else:
                rgb = color_image if isinstance(color_image, np.ndarray) else np.array(color_image)
            vis = np.copy(rgb)
            h0, w0 = vis.shape[:2]
            post_rotate_annotations = []  # For D435i: draw after rotating display
            # Run ArUco fingertip estimation when camera intrinsics available
            try:
                # Run expensive ArUco update only every Nth frame, draw latest results each time
                if self.latest_camera_info is not None:
                    if (self._frame_idx % max(1, int(self._aruco_every_n))) == 0:
                        self.aruco_detector.update(rgb, self.latest_camera_info)
                        markers = self.aruco_detector.get_detected_marker_dict()
                        self.latest_aruco_markers = markers if markers is not None else {}
                        self.latest_fingertips = self.aruco_to_fingertips.get_fingertips(markers)
                        self.latest_d405_pixel = None
                        updated_d405 = False
                        if markers:
                            try:
                                d405_entry = markers.get(135)
                                if d405_entry is None:
                                    for _mid, mdata in markers.items():
                                        info = mdata.get('info', {}) if isinstance(mdata, dict) else {}
                                        if info.get('name') == 'd405_back':
                                            d405_entry = mdata
                                            break
                                if d405_entry is not None and self.latest_camera_info is not None:
                                    pos = d405_entry.get('pos')
                                    if pos is not None:
                                        uv = self.dh.pixel_from_3d(np.array(pos, dtype=np.float32), self.latest_camera_info)
                                        self.latest_d405_pixel = (float(uv[0]), float(uv[1]))
                                        self.latest_d405_frame_idx = self._frame_idx
                                        updated_d405 = True
                            except Exception:
                                self.latest_d405_pixel = None
                                self.latest_d405_frame_idx = -1
                        if not updated_d405:
                            self.latest_d405_frame_idx = -1
                # Draw fingertips from latest estimation
                if self.latest_fingertips:
                    self.aruco_to_fingertips.draw_fingertip_frames(
                        self.latest_fingertips,
                        vis,
                        self.latest_camera_info,
                        axis_length_in_m=0.02,
                        draw_origins=True,
                        write_coordinates=True
                    )
                markers_to_draw = self.latest_aruco_markers if self.latest_aruco_markers else None
                if markers_to_draw:
                    try:
                        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                        vis_bgr = self.aruco_detector.draw_markers(vis_bgr)
                        vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                    except Exception:
                        pass
                if self.latest_d405_pixel is not None:
                    try:
                        px_d405 = int(round(self.latest_d405_pixel[0]))
                        py_d405 = int(round(self.latest_d405_pixel[1]))
                        if getattr(self, 'current_camera', 'D435i') == 'D435i':
                            xr, yr = (h0 - 1 - py_d405), px_d405
                            post_rotate_annotations.append(('d405', int(xr), int(yr), None))
                        else:
                            cv2.circle(vis, (px_d405, py_d405), 8, (0, 190, 255), 2, lineType=cv2.LINE_AA)
                    except Exception:
                        pass
            except Exception:
                pass
            # Optional overlays (YOLO + tennis target), if available
            if (self.latest_yolo_send_dict is not None) and (self.latest_camera_info is not None):
                try:
                    yolo_list = self.latest_yolo_send_dict.get('yolo', [])
                    for det in yolo_list:
                        import cv2
                        poly = det.get('mask', None)
                        if poly is not None:
                            poly = np.array(poly, dtype=np.int32)
                            mask_bin = np.zeros(vis.shape[:2], dtype=np.uint8)
                            cv2.fillPoly(mask_bin, [poly], 255, lineType=cv2.LINE_AA)
                            overlay_color = np.array([255, 255, 255], dtype=np.float32)
                            alpha = 0.40
                            region = vis[mask_bin == 255].astype(np.float32)
                            blended = (alpha * overlay_color + (1.0 - alpha) * region)
                            vis[mask_bin == 255] = np.clip(blended, 0, 255).astype(np.uint8)
                        else:
                            x1, y1, x2, y2 = det.get('bbox', [0, 0, 0, 0])
                            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2, lineType=cv2.LINE_AA)
                        # Tennis-specific center and distance annotation
                        try:
                            name = det.get('name', '')
                            if name in ['tennis', 'sports ball', '網球']:
                                center_xyz = det.get('grasp_center_xyz', None)
                                if center_xyz is not None:
                                    center_uv = self.dh.pixel_from_3d(np.array(center_xyz, dtype=np.float32), self.latest_camera_info)
                                    px, py = int(round(center_uv[0])), int(round(center_uv[1]))
                                    # Use estimated_z_m if provided, else project depth patch median at center
                                    depth_m = det.get('estimated_z_m', None)
                                    if depth_m is None and self.latest_depth_image is not None and self.latest_depth_scale is not None:
                                        patch = self.latest_depth_image[max(py-2,0):py+3, max(px-2,0):px+3]
                                        if patch is not None and patch.size > 0:
                                            vals = patch.reshape(-1)
                                            if np.issubdtype(vals.dtype, np.floating):
                                                vals = vals[np.isfinite(vals)]
                                            vals = vals[vals > 0]
                                            if vals.size > 0:
                                                med = float(np.median(vals))
                                                depth_m = med if np.issubdtype(patch.dtype, np.floating) else med * float(self.latest_depth_scale)
                                    if depth_m is not None and depth_m > 0:
                                        x_cm, y_cm, z_cm = (np.array(center_xyz, dtype=np.float32) * 100.0).tolist()
                                        line = f"{x_cm:.1f}, {y_cm:.1f}, {z_cm:.1f} cm"
                                        if getattr(self, 'current_camera', 'D405') == 'D435i':
                                            # Draw after rotation: map (px,py) -> (xr,yr)
                                            xr, yr = (h0 - 1 - py), px
                                            post_rotate_annotations.append(('dot', xr, yr, line))
                                        else:
                                            cv2.circle(vis, (px, py), 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                                            font = cv2.FONT_HERSHEY_SIMPLEX
                                            font_scale = 0.6
                                            t_fg, t_bg = 1, 4
                                            (tw, th), _ = cv2.getTextSize(line, font, font_scale, t_fg)
                                            tx = int(px - tw / 2); ty = int(py - 55)
                                            cv2.putText(vis, line, (tx, ty), font, font_scale, (0, 0, 0), t_bg, cv2.LINE_AA)
                                            cv2.putText(vis, line, (tx, ty), font, font_scale, (255, 255, 255), t_fg, cv2.LINE_AA)
                        except Exception:
                            pass
                except Exception:
                    pass

            # Overlay LLM-selected point with tracking to stick to the object
            # Draw the white dot even if depth info isn't available yet; add text only when depth exists.
            have_depth = (self.latest_depth_image is not None) and (self.latest_camera_info is not None) and (self.latest_depth_scale is not None)
            overlay_enabled = (self.overlay_white_dot_cb.isChecked() if getattr(self, 'overlay_white_dot_cb', None) is not None else True)
            # Show if we have any candidate: either parsed LLM crosses or a user/LLM tracked pixel
            active_session = (self.llm_session_active or self.nav_session_active)
            show_white_point = (active_session and overlay_enabled and
                                (bool(self.latest_llm_points) or (self.llm_tracked_px is not None and self.llm_tracked_py is not None)))
            
            if show_white_point:
                try:
                    import cv2
                    h, w = vis.shape[:2]
                    
                    # Initialize tracker on first use if we have LLM points and user hasn't manually clicked
                    if (self.latest_llm_points and not self.user_clicked_position and 
                        (self.llm_tracker is None or (self.llm_tracked_px is None))):
                        cx0, cy0 = w // 2, h // 2
                        best = None
                        for (px0, py0) in self.latest_llm_points:
                            d0 = float(np.hypot(px0 - cx0, py0 - cy0))
                            if (best is None) or (d0 < best[0]):
                                best = (d0, px0, py0)
                        if best is not None:
                            self.llm_tracked_px, self.llm_tracked_py = int(best[1]), int(best[2])
                            self.llm_tracker = SimplePointTracker(template_size=41, search_radius=40)
                            self.llm_tracker.initialize(color_image, self.llm_tracked_px, self.llm_tracked_py)
                    
                    # If we have a tracked point but no tracker initialized, initialize it
                    elif (self.llm_tracked_px is not None and self.llm_tracked_py is not None and 
                          (self.llm_tracker is None or not self.llm_tracker.initialized)):
                        self.llm_tracker = SimplePointTracker(template_size=41, search_radius=40)
                        self.llm_tracker.initialize(color_image, self.llm_tracked_px, self.llm_tracked_py)
                    
                    # Update tracker location if we have one
                    if self.llm_tracker is not None and self.llm_tracker.initialized:
                        self.llm_tracked_px, self.llm_tracked_py = self.llm_tracker.update(color_image)
                        
                    # Use current tracked position
                    if self.llm_tracked_px is not None and self.llm_tracked_py is not None:
                        px, py = int(self.llm_tracked_px), int(self.llm_tracked_py)
                        # Snap off pure white/black to nearest valid pixel
                        try:
                            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
                            px_adj, py_adj = self._nearest_non_white_black(hsv, px, py, radius=5)
                            px, py = px_adj, py_adj
                            self.llm_tracked_px, self.llm_tracked_py = px, py
                        except Exception:
                            pass
                        # Robust depth (optional): compute only when depth info exists, and decimate to reduce cost
                        depth_m = None
                        if have_depth and (self._frame_idx % 2 == 0):
                            k = 3
                            y0 = max(0, py - k); y1 = min(self.latest_depth_image.shape[0], py + k + 1)
                            x0 = max(0, px - k); x1 = min(self.latest_depth_image.shape[1], px + k + 1)
                            patch = self.latest_depth_image[y0:y1, x0:x1]
                            if patch is not None and patch.size > 0:
                                vals = patch.reshape(-1)
                                if np.issubdtype(vals.dtype, np.floating):
                                    vals = vals[np.isfinite(vals)]
                                    vals_m = vals
                                else:
                                    vals_m = vals.astype(np.float32) * float(self.latest_depth_scale)
                                vals_m = vals_m[vals_m > 0]
                                if vals_m.size > 0:
                                    depth_m = float(np.median(vals_m))
                        # Draw overlay
                        if getattr(self, 'current_camera', 'D405') == 'D435i':
                            # Map to rotated display coords after rotation
                            xr, yr = (h0 - 1 - py), px
                            post_rotate_annotations.append(('dot', int(xr), int(yr), None))
                            if (depth_m is not None) and (depth_m > 0):
                                center_xyz = self.dh.pixel_to_3d(np.array([px, py], dtype=np.float32), depth_m, self.latest_camera_info)
                                x_cm, y_cm, z_cm = (center_xyz * 100.0).astype(np.float32)
                                line = f"{x_cm:.1f}, {y_cm:.1f}, {z_cm:.1f} cm"
                                post_rotate_annotations.append(('text', int(xr), int(yr), line))
                        else:
                            # Always draw a fast white dot overlay
                            cv2.circle(vis, (px, py), 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                            # Only add text when we have depth and a recent estimate
                            if (depth_m is not None) and (depth_m > 0):
                                center_xyz = self.dh.pixel_to_3d(np.array([px, py], dtype=np.float32), depth_m, self.latest_camera_info)
                                x_cm, y_cm, z_cm = (center_xyz * 100.0).astype(np.float32)
                                line = f"{x_cm:.1f}, {y_cm:.1f}, {z_cm:.1f} cm"
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.6
                                t_fg, t_bg = 1, 4
                                (tw, th), _ = cv2.getTextSize(line, font, font_scale, t_fg)
                                tx = int(px - tw / 2); ty = int(py - 55)
                                cv2.putText(vis, line, (tx, ty), font, font_scale, (0, 0, 0), t_bg, cv2.LINE_AA)
                                cv2.putText(vis, line, (tx, ty), font, font_scale, (255, 255, 255), t_fg, cv2.LINE_AA)
                except Exception:
                    pass
            # Rotate for D435i and draw any queued annotations in rotated frame with upright text
            try:
                if getattr(self, 'current_camera', 'D405') == 'D435i':
                    import cv2
                    vis = cv2.rotate(vis, cv2.ROTATE_90_CLOCKWISE)
                    for kind, xr, yr, line in post_rotate_annotations:
                        if kind == 'dot':
                            cv2.circle(vis, (int(xr), int(yr)), 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                        elif kind == 'd405':
                            cv2.circle(vis, (int(xr), int(yr)), 8, (0, 190, 255), 2, lineType=cv2.LINE_AA)
                        elif kind == 'text':
                            # Draw upright text above the already-rotated cross
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.6
                            t_fg, t_bg = 1, 4
                            (tw, th), _ = cv2.getTextSize(line, font, font_scale, t_fg)
                            tx = int(xr - tw / 2); ty = int(yr - 55)
                            cv2.putText(vis, line, (tx, ty), font, font_scale, (0, 0, 0), t_bg, cv2.LINE_AA)
                            cv2.putText(vis, line, (tx, ty), font, font_scale, (255, 255, 255), t_fg, cv2.LINE_AA)
            except Exception:
                pass
            # Draw without PIL to keep GUI smooth
            self.image_label.load_rgb_np(vis)
            try:
                self._last_ui_frame_ts = now_ts
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Failed to handle camera frame: {e}")

    def on_yolo_result(self, send_dict):
        self.latest_yolo_send_dict = send_dict
        try:
            pix = send_dict.get('d405_pixel', None)
            if isinstance(pix, (tuple, list)) and len(pix) == 2:
                px = float(pix[0]); py = float(pix[1])
                self.latest_d405_pixel = (px, py)
                self.latest_d405_frame_idx = max(0, int(getattr(self, '_frame_idx', 0)))
        except Exception:
            pass

    def _head_track_tick(self):
        """Periodically nudge head pan/tilt to keep gripper centered.

        Uses ArUco fingertip origins if available; otherwise does nothing.
        Skip when the demo process is managing the robot to avoid port conflicts.
        """
        try:
            if not self.llm_session_active:
                return
            if getattr(self, 'manage_llm_demo_checkbox', None) is not None and self.manage_llm_demo_checkbox.isChecked():
                return
            if self.last_color_frame_bgr is None:
                return
            if self.latest_camera_info is None:
                return
            K = self.latest_camera_info.get('camera_matrix', None)
            if K is None:
                return
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            if abs(fx) < 1e-6 or abs(fy) < 1e-6:
                return

            h, w = self.last_color_frame_bgr.shape[:2]
            target_px = None
            target_py = None
            using_d405 = False

            if self.latest_d405_pixel is not None and self.latest_d405_frame_idx >= 0:
                if (self._frame_idx - self.latest_d405_frame_idx) <= max(3, int(self._aruco_every_n) * 2):
                    target_px = float(self.latest_d405_pixel[0])
                    target_py = float(self.latest_d405_pixel[1])
                    using_d405 = True

            if target_px is None or target_py is None:
                ft = getattr(self, 'latest_fingertips', None)
                if not ft:
                    return
                # Project fingertip origins to pixel coordinates via intrinsics
                def proj(p):
                    x, y, z = float(p[0]), float(p[1]), float(p[2])
                    if z <= 1e-6:
                        z = 1e-6
                    u = (fx * x) / z + cx
                    v = (fy * y) / z + cy
                    return u, v

                pixels = []
                for side in ['left', 'right']:
                    d = ft.get(side)
                    if d and ('pos' in d):
                        u, v = proj(d['pos'])
                        pixels.append((u, v))
                if not pixels:
                    return
                target_px = float(sum([u for (u, v) in pixels]) / len(pixels))
                target_py = float(sum([v for (u, v) in pixels]) / len(pixels))

            max_w = max(float(w) - 1.0, 0.0)
            max_h = max(float(h) - 1.0, 0.0)
            target_px = float(np.clip(target_px, 0.0, max_w))
            target_py = float(np.clip(target_py, 0.0, max_h))

            if getattr(self, 'current_camera', 'D435i') == 'D435i':
                desired_px = float(self.d405_display_y_ratio) * max_w
                desired_py = max_h - (float(self.d405_display_x_ratio) * max_h)
            else:
                desired_px = max_w * 0.5
                desired_py = max_h * 0.5

            dx = target_px - desired_px
            dy = target_py - desired_py

            pan_delta = float(dx / fx)
            if using_d405:
                bias_rad = 0.0
            else:
                try:
                    bias_rad = float(getattr(self, 'llm_tilt_bias_rad', HEAD_TILT_BIAS_DEFAULT))
                except Exception:
                    bias_rad = float(HEAD_TILT_BIAS_DEFAULT)
            v_bias_px = float(-fy * bias_rad)
            tilt_delta = float(-(dy - v_bias_px) / fy)

            if nudge_head_pan_tilt_custom is not None:
                nudge_head_pan_tilt_custom(pan_delta, tilt_delta, max_step=0.15)
        except Exception:
            pass

    def _pre_align_head_to_d405(self, max_iters=10, pixel_tol=20.0):
        """Aggressively slew head pan/tilt so D405 ArUco sits center-right.

        Runs before starting the arm controller to ensure the D405 marker
        is visible. Uses larger step size than the periodic tracker.
        """
        try:
            if nudge_head_pan_tilt_custom is None:
                return
            if self.latest_camera_info is None or self.last_color_frame_bgr is None:
                return
            K = self.latest_camera_info.get('camera_matrix', None)
            if K is None:
                return
            fx = float(K[0, 0]); fy = float(K[1, 1]); cx = float(K[0, 2]); cy = float(K[1, 2])
            if abs(fx) < 1e-6 or abs(fy) < 1e-6:
                return

            h, w = self.last_color_frame_bgr.shape[:2]
            max_w = max(float(w) - 1.0, 0.0)
            max_h = max(float(h) - 1.0, 0.0)

            # If we don't have a recent D405 pixel, try one update from the latest frame
            target_px = None; target_py = None
            try:
                if self.latest_d405_pixel is None or (self._frame_idx - self.latest_d405_frame_idx) > max(3, int(self._aruco_every_n) * 2):
                    rgb = cv2.cvtColor(self.last_color_frame_bgr, cv2.COLOR_BGR2RGB)
                    self.aruco_detector.update(rgb, self.latest_camera_info)
                    markers = self.aruco_detector.get_detected_marker_dict()
                    d405_entry = None
                    if markers:
                        d405_entry = markers.get(135)
                        if d405_entry is None:
                            for _mid, mdata in markers.items():
                                info = mdata.get('info', {}) if isinstance(mdata, dict) else {}
                                if info.get('name') == 'd405_back':
                                    d405_entry = mdata
                                    break
                    if d405_entry is not None:
                        pos = d405_entry.get('pos')
                        if pos is not None:
                            uv = self.dh.pixel_from_3d(np.array(pos, dtype=np.float32), self.latest_camera_info)
                            self.latest_d405_pixel = (float(uv[0]), float(uv[1]))
                            self.latest_d405_frame_idx = self._frame_idx
            except Exception:
                pass

            if self.latest_d405_pixel is not None:
                target_px = float(self.latest_d405_pixel[0])
                target_py = float(self.latest_d405_pixel[1])
            else:
                # Fallback: use fingertips average if available
                ft = getattr(self, 'latest_fingertips', None)
                if ft:
                    def proj(p):
                        x, y, z = float(p[0]), float(p[1]), float(p[2])
                        z = z if z > 1e-6 else 1e-6
                        u = (fx * x) / z + cx
                        v = (fy * y) / z + cy
                        return u, v
                    pixels = []
                    for side in ['left', 'right']:
                        d = ft.get(side)
                        if d and ('pos' in d):
                            u, v = proj(d['pos'])
                            pixels.append((u, v))
                    if pixels:
                        target_px = float(sum([u for (u, v) in pixels]) / len(pixels))
                        target_py = float(sum([v for (u, v) in pixels]) / len(pixels))
            if target_px is None or target_py is None:
                return

            target_px = float(np.clip(target_px, 0.0, max_w))
            target_py = float(np.clip(target_py, 0.0, max_h))

            # Desired location for D405 in raw (unrotated) frame so that in the rotated
            # display it appears middle-right.
            desired_px = float(self.d405_display_y_ratio) * max_w
            desired_py = max_h - (float(self.d405_display_x_ratio) * max_h)

            for _ in range(int(max_iters)):
                dx = target_px - desired_px
                dy = target_py - desired_py
                # Stop when close enough in pixels
                if (abs(dx) <= pixel_tol) and (abs(dy) <= pixel_tol):
                    break
                pan_delta = float(dx / fx)
                tilt_delta = float(-dy / fy)  # no bias for pre-alignment
                try:
                    nudge_head_pan_tilt_custom(pan_delta, tilt_delta, max_step=0.35)
                except Exception:
                    break
                time.sleep(0.06)
        except Exception:
            pass

    def _auto_detect_white_points(self, bgr_image, max_points=1):
        """Detect bright low-saturation regions as white points.

        Returns list of (x,y) pixel coordinates in image space.
        """
        try:
            if bgr_image is None:
                return []
            img = bgr_image
            h, w = img.shape[:2]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Low saturation, high value
            lower = np.array([0, 0, 200], dtype=np.uint8)
            upper = np.array([180, 40, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return []
            cx, cy = w // 2, h // 2
            cands = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 20:
                    continue
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
                d = float(np.hypot(x - cx, y - cy))
                cands.append((area, -d, x, y))
            cands.sort(reverse=True)
            pts = [(x, y) for (_, _, x, y) in cands[:max_points]]
            return pts
        except Exception:
            return []

    def _select_point_from_crosses(self, bgr_image, crosses):
        """Given a set of LLM crosses, choose a good interior grasp pixel.

        Strategy:
        - Estimate dominant HSV color near the crosses.
        - Build a color mask around that dominant color and restrict it by the
          convex hull of the crosses to focus on the intended object.
        - Use distance transform to find the most interior pixel (far from edges).
        - Fallback to the mean of crosses if mask is empty.
        """
        try:
            if bgr_image is None or not crosses:
                return None
            img = bgr_image
            h, w = img.shape[:2]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(hsv)
            # Sample small patches around each cross to collect HSV samples
            hs, ss, vs = [], [], []
            k = 4
            for (x, y) in crosses:
                xi0 = max(0, int(x) - k); xi1 = min(w, int(x) + k + 1)
                yi0 = max(0, int(y) - k); yi1 = min(h, int(y) + k + 1)
                if (xi1 - xi0) <= 0 or (yi1 - yi0) <= 0:
                    continue
                hs.append(H[yi0:yi1, xi0:xi1].reshape(-1))
                ss.append(S[yi0:yi1, xi0:xi1].reshape(-1))
                vs.append(V[yi0:yi1, xi0:xi1].reshape(-1))
            if not hs:
                # fallback: average crosses
                mx = int(np.mean([p[0] for p in crosses])); my = int(np.mean([p[1] for p in crosses]))
                return (mx, my)
            hs = np.concatenate(hs); ss = np.concatenate(ss); vs = np.concatenate(vs)
            # Robust central tendency
            h_med = int(np.median(hs)); s_med = int(np.median(ss)); v_med = int(np.median(vs))
            # Build color threshold around median. Allowable spreads
            dh = 15
            ds = 60
            dv = 60
            lower = np.array([max(0, h_med - dh), max(0, s_med - ds), max(0, v_med - dv)], dtype=np.uint8)
            upper = np.array([min(180, h_med + dh), min(255, s_med + ds), min(255, v_med + dv)], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, lower, upper)
            # Build convex hull mask from crosses
            hull = None
            pts = np.array([[int(x), int(y)] for (x, y) in crosses], dtype=np.int32)
            if len(pts) >= 3:
                hull = cv2.convexHull(pts)
            else:
                hull = pts
            hull_mask = np.zeros((h, w), dtype=np.uint8)
            if hull is not None and len(hull) >= 3:
                cv2.fillConvexPoly(hull_mask, hull, 255, lineType=cv2.LINE_AA)
            elif len(hull) == 2:
                cv2.line(hull_mask, tuple(hull[0]), tuple(hull[1]), 255, thickness=9, lineType=cv2.LINE_AA)
            elif len(hull) == 1:
                cv2.circle(hull_mask, tuple(hull[0][0] if hull.ndim == 3 else hull[0]), 6, 255, -1, lineType=cv2.LINE_AA)
            # Combine masks and clean up
            mask = cv2.bitwise_and(color_mask, hull_mask)
            # Remove likely white background and black pixels
            # Treat near-white/grey as background too: low S, moderately high V
            white_lower = np.array([0, 0, 200], dtype=np.uint8)
            white_upper = np.array([180, 40, 255], dtype=np.uint8)
            lightgray_lower = np.array([0, 0, 160], dtype=np.uint8)
            lightgray_upper = np.array([180, 60, 255], dtype=np.uint8)
            black_lower = np.array([0, 0, 0], dtype=np.uint8)
            black_upper = np.array([180, 255, 40], dtype=np.uint8)
            white_mask_strict = cv2.inRange(hsv, white_lower, white_upper)
            white_mask_light = cv2.inRange(hsv, lightgray_lower, lightgray_upper)
            white_mask = cv2.bitwise_or(white_mask_strict, white_mask_light)
            black_mask = cv2.inRange(hsv, black_lower, black_upper)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(white_mask))
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(black_mask))
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            # Further erode to avoid edges when selecting interior
            mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)
            if cv2.countNonZero(mask) == 0:
                # fallback: mean of crosses
                mx = int(np.mean([p[0] for p in crosses])); my = int(np.mean([p[1] for p in crosses]))
                return (mx, my)
            # Combine cross density with interior distance for robust interior selection
            dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
            den = np.zeros((h, w), dtype=np.float32)
            for (x, y) in crosses:
                xi = int(np.clip(x, 0, w - 1)); yi = int(np.clip(y, 0, h - 1))
                den[yi, xi] += 1.0
            den = cv2.GaussianBlur(den, (0, 0), sigmaX=6, sigmaY=6)
            if den.max() > 0:
                den = den / den.max()
            if dist.max() > 0:
                dist = dist / dist.max()
            score = den * dist * ((mask > 0).astype(np.float32))
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(score)
            cx, cy = int(maxLoc[0]), int(maxLoc[1])
            return (cx, cy)
        except Exception:
            return None

    def _nearest_non_white_black(self, hsv_image, px, py, radius=5):
        """Find nearest pixel around (px,py) that is not white background nor black.

        If none found, returns the input (px,py).
        """
        h, w = hsv_image.shape[:2]
        px = int(np.clip(px, 0, w - 1)); py = int(np.clip(py, 0, h - 1))
        # Define simple white/black masks thresholds
        H, S, V = cv2.split(hsv_image)
        # Pixel-wise quick check function
        def is_bad(x, y):
            s = int(S[y, x]); v = int(V[y, x])
            # Exclude white and light-gray (low S, mid-high V), and very dark (black)
            return ((s <= 60 and v >= 160) or (v <= 40))
        if not is_bad(px, py):
            return px, py
        # Search in growing window
        for r in range(1, int(radius) + 1):
            x0 = max(0, px - r); x1 = min(w - 1, px + r)
            y0 = max(0, py - r); y1 = min(h - 1, py + r)
            # Check perimeter of the square window to keep it cheap
            for x in range(x0, x1 + 1):
                if not is_bad(x, y0):
                    return x, y0
                if not is_bad(x, y1):
                    return x, y1
            for y in range(y0, y1 + 1):
                if not is_bad(x0, y):
                    return x0, y
                if not is_bad(x1, y):
                    return x1, y
        return px, py

    def _display_to_sensor_coords(self, x_disp: float, y_disp: float, source_size=None):
        """Convert GUI/display coordinates to raw sensor pixel coordinates."""
        try:
            h_raw, w_raw = self.last_color_frame_bgr.shape[:2]
        except Exception:
            h_raw, w_raw = 480, 640
        if source_size is not None and len(source_size) == 2:
            disp_w, disp_h = float(source_size[0]), float(source_size[1])
        else:
            disp_w, disp_h = float(w_raw), float(h_raw)
        # Avoid division by zero
        disp_w = max(2.0, float(disp_w))
        disp_h = max(2.0, float(disp_h))
        # Normalized coordinates in [0,1]
        x_norm = float(np.clip(x_disp, 0.0, disp_w - 1.0)) / (disp_w - 1.0)
        y_norm = float(np.clip(y_disp, 0.0, disp_h - 1.0)) / (disp_h - 1.0)
        if getattr(self, 'current_camera', 'D405') == 'D435i':
            # GUI shows a 90° clockwise rotation: display_x corresponds to raw row (inverted),
            # display_y maps directly to raw column.
            px = y_norm * (float(w_raw) - 1.0)
            py = (1.0 - x_norm) * (float(h_raw) - 1.0)
        else:
            px = x_norm * (float(w_raw) - 1.0)
            py = y_norm * (float(h_raw) - 1.0)
        px_i = int(np.clip(round(px), 0, max(0, w_raw - 1)))
        py_i = int(np.clip(round(py), 0, max(0, h_raw - 1)))
        return px_i, py_i

    def _choose_point_from_crosses_by_density(self, crosses, radius=8):
        """Pick a point where RoboPoint crosses are densest.

        - If exact duplicates exist, choose the most frequent coordinate.
        - Otherwise, for each point, count neighbors within `radius` pixels and
          select the cluster with the highest count; return its centroid.
        """
        try:
            if not crosses:
                return None
            pts = [(int(x), int(y)) for (x, y) in crosses]
            # 1) Exact duplicates (mode)
            from collections import Counter
            cnt = Counter(pts)
            ((mx, my), maxc) = max(cnt.items(), key=lambda kv: kv[1])
            if maxc >= 2:
                return (mx, my)
            # 2) Radius-based density
            pts_np = np.array(pts, dtype=np.int32)
            best_idx = -1
            best_count = -1
            best_subset = None
            r2 = radius * radius
            for i in range(len(pts_np)):
                dx = pts_np[:, 0] - pts_np[i, 0]
                dy = pts_np[:, 1] - pts_np[i, 1]
                mask = (dx * dx + dy * dy) <= r2
                count = int(np.count_nonzero(mask))
                if count > best_count:
                    best_count = count
                    best_idx = i
                    best_subset = pts_np[mask]
            if best_subset is not None and best_count > 0:
                cx = int(round(np.mean(best_subset[:, 0])))
                cy = int(round(np.mean(best_subset[:, 1])))
                return (cx, cy)
            return pts[0]
        except Exception:
            return None

    def _cluster_crosses(self, crosses):
        """Group crosses into clusters using KMeans based on spatial proximity."""
        import cv2
        if not crosses:
            return [[]]
        
        pts = np.array([[int(x), int(y)] for (x, y) in crosses], dtype=np.float32)
        n = len(pts)
        K = 1
        if n >= 24:
            K = 3
        elif n >= 10:
            K = 2
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        flags = cv2.KMEANS_PP_CENTERS
        compactness, labels, centers = cv2.kmeans(pts, K, None, criteria, 3, flags)
        
        clusters = []
        for k in range(K):
            mask = (labels.ravel() == k)
            sub = pts[mask].astype(np.int32)
            if len(sub) > 0:
                clusters.append([(int(x), int(y)) for (x, y) in sub])
        
        # Sort by cluster size, larger clusters first
        clusters.sort(key=lambda c: -len(c))
        return clusters

    def clean_llm_points(self):
        """Clear annotated white points and reset tracking without popups."""
        try:
            # Clear all LLM-derived points and related state
            self.latest_llm_points = []
            self.latest_llm_points_size = None
            self.llm_clusters = None
            self.llm_cluster_idx = -1

            # Reset tracking state
            self.llm_tracked_px = None
            self.llm_tracked_py = None
            self.llm_tracker = None
            self.user_clicked_position = False

            # Stop any running grasp controller (arm_motion) and legacy white-point publisher
            arm_running = (self.arm_grasp_proc.proc is not None) and (self.arm_grasp_proc.proc.poll() is None)
            white_running = (self.white_point_proc.proc is not None) and (self.white_point_proc.proc.poll() is None)
            try:
                if arm_running:
                    self.arm_grasp_proc.stop()
            except Exception:
                pass
            try:
                if white_running:
                    self.white_point_proc.stop()
            except Exception:
                pass

            if arm_running or white_running:
                self.llm_status.setText("LLM Grasping: Stopped, points cleaned")
            else:
                self.llm_status.setText("LLM Grasping: Points cleaned")

            # Update button states if needed
            self.btn_start_llm.setEnabled(True)
            self.btn_stop_llm.setEnabled(False)
        except Exception as e:
            logger.error(f"Failed to clean points: {e}")

    def on_image_clicked(self, x, y):
        """Handle mouse clicks on the image to set white point position."""
        try:
            # Coordinates from ImageLabel are in the displayed image space.
            # For D405, the display matches the sensor orientation.
            # For D435i, we display a 90-degree clockwise rotated image.
            # Map clicks on the rotated display back to the sensor coordinates.
            px, py = x, y
            if getattr(self, 'current_camera', 'D405') == 'D435i' and \
               hasattr(self, 'last_color_frame_bgr') and self.last_color_frame_bgr is not None:
                h, w = self.last_color_frame_bgr.shape[:2]
                # Display rotation used: x_display = h-1 - y_orig, y_display = x_orig
                # Inverse mapping from display -> original:
                # x_orig = y_display, y_orig = h-1 - x_display
                px, py = int(y), int(h - 1 - x)
            
            # Check if we have valid camera frame dimensions for coordinate validation
            if hasattr(self, 'last_color_frame_bgr') and self.last_color_frame_bgr is not None:
                h, w = self.last_color_frame_bgr.shape[:2]
                # Clamp coordinates to camera frame bounds
                px = max(0, min(px, w - 1))
                py = max(0, min(py, h - 1))
            
            # Avoid white/gray/black pixels if possible
            if hasattr(self, 'last_color_frame_bgr') and self.last_color_frame_bgr is not None:
                hsv = cv2.cvtColor(self.last_color_frame_bgr, cv2.COLOR_BGR2HSV)
                px, py = self._nearest_non_white_black(hsv, px, py, radius=6)
            
            # Update tracking target
            self.llm_tracked_px, self.llm_tracked_py = int(px), int(py)
            self.llm_tracker = None  # Reset tracker to re-initialize with new point
            self.user_clicked_position = True  # Mark that user manually set the position
            
            # Restart arm grasp controller when session is active
            arm_running = (self.arm_grasp_proc.proc is not None) and (self.arm_grasp_proc.proc.poll() is None)
            if arm_running:
                try:
                    self.arm_grasp_proc.stop()
                    time.sleep(0.1)
                except Exception:
                    pass
            if self.llm_session_active:
                self.arm_grasp_proc.start(self.llm_tracked_px, self.llm_tracked_py, self.use_remote_yolo)
                self.llm_status.setText(f"LLM Grasping: Running at ({self.llm_tracked_px}, {self.llm_tracked_py})")
                self.btn_stop_llm.setEnabled(True)
            else:
                self.llm_status.setText(f"LLM Grasping: Target set to ({self.llm_tracked_px}, {self.llm_tracked_py})")
            
            # If navigation session is active, (re)start base navigation with this point
            if self.nav_session_active:
                try:
                    if self.base_nav_proc.proc is not None and self.base_nav_proc.proc.poll() is None:
                        self.base_nav_proc.stop()
                except Exception:
                    pass
                # Use fixed options when (re)starting navigation after click
                stop_dist = 1.0
                self.base_nav_proc.start(self.llm_tracked_px, self.llm_tracked_py, self.use_remote_stream, stop_dist, True,
                                         tilt_down_negative=True, invert_yaw=True)
                self.nav_status.setText(f"LLM Navigation: Running towards ({self.llm_tracked_px}, {self.llm_tracked_py})")
                self.nav_stop_btn.setEnabled(True)
                
        except Exception as e:
            logger.error(f"Failed to handle image click: {e}")

    def on_head_tilt_changed(self, value_deg: float):
        """Debounce and then set the head tilt angle to the requested degrees."""
        try:
            self._pending_tilt_deg = float(value_deg)
            # Restart debounce timer
            self._tilt_apply_timer.stop(); self._tilt_apply_timer.start()
        except Exception:
            pass

    def _apply_pending_tilt(self):
        try:
            if set_head_tilt_deg_custom is not None:
                set_head_tilt_deg_custom(float(self._pending_tilt_deg))
        except Exception as e:
            logger.warning(f"Head tilt set failed: {e}")

    def start_camera(self):
        # Stop remote subscriber if running locally
        if not self.use_remote_stream and self.camera_thread is not None:
            try:
                self.camera_thread.stop(); self.camera_thread.wait(500)
            except Exception:
                pass
            self.camera_thread = None

        # Start direct RealSense capture (fixed to D435i)
        try:
            cam = 'D435i'
            exposure = self.exposure_selector.currentText()
            # Update current camera for display orientation handling
            self.current_camera = cam
            if self.direct_camera_thread is not None:
                return
            self.direct_camera_thread = DirectCameraThread(camera=cam, exposure=exposure)
            self.direct_camera_thread.frame_received.connect(self.on_camera_frame)
            self.direct_camera_thread.status_changed.connect(self.on_camera_status)
            self.direct_camera_thread.start()
            self.btn_start_sender.setEnabled(False); self.btn_stop_sender.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Camera", f"Failed to start: {e}")

    def stop_camera(self):
        try:
            if self.direct_camera_thread is not None:
                self.direct_camera_thread.stop(); self.direct_camera_thread.wait(1000)
        finally:
            self.direct_camera_thread = None
            self.btn_start_sender.setEnabled(True); self.btn_stop_sender.setEnabled(False)

    def start_services(self):
        try:
            self.service_status.setText("Controller: Starting... | Model Worker: Starting...")
            self.server_process.start_controller()
            time.sleep(3)
            self.server_process.start_model_worker(controller_url=self.server_process.controller_url,
                                                   model_path=self.model_path,
                                                   load_4bit=self.load_4bit)
            self.start_services_btn.setEnabled(False); self.stop_services_btn.setEnabled(True)
            QTimer.singleShot(10000, self.refresh_model_list)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start services: {e}")

    def stop_services(self):
        self.server_process.stop_all()
        self.service_status.setText("Controller: Stopped | Model Worker: Stopped")
        self.start_services_btn.setEnabled(True)
        self.stop_services_btn.setEnabled(False)
        self.send_button.setEnabled(False)

    def start_llm_navigation(self):
        try:
            self.nav_session_active = True
            # Do not change or restart camera when starting navigation; keep current stream.
            try:
                # Ensure any grasp head-tracking nudge timer is stopped to avoid extra head moves
                self.head_track_timer.stop()
            except Exception:
                pass
            # Go to a fixed start pose (head/gripper forward)
            try:
                if go_to_start_pose_custom is not None:
                    # Match vlservoing.py: set head pan forward (0 rad) and
                    # use current UI tilt value (deg) for head tilt
                    tilt_deg = float(self.head_tilt_deg.value()) if hasattr(self, 'head_tilt_deg') else 0.0
                    go_to_start_pose_custom(head_tilt_deg=tilt_deg, head_pan_rad=0.0)
                if point_head_to_gripper_custom is not None:
                    point_head_to_gripper_custom()
                # Do not lock head/gripper so navigation can control head smoothly
            except Exception as e:
                logger.warning(f"Failed to set start pose: {e}")
            # Determine target point
            px_py = None
            if (self.llm_tracked_px is not None) and (self.llm_tracked_py is not None):
                px_py = (int(self.llm_tracked_px), int(self.llm_tracked_py))
            elif self.latest_llm_points:
                cx = int(np.mean([p[0] for p in self.latest_llm_points])); cy = int(np.mean([p[1] for p in self.latest_llm_points]))
                px_py = (cx, cy)
            if px_py is not None:
                # Fixed stop distance and control options
                stop_dist = 1.0
                try:
                    if self.base_nav_proc.proc is not None and self.base_nav_proc.proc.poll() is None:
                        self.base_nav_proc.stop()
                except Exception:
                    pass
                # Fixed controls: tilt-only, down-negative, inverted yaw
                self.base_nav_proc.start(px_py[0], px_py[1], self.use_remote_stream, stop_dist, True,
                                         tilt_down_negative=True, invert_yaw=True)
                self.nav_status.setText(f"LLM Navigation: Running at ({px_py[0]}, {px_py[1]})")
                self.nav_start_btn.setEnabled(True)
                self.nav_stop_btn.setEnabled(True)
            else:
                self.nav_status.setText("LLM Navigation: Waiting for white point")
                self.nav_stop_btn.setEnabled(True)
        except Exception as e:
            logger.error(f"LLM Navigation start error: {e}")
            self.nav_status.setText("LLM Navigation: Error starting")

    def stop_llm_navigation(self):
        try:
            self.base_nav_proc.stop()
            try:
                if unlock_head_and_gripper_custom is not None:
                    unlock_head_and_gripper_custom()
            except Exception:
                pass
        finally:
            self.nav_session_active = False
            self.nav_start_btn.setEnabled(True)
            self.nav_stop_btn.setEnabled(False)
            self.nav_status.setText("LLM Navigation: Not Running")
            try:
                self.head_track_timer.stop()
            except Exception:
                pass

    def check_service_status(self):
        """Poll lightweight status with very short timeouts to avoid UI stalls."""
        ctrl_status = "Not Running"
        try:
            response = requests.post(f"{self.server_process.controller_url}/list_models", timeout=0.2)
            if response.status_code == 200:
                ctrl_status = "Running"
            else:
                ctrl_status = "Error"
        except Exception:
            if self.server_process.controller_process and self.server_process.controller_process.poll() is None:
                ctrl_status = "Starting..."
            else:
                ctrl_status = "Not Running"
        worker_status = "Running" if len(self.models) > 0 else "No Models"
        try:
            self.service_status.setText(f"Controller: {ctrl_status} | Model Worker: {worker_status}")
        except Exception:
            pass
        self.send_button.setEnabled((ctrl_status == "Running") and len(self.models) > 0)

    def refresh_model_list(self):
        try:
            base = self.server_process.controller_url
            ret = requests.post(f"{base}/refresh_all_workers", timeout=5)
            if ret.status_code == 200:
                ret = requests.post(f"{base}/list_models", timeout=5)
                if ret.status_code == 200:
                    self.models = ret.json().get("models", [])
                    # Populate selector if available
                    try:
                        self.model_selector.clear()
                        self.model_selector.addItems(self.models)
                    except Exception:
                        pass
                    # Update combined service status line
                    try:
                        worker_status = "Running" if self.models else "No Models"
                        # Preserve current controller status string if available
                        current = getattr(self, 'service_status', None)
                        ctrl_str = "Controller: Running"
                        if current is not None and current.text():
                            # try to keep previous controller status segment
                            parts = current.text().split('|')
                            if parts:
                                ctrl_str = parts[0].strip()
                        self.service_status.setText(f"{ctrl_str} | Model Worker: {worker_status}")
                        if self.models:
                            logger.info(f"Loaded models: {self.models}")
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Failed to refresh model list: {e}")

    def update_temperature(self, value):
        self.temperature_label.setText(f"{value/100.0:.1f}")

    def update_top_p(self, value):
        self.top_p_label.setText(f"{value/100.0:.1f}")

    def send_message(self):
        """Send message to model (mirrors D405 GUI behavior)."""
        text = self.text_input.text().strip()
        image = getattr(self.image_label, 'original_image', None)
        # If no manually loaded image, fall back to latest camera frame
        # so the LLM receives an image and can return annotated results.
        if image is None:
            try:
                if getattr(self, 'last_color_frame_bgr', None) is not None:
                    import cv2
                    bgr = self.last_color_frame_bgr
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    # For D435i, match the GUI by rotating 90 degrees clockwise
                    try:
                        if getattr(self, 'current_camera', 'D405') == 'D435i':
                            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
                    except Exception:
                        pass
                    from PIL import Image
                    image = Image.fromarray(rgb)
            except Exception:
                image = None
        if not text and image is None:
            return
        if not self.models:
            QMessageBox.warning(self, "Warning", "No models available. Please wait for services to start.")
            return
        # Add prompt suffix like D405 GUI
        text += " Your answer should be formatted as a list of tuples, " \
                "i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the " \
                "x and y coordinates of a point satisfying the conditions above." \
                " The coordinates should be between 0 and 1, indicating the " \
                "normalized pixel locations of the points in the image."

        # Prepare conversation state
        if image is not None:
            if '<image>' not in text:
                text = '<image>\n' + text
            content = (text, image, 'Pad')  # default to Pad processing
            self.conversation_state = default_conversation.copy()
        else:
            content = text

        self.conversation_state.append_message(self.conversation_state.roles[0], content)
        self.conversation_state.append_message(self.conversation_state.roles[1], None)

        # Add user message to chat
        try:
            self.chat_widget.append(f"\nUser:\n{text}\n")
        except Exception:
            pass

        # Clear input and disable send during processing
        self.text_input.clear()
        self.send_button.setEnabled(False)
        self.send_button.setText("Processing...")

        # Send request
        self.send_request_to_worker()

    def find_vectors(self, text):
        import re
        pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
        matches = re.findall(pattern, text)
        vectors = []
        for match in matches:
            vector = [float(num) if '.' in num else int(num) for num in match.split(',')]
            vectors.append(vector)
        return vectors

    def visualize_2d(self, img, points, bboxes, scale, cross_size=9, cross_width=4):
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

    def process_response_visualization(self, response):
        try:
            vectors = self.find_vectors(response)
            vectors_2d = [vec for vec in vectors if len(vec) == 2]
            vectors_bbox = [vec for vec in vectors if len(vec) == 4]
            if not vectors_2d and not vectors_bbox:
                return response
            pil_images, images, transforms = self.conversation_state.get_images()
            if not pil_images:
                return response
            image = pil_images[-1].copy()
            transform = transforms[-1]
            # Use the padded model input size for de-normalization (Pad mode = 640x480)
            # LLM outputs are normalized in the padded image space
            ref_w, ref_h = 640, 480
            new_vectors = []
            for x, y in vectors_2d:
                if isinstance(x, float) and x <= 1:
                    x = x * ref_w
                    y = y * ref_h
                # For Pad mode the displayed chat image is already the 640x480 padded image,
                # so no additional transform is needed for visualization.
                new_vectors.append((x, y))
            new_bbox = []
            for x1, y1, x2, y2 in vectors_bbox:
                if isinstance(x1, float) and x1 <= 1:
                    x1, y1, x2, y2 = x1 * ref_w, y1 * ref_h, x2 * ref_w, y2 * ref_h
                # Same reasoning: draw directly in padded chat image space
                new_bbox.append((x1, y1, x2, y2))
            # Cache points for live overlay as in D405 GUI
            try:
                # Store points in sensor coordinates (convert from GUI/padded image space)
                w_vis, h_vis = ref_w, ref_h
                pts_sensor = []
                for (x, y) in new_vectors:
                    xi = int(round(x)); yi = int(round(y))
                    if 0 <= xi < w_vis and 0 <= yi < h_vis:
                        sx, sy = self._display_to_sensor_coords(xi, yi, (w_vis, h_vis))
                        pts_sensor.append((sx, sy))
                self.latest_llm_points = pts_sensor
                self.latest_llm_points_size = (w_vis, h_vis)
                # Reset tracker so it re-initializes on next camera frame
                # Only reset position if user hasn't manually clicked
                if not self.user_clicked_position:
                    self.llm_tracker = None
                    self.llm_tracked_px = None
                    self.llm_tracked_py = None
                # Reset cluster state when new points are received
                self.llm_clusters = None
                self.llm_cluster_idx = -1

                # If an LLM session is active and the publisher isn't running yet,
                # start grasping now when points arrive.
                if self.llm_session_active:
                    has_arm_running = (self.arm_grasp_proc.proc is not None) and (self.arm_grasp_proc.proc.poll() is None)
                    if not has_arm_running and len(self.latest_llm_points) > 0:
                        chosen = self._choose_point_from_crosses_by_density(self.latest_llm_points, radius=8)
                        refined = self._select_point_from_crosses(getattr(self, 'last_color_frame_bgr', None), self.latest_llm_points)
                        if refined is not None:
                            chosen = refined if (chosen is None) else chosen
                        if chosen is not None:
                            px, py = int(chosen[0]), int(chosen[1])
                            try:
                                if self.last_color_frame_bgr is not None:
                                    hsv = cv2.cvtColor(self.last_color_frame_bgr, cv2.COLOR_BGR2HSV)
                                    px, py = self._nearest_non_white_black(hsv, px, py, radius=6)
                            except Exception:
                                pass
                            self.llm_tracked_px, self.llm_tracked_py = int(px), int(py)
                            self.llm_tracker = None
                            try:
                                self.arm_grasp_proc.start(self.llm_tracked_px, self.llm_tracked_py, self.use_remote_yolo)
                                self.llm_status.setText(f"LLM Grasping: Running at ({self.llm_tracked_px}, {self.llm_tracked_py})")
                                self.btn_stop_llm.setEnabled(True)
                            except Exception:
                                pass
            except Exception:
                pass
            annotated = self.visualize_2d(image, new_vectors, new_bbox, 1.0)
            return (response, annotated, 'Pad')
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return response

    def send_request_to_worker(self):
        try:
            # Determine conversation template by model
            model_name = self.model_selector.currentText() if hasattr(self, 'model_selector') else (self.models[0] if self.models else '')
            name_lower = model_name.lower() if isinstance(model_name, str) else ''
            if 'vicuna' in name_lower:
                template_name = 'vicuna_v1'
            elif 'llama' in name_lower:
                template_name = 'llava_llama_2'
            elif 'mistral' in name_lower:
                template_name = 'mistral_instruct'
            elif 'mpt' in name_lower:
                template_name = 'mpt'
            else:
                template_name = 'llava_v1'

            if len(self.conversation_state.messages) == 2:
                new_state = conv_templates[template_name].copy()
                new_state.append_message(new_state.roles[0], self.conversation_state.messages[-2][1])
                new_state.append_message(new_state.roles[1], None)
                self.conversation_state = new_state

            prompt = self.conversation_state.get_prompt()
            pil_images, images, transforms = self.conversation_state.get_images()

            request_data = {
                'model': model_name,
                'prompt': prompt,
                'temperature': self.temperature_slider.value() / 100.0,
                'top_p': self.top_p_slider.value() / 100.0,
                'max_new_tokens': min(self.max_tokens_spin.value(), 1536),
                'stop': self.conversation_state.sep if self.conversation_state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else self.conversation_state.sep2,
                'images': images,
            }
            self.worker_thread.set_request_data(request_data)
            self.worker_thread.start()
        except Exception as e:
            self.handle_error(f"Failed to send request: {e}")

    def handle_response(self, response):
        self._pending_response_text = response

    def handle_error(self, error_message):
        try:
            self.chat_widget.add_message("System", f"Error: {error_message}")
        except Exception:
            pass
        self._pending_response_text = None
        self.request_finished()

    def request_finished(self):
        if self._pending_response_text is not None:
            # Save assistant message into conversation state and render visualization like D405 GUI
            try:
                self.conversation_state.messages[-1] = [self.conversation_state.roles[1], self._pending_response_text]
            except Exception:
                pass
            processed = self.process_response_visualization(self._pending_response_text)
            if isinstance(processed, tuple):
                # (text, annotated_image, mode)
                self.chat_widget.add_message("Assistant", processed, is_image=True)
            else:
                self.chat_widget.add_message("Assistant", processed)
            self._pending_response_text = None

        self.send_button.setEnabled(True)
        self.send_button.setText("Send")

    def regenerate_response(self):
        pass

    def clear_chat(self):
        try:
            self.chat_widget.clear()
            self.conversation_state = default_conversation.copy()
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            if self.camera_thread is not None:
                self.camera_thread.stop(); self.camera_thread.wait(1000)
            if self.direct_camera_thread is not None:
                self.direct_camera_thread.stop(); self.direct_camera_thread.wait(1000)
            if self.yolo_thread is not None:
                self.yolo_thread.stop(); self.yolo_thread.wait(1000)
            # Stop nav/grasp external processes
            try:
                self.base_nav_proc.stop()
            except Exception:
                pass
            self.stop_services()
        except Exception:
            pass
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="RoboPoint Visual Servoing GUI")
    parser.add_argument("--controller-url", type=str, default="http://10.0.0.1:11000", help="Controller URL")
    parser.add_argument("--model-path", type=str, default="wentao-yuan/robopoint-v1-vicuna-v1.5-13b", help="Model path")
    parser.add_argument("--load-4bit", action="store_true", default=True, help="Load model in 4-bit mode")
    parser.add_argument('-r', '--remote', action='store_true', help='Receive camera images from a remote robot (subscribe to robot IP).')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("RoboPoint (Visual Servoing)")
    window = RoboPointMainWindow(controller_url=args.controller_url, autostart=False,
                                 model_path=args.model_path, load_4bit=args.load_4bit,
                                 use_remote_stream=args.remote)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
