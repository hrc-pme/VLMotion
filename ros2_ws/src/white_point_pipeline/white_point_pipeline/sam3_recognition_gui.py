#!/usr/bin/env python3
"""Camera-driven SAM3 counting and intervention GUI."""

import argparse
import base64
import datetime
import json
import os
import sys
import time
from pathlib import Path


def configure_qt_environment():
    """Keep OpenCV's bundled Qt plugins from shadowing system PyQt5."""
    platform_paths = (
        "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms",
        "/usr/lib/x86_64-linux-gnu/qt/plugins/platforms",
        "/usr/lib/qt/plugins/platforms",
    )
    selected = next((path for path in platform_paths if os.path.isdir(path)), None)
    if selected:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = selected
        os.environ["QT_PLUGIN_PATH"] = str(Path(selected).parent)
    else:
        for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
            value = os.environ.get(key, "")
            if "cv2" in value or "opencv" in value.lower():
                os.environ.pop(key, None)
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


configure_qt_environment()

import cv2

# Importing cv2 may point Qt back to cv2/qt/plugins, so restore the system path.
configure_qt_environment()

import numpy as np
import rclpy
import requests
from cv_bridge import CvBridge
from PyQt5.QtCore import QObject, QThread, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Bool, String


GUI_COUNTING_METHOD = "sam3_calibrated"
GUI_INTERVENTION_PMATCH_THRESHOLD = 0.30
GUI_OUTPUT_SCHEMA_VERSION = "1.1"


def _normalized_point_to_pixels(point, image_width, image_height):
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    try:
        x_value, y_value = (float(point[0]), float(point[1]))
    except (TypeError, ValueError):
        return None
    return [
        int(round(min(max(x_value, 0.0), 1.0) * max(image_width - 1, 0))),
        int(round(min(max(y_value, 0.0), 1.0) * max(image_height - 1, 0))),
    ]


def _normalized_bbox_to_pixels(bbox, image_width, image_height):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    return [
        int(round(min(max(x1, 0.0), 1.0) * image_width)),
        int(round(min(max(y1, 0.0), 1.0) * image_height)),
        int(round(min(max(x2, 0.0), 1.0) * image_width)),
        int(round(min(max(y2, 0.0), 1.0) * image_height)),
    ]


def add_gui_output_contract(result, image, instrument_type):
    """Add a stable, explicit instance/count contract without changing inference."""
    image_height, image_width = image.shape[:2]
    coordinate_system = {
        "frame_id": "input_image_pixels",
        "origin": "top_left",
        "x_axis": "right",
        "y_axis": "down",
        "units": "pixel",
        "image_width": int(image_width),
        "image_height": int(image_height),
    }
    instances = []
    for mask in result.get("sam3_masks", []):
        if mask.get("status") != "detected":
            continue
        center_normalized = mask.get("center")
        bbox_normalized = mask.get("bbox")
        confidence = mask.get("confidence", mask.get("query_match_score"))
        instances.append({
            "instrument_id": str(mask.get("label", "-")),
            "instrument_type": str(instrument_type),
            "position": {
                "center_px": _normalized_point_to_pixels(
                    center_normalized, image_width, image_height
                ),
                "center_normalized": center_normalized,
                "bbox_px": _normalized_bbox_to_pixels(
                    bbox_normalized, image_width, image_height
                ),
                "bbox_normalized": bbox_normalized,
            },
            "confidence": confidence,
            "coordinate_frame": coordinate_system["frame_id"],
            "status": mask.get("status", "detected"),
        })

    result.update({
        "output_schema_version": GUI_OUTPUT_SCHEMA_VERSION,
        "instrument_type": str(instrument_type),
        "count": int(result.get("estimated_count", len(instances))),
        "coordinate_system": coordinate_system,
        "instrument_instances": instances,
        "instrument_id_scope": "single_inference_frame",
    })
    return result


def _depth_value_scale(encoding):
    if encoding in {"16UC1", "mono16"}:
        return 0.001
    if encoding == "32FC1":
        return 1.0
    return None


def _instance_depth_and_position(instance, depth_image, encoding, camera_matrix):
    """Estimate an instance center in the optical frame from aligned depth."""
    scale = _depth_value_scale(encoding)
    center = instance.get("position", {}).get("center_normalized")
    bbox = instance.get("position", {}).get("bbox_normalized")
    if scale is None or center is None or len(center) != 2:
        return None

    depth_height, depth_width = depth_image.shape[:2]
    u = int(round(float(center[0]) * max(depth_width - 1, 0)))
    v = int(round(float(center[1]) * max(depth_height - 1, 0)))
    u = min(max(u, 0), max(depth_width - 1, 0))
    v = min(max(v, 0), max(depth_height - 1, 0))

    radius = 5
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        bbox_width = abs(float(bbox[2]) - float(bbox[0])) * depth_width
        bbox_height = abs(float(bbox[3]) - float(bbox[1])) * depth_height
        radius = int(round(min(bbox_width, bbox_height) * 0.08))
        radius = min(max(radius, 3), 15)

    y1, y2 = max(v - radius, 0), min(v + radius + 1, depth_height)
    x1, x2 = max(u - radius, 0), min(u + radius + 1, depth_width)
    patch_m = np.asarray(depth_image[y1:y2, x1:x2], dtype=np.float64) * scale
    valid = patch_m[np.isfinite(patch_m) & (patch_m >= 0.1) & (patch_m <= 6.0)]
    depth_method = "median_center_patch"
    sample_u, sample_v = u, v
    if valid.size == 0 and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        bx1 = min(max(int(float(bbox[0]) * depth_width), 0), depth_width - 1)
        by1 = min(max(int(float(bbox[1]) * depth_height), 0), depth_height - 1)
        bx2 = min(max(int(np.ceil(float(bbox[2]) * depth_width)), bx1 + 1), depth_width)
        by2 = min(max(int(np.ceil(float(bbox[3]) * depth_height)), by1 + 1), depth_height)
        bbox_m = np.asarray(depth_image[by1:by2, bx1:bx2], dtype=np.float64) * scale
        valid_mask = np.isfinite(bbox_m) & (bbox_m >= 0.1) & (bbox_m <= 6.0)
        valid_y, valid_x = np.nonzero(valid_mask)
        if valid_y.size:
            image_x = valid_x + bx1
            image_y = valid_y + by1
            distances = (image_x - u) ** 2 + (image_y - v) ** 2
            sample_count = min(25, valid_y.size)
            nearest = np.argpartition(distances, sample_count - 1)[:sample_count]
            valid = bbox_m[valid_y[nearest], valid_x[nearest]]
            closest = nearest[int(np.argmin(distances[nearest]))]
            sample_u = int(image_x[closest])
            sample_v = int(image_y[closest])
            depth_method = "median_nearest_valid_in_bbox"
    if valid.size == 0:
        return None

    depth_m = float(np.median(valid))
    output = {
        "depth_px": [u, v],
        "depth_sample_px": [sample_u, sample_v],
        "depth_m": round(depth_m, 4),
        "depth_method": depth_method,
        "depth_valid_samples": int(valid.size),
        "depth_patch_radius_px": int(radius),
    }
    if camera_matrix is not None:
        matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
        cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
        if fx > 0.0 and fy > 0.0:
            output["center_camera_m"] = [
                round((u - cx) * depth_m / fx, 4),
                round((v - cy) * depth_m / fy, 4),
                round(depth_m, 4),
            ]
    return output


def add_depth_capture_metadata(result, depth_snapshot):
    """Attach aligned depth and per-instance camera-frame positions."""
    snapshot = depth_snapshot or {}
    depth_image = snapshot.get("depth_image")
    camera_matrix = snapshot.get("camera_matrix")
    encoding = snapshot.get("encoding")
    computed_count = 0
    if depth_image is not None:
        for instance in result.get("instrument_instances", []):
            depth_result = _instance_depth_and_position(
                instance, depth_image, encoding, camera_matrix
            )
            if depth_result is None:
                instance["depth_status"] = "no_valid_depth"
                continue
            instance["position"].update(depth_result)
            instance["depth_status"] = "valid"
            if "center_camera_m" in depth_result:
                instance["camera_coordinate_frame"] = snapshot.get("frame_id")
                computed_count += 1

    result["depth_capture"] = {
        "available": depth_image is not None,
        "aligned_to_color": True,
        "encoding": encoding,
        "units": (
            "millimeter"
            if encoding in {"16UC1", "mono16"}
            else "meter" if encoding == "32FC1" else None
        ),
        "shape": (
            None
            if depth_image is None
            else [int(value) for value in depth_image.shape]
        ),
        "stamp": snapshot.get("stamp"),
        "frame_id": snapshot.get("frame_id"),
        "camera_matrix": (
            None
            if camera_matrix is None
            else np.asarray(camera_matrix, dtype=float).tolist()
        ),
        "position_3d_computed": computed_count > 0,
        "instances_with_3d_position": computed_count,
        "instance_depth_method": "center_patch_with_bbox_fallback",
    }
    return result


def cv_to_pixmap(image, width, height):
    if image is None:
        return QPixmap()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage).scaled(
        max(1, width),
        max(1, height),
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )


def parse_worker_stream(response, prompt):
    response.raise_for_status()
    final_text = ""
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if not chunk:
            continue
        data = json.loads(chunk.decode("utf-8"))
        if data.get("error_code", 0) != 0:
            raise RuntimeError(data.get("text", "Worker returned an error"))
        final_text = data.get("text", "")
    if final_text.startswith(prompt):
        final_text = final_text[len(prompt):]
    if not final_text.strip():
        raise RuntimeError("Worker returned an empty response")
    return json.loads(final_text.strip())




class SignalBridge(QObject):
    image_received = pyqtSignal(object)
    depth_received = pyqtSignal(object)


class CameraNode(Node):
    def __init__(self, bridge):
        super().__init__("sam3_recognition_gui")
        self.bridge = bridge
        self.cv_bridge = CvBridge()
        self.latest_image = None
        self.latest_depth_image = None
        self.latest_depth_encoding = None
        self.latest_depth_stamp = None
        self.camera_matrix = None
        self.camera_frame = None
        self.declare_parameter("color_topic", "/d435/color/image_raw")
        self.declare_parameter(
            "depth_topic", "/d435/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter(
            "camera_info_topic", "/d435/color/camera_info"
        )
        color_topic = self.get_parameter("color_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        if str(color_topic).endswith("/compressed"):
            self.subscription = self.create_subscription(
                CompressedImage, color_topic, self._compressed_callback, 10
            )
        else:
            self.subscription = self.create_subscription(
                Image, color_topic, self._image_callback, 10
            )
        self.depth_subscription = self.create_subscription(
            Image, depth_topic, self._depth_callback, 10
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo, camera_info_topic, self._camera_info_callback, 10
        )
        result_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.result_pub = self.create_publisher(
            String, "/sam3_recognition/result", result_qos
        )
        self.intervention_pub = self.create_publisher(
            Bool, "/sam3_recognition/requires_intervention", 10
        )
        self.overlay_pub = self.create_publisher(
            Image, "/sam3_recognition/overlay", 10
        )
        self.get_logger().info(f"SAM3 GUI subscribed to {color_topic}")
        self.get_logger().info(
            f"SAM3 GUI subscribed to aligned depth {depth_topic}"
        )
        self.get_logger().info(
            f"SAM3 GUI subscribed to camera intrinsics {camera_info_topic}"
        )

    def _emit(self, image):
        if image is None:
            return
        self.latest_image = image.copy()
        self.bridge.image_received.emit(image.copy())

    def _image_callback(self, msg):
        try:
            self._emit(self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"))
        except Exception as exc:
            self.get_logger().error(f"Failed to decode camera image: {exc}")

    def _compressed_callback(self, msg):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        self._emit(cv2.imdecode(data, cv2.IMREAD_COLOR))

    def _depth_callback(self, msg):
        try:
            depth = self.cv_bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
            self.latest_depth_image = np.asarray(depth).copy()
            self.latest_depth_encoding = str(msg.encoding)
            self.latest_depth_stamp = {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            }
            if not self.camera_frame:
                self.camera_frame = str(msg.header.frame_id)
            self.bridge.depth_received.emit({
                "shape": tuple(int(value) for value in self.latest_depth_image.shape),
                "encoding": self.latest_depth_encoding,
                "frame_id": self.camera_frame,
            })
        except Exception as exc:
            self.get_logger().error(f"Failed to decode aligned depth: {exc}")

    def _camera_info_callback(self, msg):
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.camera_frame = str(msg.header.frame_id)

    def get_depth_snapshot(self):
        """Return a copy of the latest aligned depth and its camera metadata."""
        return {
            "depth_image": (
                None
                if self.latest_depth_image is None
                else self.latest_depth_image.copy()
            ),
            "encoding": self.latest_depth_encoding,
            "stamp": (
                None
                if self.latest_depth_stamp is None
                else dict(self.latest_depth_stamp)
            ),
            "camera_matrix": (
                None
                if self.camera_matrix is None
                else self.camera_matrix.copy()
            ),
            "frame_id": self.camera_frame,
        }

    def publish_result(self, result, overlay):
        result_msg = String()
        result_msg.data = json.dumps(result, ensure_ascii=False)
        self.result_pub.publish(result_msg)
        intervention_msg = Bool()
        intervention_msg.data = bool(result.get("requires_intervention", False))
        self.intervention_pub.publish(intervention_msg)
        if overlay is not None:
            self.overlay_pub.publish(self.cv_bridge.cv2_to_imgmsg(overlay, "bgr8"))


class RosSpinThread(QThread):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.running = True

    def run(self):
        try:
            while self.running and rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.1)
        except ExternalShutdownException:
            pass

    def stop(self):
        self.running = False
        self.wait(1500)


class InferenceThread(QThread):
    succeeded = pyqtSignal(dict, object, str)
    failed = pyqtSignal(str)
    worker_ready = pyqtSignal(str)

    def __init__(
        self,
        image,
        target_object,
        display_target,
        controller_url,
        model_name,
        overlay_path,
        output_dir,
        depth_snapshot,
    ):
        super().__init__()
        self.image = image.copy()
        self.target_object = target_object
        self.display_target = display_target
        self.controller_url = controller_url.rstrip("/")
        self.model_name = model_name
        self.overlay_path = Path(overlay_path)
        self.output_dir = Path(output_dir)
        self.depth_snapshot = depth_snapshot or {}

    def run(self):
        try:
            worker_response = requests.post(
                self.controller_url + "/get_worker_address",
                json={"model": self.model_name},
                timeout=10,
            )
            worker_response.raise_for_status()
            worker_address = worker_response.json().get("address", "")
            if not worker_address:
                raise RuntimeError(f"No worker available for {self.model_name}")
            self.worker_ready.emit(worker_address)

            ok, encoded = cv2.imencode(
                ".png",
                self.image,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            )
            if not ok:
                raise RuntimeError("Could not encode the camera frame")
            image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
            counted_noun = (
                self.target_object
                if self.target_object.lower() == "scissors"
                else f"{self.target_object}s"
            )
            prompt = (
                f"USER: <image>\nHow many visible {counted_noun} "
                "are in this image?\nASSISTANT:"
            )
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 256,
                "images": [image_b64],
                "stop": "</s>",
                "counting_pass": "pass1",
                "counting_method": GUI_COUNTING_METHOD,
                "target_object": self.target_object,
                "coverage_repeats": 1,
            }
            request_started = time.time()
            response = requests.post(
                worker_address.rstrip("/") + "/worker_generate_stream",
                json=payload,
                stream=True,
                timeout=600,
            )
            result = parse_worker_stream(response, prompt)
            result["gui_target"] = self.display_target
            result["sam3_prompt"] = self.target_object
            result = add_gui_output_contract(
                result, self.image, self.display_target
            )
            result = add_depth_capture_metadata(result, self.depth_snapshot)

            overlay = None
            if int(result.get("estimated_count", 0)) > 0:
                worker_overlay = result.get("overlay_paths", {}).get(
                    result.get("deployed_overlay_key", "")
                )
                selected_overlay_path = (
                    Path(worker_overlay) if worker_overlay else self.overlay_path
                )
                for _ in range(20):
                    if (
                        selected_overlay_path.exists()
                        and selected_overlay_path.stat().st_mtime
                        >= request_started - 1.0
                    ):
                        overlay = cv2.imread(
                            str(selected_overlay_path), cv2.IMREAD_COLOR
                        )
                        if overlay is not None:
                            break
                    time.sleep(0.1)
            if overlay is None:
                overlay = self.image.copy()
                cv2.putText(
                    overlay,
                    "No SAM3 detections",
                    (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (30, 30, 230),
                    2,
                    cv2.LINE_AA,
                )

            run_dir = self._save_run(result, overlay)
            self.succeeded.emit(result, overlay, str(run_dir))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def _save_run(self, result, overlay):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self.output_dir / f"{stamp}_{self.display_target.replace(' ', '_')}"
        run_dir.mkdir(parents=True, exist_ok=False)
        cv2.imwrite(str(run_dir / "camera.jpg"), self.image)
        cv2.imwrite(str(run_dir / "sam3_overlay.jpg"), overlay)
        depth_image = self.depth_snapshot.get("depth_image")
        if depth_image is not None:
            np.save(run_dir / "aligned_depth.npy", depth_image)
            if depth_image.dtype == np.uint16:
                cv2.imwrite(str(run_dir / "aligned_depth.png"), depth_image)
        (run_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return run_dir


class ImagePanel(QFrame):
    def __init__(self, title):
        super().__init__()
        self.image = None
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.image_label = QLabel("Waiting for camera")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(420, 300)
        self.image_label.setStyleSheet("background: #16191d; color: #aeb4ba;")
        layout.addWidget(title_label)
        layout.addWidget(self.image_label, 1)

    def set_image(self, image):
        self.image = None if image is None else image.copy()
        self.refresh()

    def clear_image(self, message):
        self.image = None
        self.image_label.clear()
        self.image_label.setText(message)

    def refresh(self):
        if self.image is None:
            return
        self.image_label.setPixmap(
            cv_to_pixmap(
                self.image,
                self.image_label.width() - 4,
                self.image_label.height() - 4,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()


class MainWindow(QMainWindow):
    def __init__(
        self,
        node,
        bridge,
        controller_url,
        model_name,
        overlay_path,
        output_dir,
        default_target,
    ):
        super().__init__()
        self.node = node
        self.controller_url = controller_url
        self.model_name = model_name
        self.overlay_path = overlay_path
        self.output_dir = output_dir
        self.current_image = None
        self.frozen_image = None
        self.depth_available = False
        self.display_live = True
        self.inference_thread = None
        self.setWindowTitle("SAM3 Recognition and Intervention")
        self.resize(1480, 900)
        self._build_ui(default_target)
        bridge.image_received.connect(self._on_camera_image)
        bridge.depth_received.connect(self._on_depth_received)
        QTimer.singleShot(1000, self._check_controller)

    def _build_ui(self, default_target):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        toolbar = QHBoxLayout()
        title = QLabel("SAM3 Recognition")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        toolbar.addWidget(title)
        toolbar.addStretch()
        toolbar.addWidget(QLabel("Target"))
        self.target_combo = QComboBox()
        self.target_combo.setEditable(True)
        self.target_combo.addItems(["scissors", "fork"])
        self.target_combo.setCurrentText(default_target)
        self.target_combo.setMinimumWidth(170)
        toolbar.addWidget(self.target_combo)
        self.recognize_button = QPushButton("Recognize")
        self.recognize_button.setMinimumSize(110, 38)
        self.recognize_button.clicked.connect(self._start_inference)
        toolbar.addWidget(self.recognize_button)
        self.live_button = QPushButton("Live Preview")
        self.live_button.setMinimumHeight(38)
        self.live_button.clicked.connect(self._resume_live_preview)
        toolbar.addWidget(self.live_button)
        root.addLayout(toolbar)

        self.connection_label = QLabel(
            "RGB: waiting  |  Depth: waiting  |  Controller: checking"
        )
        self.connection_label.setStyleSheet(
            "padding: 7px; background: #e9edf0; border: 1px solid #c8ced3;"
        )
        root.addWidget(self.connection_label)

        image_splitter = QSplitter(Qt.Horizontal)
        self.camera_panel = ImagePanel("Inference Input")
        self.overlay_panel = ImagePanel("SAM3 Overlay")
        self.overlay_panel.image_label.setText("Overlay appears after recognition")
        image_splitter.addWidget(self.camera_panel)
        image_splitter.addWidget(self.overlay_panel)
        image_splitter.setSizes([720, 720])
        root.addWidget(image_splitter, 3)

        lower_splitter = QSplitter(Qt.Horizontal)
        decision_group = QGroupBox("Decision")
        decision_layout = QGridLayout(decision_group)
        self.decision_label = QLabel("Not evaluated")
        self.decision_label.setAlignment(Qt.AlignCenter)
        self.decision_label.setFont(QFont("Arial", 17, QFont.Bold))
        self.decision_label.setMinimumHeight(55)
        self.count_value = QLabel("-")
        self.count_value.setFont(QFont("Arial", 30, QFont.Bold))
        self.count_value.setAlignment(Qt.AlignCenter)
        self.target_value = QLabel("-")
        self.run_path_value = QLabel("-")
        self.run_path_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.reason_text = QTextEdit()
        self.reason_text.setReadOnly(True)
        decision_layout.addWidget(self.decision_label, 0, 0, 1, 2)
        decision_layout.addWidget(QLabel("Instrument count"), 1, 0)
        decision_layout.addWidget(self.count_value, 1, 1)
        decision_layout.addWidget(QLabel("Target object"), 2, 0)
        decision_layout.addWidget(self.target_value, 2, 1)
        decision_layout.addWidget(QLabel("Output path"), 3, 0)
        decision_layout.addWidget(self.run_path_value, 3, 1)
        decision_layout.addWidget(self.reason_text, 4, 0, 1, 2)
        lower_splitter.addWidget(decision_group)

        masks_group = QGroupBox("Detected Instruments")
        masks_layout = QVBoxLayout(masks_group)
        self.mask_table = QTableWidget(0, 9)
        self.mask_table.setHorizontalHeaderLabels(
            [
                "Instrument ID",
                "Type",
                "Center (px)",
                "Depth (m)",
                "Position 3D (m)",
                "BBox (px)",
                "Confidence",
                "Coordinate frame",
                "Status",
            ]
        )
        self.mask_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.mask_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.mask_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        masks_layout.addWidget(self.mask_table)
        lower_splitter.addWidget(masks_group)
        lower_splitter.setSizes([570, 850])
        root.addWidget(lower_splitter, 2)

        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f5f6f7; color: #202428; }
            QPushButton { background: #176b87; color: white; border: 0; padding: 8px 16px; }
            QPushButton:hover { background: #125b73; }
            QPushButton:disabled { background: #9aa4aa; }
            QComboBox, QTextEdit, QTableWidget { background: white; border: 1px solid #bcc4c9; }
            QGroupBox { font-weight: bold; border: 1px solid #bcc4c9; margin-top: 8px; padding-top: 8px; }
            QHeaderView::section { background: #e3e7ea; padding: 6px; border: 0; border-right: 1px solid #c8ced3; }
        """)

    def _on_camera_image(self, image):
        self.current_image = image.copy()
        if self.display_live and (
            self.inference_thread is None or not self.inference_thread.isRunning()
        ):
            self.camera_panel.set_image(image)
        self.connection_label.setText(
            self.connection_label.text().replace("RGB: waiting", "RGB: connected")
        )

    def _on_depth_received(self, metadata):
        self.depth_available = True
        depth_status = (
            f"Depth: connected ({metadata.get('encoding', '-')}, "
            f"{metadata.get('shape', '-')})"
        )
        current = self.connection_label.text()
        parts = current.split("  |  ")
        if len(parts) >= 3:
            parts[1] = depth_status
            self.connection_label.setText("  |  ".join(parts))

    def _check_controller(self):
        try:
            response = requests.post(
                self.controller_url.rstrip("/") + "/list_models", json={}, timeout=2
            )
            models = response.json().get("models", []) if response.ok else []
            status = "connected" if models else "waiting for worker"
        except Exception:
            status = "unavailable"
        sensor_status = self.connection_label.text().rsplit(
            "  |  Controller:", 1
        )[0]
        self.connection_label.setText(
            f"{sensor_status}  |  Controller: {status}"
        )

    def _start_inference(self):
        if self.current_image is None:
            QMessageBox.warning(self, "No image", "No camera frame has been received.")
            return
        display_target = self.target_combo.currentText().strip()
        if not display_target:
            QMessageBox.warning(self, "Missing target", "Enter a recognition target.")
            return
        sam3_prompt = display_target.lower()
        self.frozen_image = self.current_image.copy()
        self.display_live = False
        self.camera_panel.set_image(self.frozen_image)
        self.recognize_button.setEnabled(False)
        self.live_button.setEnabled(False)
        self.recognize_button.setText("Processing")
        self.decision_label.setText("Running inference")
        self.decision_label.setStyleSheet("background: #fff3cd; color: #5d4800;")
        self.reason_text.setPlainText(
            "Running fine-tuned SAM3 with target-specific calibration..."
        )
        self.inference_thread = InferenceThread(
            self.frozen_image,
            sam3_prompt,
            display_target,
            self.controller_url,
            self.model_name,
            self.overlay_path,
            self.output_dir,
            self.node.get_depth_snapshot(),
        )
        self.inference_thread.worker_ready.connect(
            lambda address: self.connection_label.setText(
                "RGB: connected  |  "
                f"Depth: {'connected' if self.depth_available else 'waiting'}  |  "
                f"Worker: processing ({address})"
            )
        )
        self.inference_thread.succeeded.connect(self._show_result)
        self.inference_thread.failed.connect(self._show_error)
        self.inference_thread.finished.connect(self._inference_finished)
        self.inference_thread.start()

    def _show_result(self, result, overlay, run_path):
        intervention = bool(result.get("requires_intervention", False))
        self.overlay_panel.set_image(overlay)
        self.count_value.setText(str(result.get("estimated_count", 0)))
        self.target_value.setText(str(result.get("gui_target", "-")))
        self.run_path_value.setText(run_path)
        if intervention:
            self.decision_label.setText("INTERVENTION REQUIRED")
            self.decision_label.setStyleSheet(
                "background: #f8d7da; color: #842029; border: 1px solid #e5aeb3;"
            )
        else:
            self.decision_label.setText("ACCEPT SAM3 RESULT")
            self.decision_label.setStyleSheet(
                "background: #d7eddf; color: #185b32; border: 1px solid #acd2ba;"
            )
        reasons = result.get("intervention_reasons", [])
        corrections = result.get("correction_reasons", [])
        uncertain = result.get("uncertain_regions", [])
        lines = [
            "Intervention reasons: " + (", ".join(reasons) if reasons else "none"),
            "Explanation: " + str(
                result.get("intervention_explanation", "-")
            ),
            "Automatic corrections: " + (", ".join(corrections) if corrections else "none"),
            f"Uncertain regions: {len(uncertain)}",
            f"Fork border contacts: {result.get('fork_border_contact_count', 0)}",
            f"Fork border ratio: {result.get('fork_border_contact_ratio', 0)}",
            f"Minimum score margin: {result.get('minimum_score_margin', '-')}",
            f"Shadow score threshold: {result.get('shadow_score_threshold', '-')}",
            f"Shadow candidates: {result.get('shadow_candidate_count', 0)}",
            f"Shadow IoM band: {result.get('shadow_iom_low_threshold', '-')}"
            f"–{result.get('shadow_iom_high_threshold', '-')}",
            f"SAM3 prompt: {result.get('sam3_prompt', '-')}",
            f"Calibration profile: {result.get('calibration_profile', '-')}",
            f"Score threshold: {result.get('score_threshold', '-')}",
            f"Custom geometry: {result.get('custom_geometry_applied', '-')}",
            f"IoM threshold: {result.get('iom_threshold', '-')}",
        ]
        depth_capture = result.get("depth_capture", {})
        lines.extend([
            f"Depth available: {depth_capture.get('available', False)}",
            "Instances with 3D position: "
            f"{depth_capture.get('instances_with_3d_position', 0)}"
            f"/{len(result.get('instrument_instances', []))}",
            f"Depth frame: {depth_capture.get('frame_id', '-')}",
        ])
        for region in uncertain:
            evidence = region.get("evidence", {})
            lines.append(
                f"{region.get('region_id', '-')}: "
                f"{region.get('uncertainty_type', 'unknown')} | "
                f"masks={region.get('related_mask_ids', [])} | "
                f"evidence={evidence} | "
                f"action={region.get('recommended_action', '-')}"
            )
        self.reason_text.setPlainText("\n".join(lines))
        self._fill_instances(result.get("instrument_instances", []))
        self.node.publish_result(result, overlay)

    def _fill_instances(self, instances):
        self.mask_table.setRowCount(len(instances))
        for row, instance in enumerate(instances):
            position = instance.get("position", {})
            values = [
                instance.get("instrument_id"),
                instance.get("instrument_type"),
                position.get("center_px"),
                position.get("depth_m"),
                position.get("center_camera_m"),
                position.get("bbox_px"),
                instance.get("confidence"),
                instance.get(
                    "camera_coordinate_frame", instance.get("coordinate_frame")
                ),
                instance.get("status"),
            ]
            for column, value in enumerate(values):
                if isinstance(value, float):
                    text = f"{value:.4f}"
                elif isinstance(value, list):
                    text = ", ".join(str(item) for item in value)
                else:
                    text = "-" if value is None else str(value)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if column == 6 and isinstance(value, (int, float)) and value < 0.30:
                    item.setBackground(QColor("#f8d7da"))
                self.mask_table.setItem(row, column, item)

    def _show_error(self, message):
        self.decision_label.setText("RECOGNITION FAILED")
        self.decision_label.setStyleSheet("background: #f8d7da; color: #842029;")
        self.reason_text.setPlainText(message)

    def _inference_finished(self):
        self.recognize_button.setEnabled(True)
        self.live_button.setEnabled(True)
        self.recognize_button.setText("Recognize")
        rgb_status = "connected" if self.current_image is not None else "waiting"
        depth_status = "connected" if self.depth_available else "waiting"
        self.connection_label.setText(
            f"RGB: {rgb_status}  |  Depth: {depth_status}  |  Worker: ready"
        )

    def _resume_live_preview(self):
        self.display_live = True
        if self.current_image is not None:
            self.camera_panel.set_image(self.current_image)
        self.overlay_panel.clear_image("Overlay appears after recognition")
        self.decision_label.setText("Not evaluated")
        self.decision_label.setStyleSheet("")
        self.count_value.setText("-")
        self.target_value.setText("-")
        self.run_path_value.setText("-")
        self.reason_text.clear()
        self.mask_table.setRowCount(0)

    def closeEvent(self, event):
        if self.inference_thread is not None and self.inference_thread.isRunning():
            QMessageBox.information(
                self,
                "Inference in progress",
                "Wait for the current recognition request to finish before closing.",
            )
            event.ignore()
            return
        event.accept()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="SAM3 recognition GUI")
    parser.add_argument("--controller-url", default="http://127.0.0.1:11000")
    parser.add_argument("--model-name", default="llava-v1.5-13b")
    parser.add_argument("--target-object", default="scissors")
    parser.add_argument("--overlay-path", default="/workspace/sam3_worker_candidates.jpg")
    parser.add_argument("--output-dir", default="/workspace/Outputs/sam3_recognition")
    custom = argv[1:]
    if "--ros-args" in custom:
        custom = custom[:custom.index("--ros-args")]
    return parser.parse_args(custom)


def main(args=None):
    parsed = parse_args(sys.argv)
    rclpy.init(args=args)
    app = QApplication([sys.argv[0]])
    bridge = SignalBridge()
    node = CameraNode(bridge)
    ros_thread = RosSpinThread(node)
    window = MainWindow(
        node,
        bridge,
        parsed.controller_url,
        parsed.model_name,
        parsed.overlay_path,
        parsed.output_dir,
        parsed.target_object,
    )
    ros_thread.start()
    window.show()
    exit_code = app.exec_()
    ros_thread.stop()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
