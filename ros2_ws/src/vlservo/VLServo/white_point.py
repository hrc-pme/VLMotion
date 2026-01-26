import os
import argparse
import time
from copy import deepcopy

import numpy as np
import cv2
import zmq

from . import yolo_networking as yn
from . import loop_timer as lt
from . import d405_helpers_without_pyrealsense as d405_dh
from . import d435i_helpers_without_pyrealsense as d435_dh
from . import aruco_detector as ad
from . import aruco_to_fingertips as af
from .white_point_tracker import WhitePointTracker
from .gui_rotation import resolve_rotation_degrees
import yaml
from yaml.loader import SafeLoader
from typing import Dict


CAMERA_SOURCES = {
    'd405': {'port': yn.d405_port, 'helpers': d405_dh},
    'd435': {'port': yn.d435i_port, 'helpers': d435_dh},
}


def _load_aruco_marker_info() -> Dict:
    """Load ArUco marker info from a robust set of locations.

    Search order:
    1) Package directory (alongside this file)
    2) Installed share config via ament (share/vlservo/config)
    3) Current working directory
    Returns empty dict on failure.
    """
    candidates = []
    try:
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(pkg_dir, 'aruco_marker_info.yaml'))
    except Exception:
        pass
    # ament share path
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory('vlservo')
        candidates.append(os.path.join(share_dir, 'config', 'aruco_marker_info.yaml'))
    except Exception:
        pass
    # CWD fallback
    candidates.append('aruco_marker_info.yaml')

    for path in candidates:
        try:
            if os.path.isfile(path):
                with open(path, 'r') as f:
                    return yaml.load(f, Loader=SafeLoader) or {}
        except Exception:
            continue
    return {}


def robust_depth_at_pixel(depth_image, px, py, depth_scale, k=3):
    h, w = depth_image.shape[:2]
    x0 = max(0, px - k); x1 = min(w, px + k + 1)
    y0 = max(0, py - k); y1 = min(h, py + k + 1)
    patch = depth_image[y0:y1, x0:x1]
    if patch is None or patch.size == 0:
        return None
    vals = patch.reshape(-1)
    # Convert to float meters
    if np.issubdtype(vals.dtype, np.floating):
        vals = vals[np.isfinite(vals)]
        vals_m = vals
    else:
        vals = vals.astype(np.float32)
        vals_m = vals * float(depth_scale)
    vals_m = vals_m[vals_m > 0]
    if vals_m.size == 0:
        return None
    return float(np.median(vals_m))




def main(use_remote_computer: bool, x: int, y: int, push_radius_m: float,
         template_size: int = 41, search_radius: int = 40, extra_push_m: float = 0.02,
         camera_source: str = 'd405'):
    # Publisher for task-relevant features (same port and schema as YOLO module)
    yolo_context = zmq.Context()
    yolo_socket = yolo_context.socket(zmq.PUB)
    if use_remote_computer:
        yolo_address = 'tcp://*:' + str(yn.yolo_port)
    else:
        yolo_address = 'tcp://' + '127.0.0.1' + ':' + str(yn.yolo_port)
    yolo_socket.setsockopt(zmq.SNDHWM, 1)
    yolo_socket.setsockopt(zmq.RCVHWM, 1)
    yolo_socket.bind(yolo_address)

    # Subscriber to camera image stream
    source_key = camera_source.lower()
    if source_key not in CAMERA_SOURCES:
        raise ValueError(f"Unknown camera source '{camera_source}'. Choose from {list(CAMERA_SOURCES)}")
    cam_cfg = CAMERA_SOURCES[source_key]
    cam_helpers = cam_cfg['helpers']

    camera_context = zmq.Context()
    camera_socket = camera_context.socket(zmq.SUB)
    camera_socket.setsockopt(zmq.SUBSCRIBE, b'')
    camera_socket.setsockopt(zmq.SNDHWM, 1)
    camera_socket.setsockopt(zmq.RCVHWM, 1)
    camera_socket.setsockopt(zmq.CONFLATE, 1)
    port = cam_cfg['port']
    if use_remote_computer:
        camera_address = 'tcp://' + yn.robot_ip + ':' + str(port)
    else:
        camera_address = 'tcp://' + '127.0.0.1' + ':' + str(port)
    camera_socket.connect(camera_address)

    # ArUco fingertip estimation (for fingertips in send_dict)
    marker_info = _load_aruco_marker_info()
    aruco_detector = ad.ArucoDetector(marker_info=marker_info, show_debug_images=False,
                                      use_apriltag_refinement=False, brighten_images=False)
    aruco_to_fingertips = af.ArucoToFingertips(default_height_above_mounting_surface=af.suctioncup_height['cup_top'])

    loop_timer = lt.LoopTimer()

    camera_info = None
    depth_scale = None
    tracker = WhitePointTracker(x, y, template_size=template_size, search_radius=search_radius)
    display_rotation_deg = resolve_rotation_degrees()

    try:
        while True:
            loop_timer.start_of_iteration()

            camera_output = camera_socket.recv_pyobj()
            color_image = camera_output.get('color_image')
            depth_image = camera_output.get('depth_image')
            depth_camera_info = camera_output.get('depth_camera_info')
            depth_scale = camera_output.get('depth_scale')
            if (color_image is None) or (depth_image is None) or (depth_camera_info is None) or (depth_scale is None):
                continue
            camera_info = depth_camera_info

            # Fingertips estimation
            try:
                aruco_detector.update(color_image, camera_info)
                markers = aruco_detector.get_detected_marker_dict()
                fingertips = aruco_to_fingertips.get_fingertips(markers)
            except Exception:
                fingertips = {}

            # Track pixel location to keep dot attached to the object
            h, w = depth_image.shape[:2]
            color_h, color_w = color_image.shape[:2]
            px_raw, py_raw, visible = tracker.update(color_image)
            horizontal_offset = None
            vertical_offset = None
            if visible:
                px = int(np.clip(px_raw, 0, color_w - 1))
                py = int(np.clip(py_raw, 0, color_h - 1))
                try:
                    horizontal_offset, vertical_offset, _, _ = tracker.get_display_offsets(
                        color_w, color_h, rotation_deg=display_rotation_deg
                    )
                except Exception:
                    horizontal_offset, vertical_offset = None, None
            else:
                px, py = None, None

            yolo_list = []
            if visible and px is not None and py is not None:
                depth_px = int(np.clip(px, 0, w - 1))
                depth_py = int(np.clip(py, 0, h - 1))
                z_m = robust_depth_at_pixel(depth_image, depth_px, depth_py, depth_scale, k=3)
                if (z_m is not None) and (z_m > 0):
                    center_surf = cam_helpers.pixel_to_3d(
                        np.array([depth_px, depth_py], dtype=np.float32), z_m, camera_info
                    )
                    grasp_center_xyz = center_surf
                    if push_radius_m and push_radius_m > 0:
                        # push along viewing ray to approximate object center
                        ray = center_surf / (np.linalg.norm(center_surf) + 1e-9)
                        grasp_center_xyz = center_surf + (push_radius_m * ray)
                    # Always add a small extra forward push if configured (default 2 cm)
                    if extra_push_m and extra_push_m > 0:
                        ray = center_surf / (np.linalg.norm(center_surf) + 1e-9)
                        grasp_center_xyz = grasp_center_xyz + (extra_push_m * ray)

                    det = {
                        'grasp_center_xyz': grasp_center_xyz,
                        'width_m': float(push_radius_m * 2.0) if push_radius_m and push_radius_m > 0 else 0.0,
                    }
                    yolo_list.append(det)

            send_dict = {
                'fingertips': fingertips,
                'yolo': yolo_list,
                'white_point_visible': bool(visible),
                'white_point_px': px if visible else None,
                'white_point_py': py if visible else None,
                'white_point_horizontal_offset': horizontal_offset,
                'white_point_vertical_offset': vertical_offset,
                'white_point_rotation_deg': display_rotation_deg,
            }

            yolo_socket.send_pyobj(send_dict)

            cv2.waitKey(1)
            loop_timer.end_of_iteration()
            loop_timer.pretty_print(minimum=True)

    finally:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='White Point to 3D Grasp Publisher',
        description='Converts a selected white-point pixel + depth to a 3D grasp_center_xyz and publishes on YOLO ZMQ channel.'
    )
    parser.add_argument('-r', '--remote', action='store_true', help='Run on the robot (bind publisher, subscribe to onboard camera stream).')
    parser.add_argument('-x', type=int, required=True, help='Pixel x (columns) in the camera color/depth image space.')
    parser.add_argument('-y', type=int, required=True, help='Pixel y (rows) in the camera color/depth image space.')
    parser.add_argument('--radius-m', type=float, default=0.0, help='Optional push radius (meters) to move from surface toward object center.')
    parser.add_argument('--template-size', type=int, default=41, help='Template patch size (odd).')
    parser.add_argument('--search-radius', type=int, default=40, help='Search radius in pixels around last position.')
    parser.add_argument('--extra-push-m', type=float, default=0.02, help='Small forward offset along viewing ray (meters). Default 0.02.')
    parser.add_argument('--camera-source', choices=sorted(CAMERA_SOURCES.keys()), default='d405',
                        help='Select which camera stream to subscribe to (default d405).')
    args = parser.parse_args()

    main(use_remote_computer=args.remote,
         x=args.x,
         y=args.y,
         push_radius_m=args.radius_m,
         template_size=args.template_size,
         search_radius=args.search_radius,
         extra_push_m=args.extra_push_m,
         camera_source=args.camera_source)
