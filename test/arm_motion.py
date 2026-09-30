#!/usr/bin/env python3
"""
Arm reach controller to move the gripper toward a selected white point
seen by the D435i camera. Uses the mobile base only to rotate (yaw) so the
point stays centered; translation is performed by arm extension, while lift
adjusts height gently. Head is actively stabilized to keep the point in view.

Inputs:
- Subscribes to D435i ZMQ stream (same schema as send_d435i_images.py).
- Initializes target from a provided pixel (x, y) in original, unrotated
  D435i image coordinates. Tracks the point via template-matching.

Outputs/Control:
- Uses stretch_body APIs to command arm/lift velocities and base angular velocity.
- Keeps head pan/tilt aligned, blending the white point with the D405 ArUco marker
  so the wrist stays visible in the D435i view. Attempts close when close/aligned.

Notes:
- The GUI rotates D435i by 90° only for display. Provided (x, y) should be
  in the original camera frame.
"""

import argparse
import math
import os
import time
from typing import Optional, Tuple

import numpy as np
import zmq
import cv2

from . import yolo_networking as yn
from . import d435i_helpers_without_pyrealsense as dh
from . import aruco_detector as ad
from .white_point_d435i import _load_aruco_marker_info

try:
    import stretch_body.robot as rb
except Exception:
    rb = None


class WhitePointTracker:
    def __init__(self, init_px: int, init_py: int, template_size: int = 41, search_radius: int = 40):
        self.px = int(init_px)
        self.py = int(init_py)
        self.template_size = int(max(11, template_size | 1))
        self.search_radius = int(max(8, search_radius))
        self.template = None
        self.initialized = False

    def _extract_patch(self, gray, cx, cy, size):
        h, w = gray.shape[:2]
        half = size // 2
        x0 = max(0, cx - half); x1 = min(w, cx + half + 1)
        y0 = max(0, cy - half); y1 = min(h, cy + half + 1)
        patch = gray[y0:y1, x0:x1]
        if patch.shape[0] != size or patch.shape[1] != size:
            pad_t = size - patch.shape[0]
            pad_l = size - patch.shape[1]
            patch = cv2.copyMakeBorder(patch, 0, pad_t, 0, pad_l, cv2.BORDER_REPLICATE)
        return patch

    def initialize(self, color_image):
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY) if color_image.ndim == 3 else color_image
        self.template = self._extract_patch(gray, self.px, self.py, self.template_size)
        self.initialized = True

    def update(self, color_image) -> Tuple[int, int]:
        if not self.initialized:
            self.initialize(color_image)
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY) if color_image.ndim == 3 else color_image
        h, w = gray.shape[:2]
        sr = self.search_radius
        half_t = self.template_size // 2
        x0 = max(0, self.px - sr - half_t); x1 = min(w, self.px + sr + half_t + 1)
        y0 = max(0, self.py - sr - half_t); y1 = min(h, self.py + sr + half_t + 1)
        search = gray[y0:y1, x0:x1]
        if search.shape[0] < self.template_size or search.shape[1] < self.template_size:
            return self.px, self.py
        res = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        cx = x0 + max_loc[0] + half_t
        cy = y0 + max_loc[1] + half_t
        self.px, self.py = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
        return self.px, self.py


def robust_depth_at_pixel(depth_image, px, py, depth_scale, k=3) -> Optional[float]:
    h, w = depth_image.shape[:2]
    x0 = max(0, px - k); x1 = min(w, px + k + 1)
    y0 = max(0, py - k); y1 = min(h, py + k + 1)
    patch = depth_image[y0:y1, x0:x1]
    if patch is None or patch.size == 0:
        return None
    vals = patch.reshape(-1)
    if np.issubdtype(vals.dtype, np.floating):
        vals_m = vals[np.isfinite(vals)]
    else:
        vals_m = vals.astype(np.float32) * float(depth_scale)
    vals_m = vals_m[vals_m > 0]
    if vals_m.size == 0:
        return None
    return float(np.median(vals_m))


def main(x: int, y: int, use_remote: bool,
         stop_z_m: float = 0.38,
         grasp_z_m: float = 0.28,
         max_time_s: float = 600.0,
         k_ext: float = 0.8,
         k_lift: float = 0.6,
         k_pan: float = 0.8,
         k_tilt: float = 0.8,
         invert_yaw: bool = True,
         head_tilt_bias: float = 0.0,
         aruco_weight: float = 0.85,
         aruco_update_n: int = 3,
         head_aruco_only: bool = False):
    if rb is None:
        raise RuntimeError("stretch_body not available; cannot control robot")

    robot = rb.Robot()
    robot.startup()

    # Start pose: arm/lift nominal; gripper open; wrist yaw forward
    try:
        try:
            if hasattr(robot, 'end_of_arm'):
                jy = robot.end_of_arm.get_joint('wrist_yaw')
                if jy is not None:
                    jy.move_to(math.pi / 2.0)
        except Exception:
            pass
        try:
            if hasattr(robot, 'arm'):
                robot.arm.move_to(0.05)
        except Exception:
            pass
        try:
            if hasattr(robot, 'lift'):
                robot.lift.move_to(0.70)
        except Exception:
            pass
        try:
            if hasattr(robot, 'end_of_arm'):
                robot.end_of_arm.get_joint('stretch_gripper').move_to(10.46)
        except Exception:
            pass
        robot.push_command(); robot.wait_command()
    except Exception:
        pass

    # Subscribe to D435i stream
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b'')
    sub.setsockopt(zmq.SNDHWM, 1)
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.setsockopt(zmq.CONFLATE, 1)
    addr = ('tcp://*:' if use_remote else 'tcp://127.0.0.1:') + str(yn.d435i_port)
    if use_remote:
        addr = 'tcp://' + yn.robot_ip + ':' + str(yn.d435i_port)
    sub.connect(addr)

    tracker = None
    t0 = time.time()
    done_hold = 0
    close_sent = False

    # Desired location for the tracked point in the rotated D435i view (center by default)
    rot_target_x_ratio = float(os.getenv('ARM_MOTION_ROT_TARGET_X', '0.50'))
    rot_target_y_ratio = float(os.getenv('ARM_MOTION_ROT_TARGET_Y', '0.50'))
    rot_target_x_ratio = float(np.clip(rot_target_x_ratio, 0.0, 1.0))
    rot_target_y_ratio = float(np.clip(rot_target_y_ratio, 0.0, 1.0))
    aruco_weight = float(np.clip(aruco_weight, 0.0, 1.0))
    marker_info = _load_aruco_marker_info()
    aruco_detector = ad.ArucoDetector(marker_info=marker_info, show_debug_images=False,
                                      use_apriltag_refinement=False, brighten_images=False)
    aruco_every_n = max(1, int(os.getenv('ARM_MOTION_ARUCO_EVERY_N', str(aruco_update_n))))
    aruco_timeout = max(aruco_every_n * 3, 12)
    d405_px = None
    d405_py = None
    d405_pos = None  # 3D position of the D405 back ArUco in D435i camera frame
    d405_frame_idx = -1
    frame_idx = 0
    # Track marker loss to trigger a gentle pan sweep for acquisition
    missing_d405_count = 0
    sweep_dir = 1.0  # + right, - left (sign depends on head frame)

    try:
        while (time.time() - t0) < max_time_s:
            frame_idx += 1
            try:
                out = sub.recv_pyobj(flags=0)
            except zmq.error.ZMQError:
                break
            color = out.get('color_image'); depth = out.get('depth_image')
            info = out.get('depth_camera_info', out.get('color_camera_info'))
            dscale = out.get('depth_scale')
            if color is None or depth is None or info is None or dscale is None:
                continue
            camera_matrix = info.get('camera_matrix') if isinstance(info, dict) else None
            if camera_matrix is None:
                continue
            fx = float(camera_matrix[0, 0])
            fy = float(camera_matrix[1, 1])
            if (frame_idx % aruco_every_n) == 0:
                try:
                    aruco_detector.update(color, info)
                    markers = aruco_detector.get_detected_marker_dict()
                    d405_entry = None
                    if markers:
                        d405_entry = markers.get(135)
                        if d405_entry is None:
                            for _mid, mdata in markers.items():
                                info_m = mdata.get('info', {}) if isinstance(mdata, dict) else {}
                                if info_m.get('name') == 'd405_back':
                                    d405_entry = mdata
                                    break
                    if d405_entry is not None:
                        pos = d405_entry.get('pos')
                        if pos is not None:
                            d405_pos = np.array(pos, dtype=np.float32)
                            uv = dh.pixel_from_3d(d405_pos, info)
                            d405_px = float(uv[0])
                            d405_py = float(uv[1])
                            d405_frame_idx = frame_idx
                except Exception:
                    pass

            h, w = depth.shape[:2]
            # Initialize tracker on first valid frame
            if tracker is None:
                px = int(round(x)); py = int(round(y))
                if px < 0 or px >= w or py < 0 or py >= h:
                    px = int(np.clip(y, 0, w - 1))
                    py = int(np.clip((h - 1) - x, 0, h - 1))
                tracker = WhitePointTracker(px, py, template_size=41, search_radius=50)
                tracker.initialize(color)

            # Update target pixel
            px, py = tracker.update(color)

            # Depth and 3D point
            z_m = robust_depth_at_pixel(depth, px, py, dscale, k=4)
            if z_m is None or not (0.05 < z_m < 10.0):
                # No valid depth: stop motions but keep tracking
                try:
                    robot.arm.set_velocity(0.0)
                    robot.lift.set_velocity(0.0)
                    robot.base.set_velocity(0.0, 0.0)
                    if hasattr(robot, 'head'):
                        robot.head.set_velocity('head_tilt', 0.0)
                        robot.head.set_velocity('head_pan', 0.0)
                    robot.push_command()
                except Exception:
                    pass
                time.sleep(0.05)
                continue

            # GUI/display aligned errors
            max_w = max(float(w) - 1.0, 0.0)
            max_h = max(float(h) - 1.0, 0.0)
            valid_d405 = False
            d405_px_clamped = None
            d405_py_clamped = None
            if (d405_px is not None) and (d405_py is not None):
                if (aruco_timeout <= 0) or ((frame_idx - d405_frame_idx) <= aruco_timeout):
                    d405_px_clamped = float(np.clip(d405_px, 0.0, max_w))
                    d405_py_clamped = float(np.clip(d405_py, 0.0, max_h))
                    valid_d405 = True
            # Update missing counter for acquisition behavior
            if valid_d405:
                missing_d405_count = 0
            else:
                missing_d405_count = min(missing_d405_count + 1, 10**9)
            # Desired display target (legacy center reference, used only as fallback)
            desired_px_raw = rot_target_y_ratio * max_w
            desired_py_raw = (1.0 - rot_target_x_ratio) * max_h

            # Compute relative offsets of the white point to the D405 ArUco marker in D435i image
            # When valid_d405 is False, fall back to display-center reference
            if valid_d405 and (d405_px_clamped is not None) and (d405_py_clamped is not None):
                dx_pix = float(px) - float(d405_px_clamped)
                dy_pix = float(py) - float(d405_py_clamped)
            else:
                dx_pix = float(px) - float(desired_px_raw)
                dy_pix = float(py) - float(desired_py_raw)

            # Normalized errors (approx angles) for control
            dx_norm = dx_pix / max(1e-6, float(fx))
            dy_norm = dy_pix / max(1e-6, float(fy))

            # Base yaw: correct horizontal offset of white point RELATIVE TO the D405 ArUco marker.
            # White left of gripper -> rotate to bring it toward center; no base translation allowed.
            # Use pixel-to-angle approximation via fx.
            k_yaw = 0.9
            w_cmd_raw = float(np.clip(-k_yaw * dx_norm, -0.5, 0.5))
            w_cmd = (-w_cmd_raw) if invert_yaw else w_cmd_raw
            w_cmd = float(np.clip(w_cmd, -0.25, 0.25))

            # Arm extension toward target based on depth error
            # Arm extension toward target based on DEPTH difference from the D405 marker (white vs gripper)
            if valid_d405 and (d405_pos is not None) and (len(d405_pos) >= 3):
                z_gripper = float(d405_pos[2])
                dist_err = float(z_m) - z_gripper
            else:
                dist_err = float(z_m) - float(stop_z_m)
            v_arm = float(np.clip(k_ext * dist_err, -0.10, 0.18))  # m/s extension (+) or retract (-)

            # Lift adjustment: nudge to keep point vertically centered in GUI
            # White above gripper -> lift up; below -> lift down (use dy_norm)
            v_lift = float(np.clip(-k_lift * dy_norm, -0.06, 0.06))

            # Head tracking
            cur_pan = 0.0
            cur_tilt = 0.0
            try:
                robot.pull_status()
                if hasattr(robot, 'head'):
                    j_pan = robot.head.get_joint('head_pan')
                    if j_pan is not None and j_pan.status is not None:
                        cur_pan = float(j_pan.status.get('pos', cur_pan))
                    j_tilt = robot.head.get_joint('head_tilt')
                    if j_tilt is not None and j_tilt.status is not None:
                        cur_tilt = float(j_tilt.status.get('pos', cur_tilt))
            except Exception:
                pass

            # Head control: directly track D405 ArUco pixel offsets firmly.
            # - When marker is present: command strong pan/tilt toward center (snap to target).
            # - When absent: continue sweeping pan to reacquire; hold current tilt.
            if (fx > 1e-6) and (fy > 1e-6) and valid_d405:
                # Pixel offsets (marker relative to display center)
                pan_off = (d405_px_clamped - desired_px_raw) / float(fx)
                tilt_off = (d405_py_clamped - desired_py_raw) / float(fy)
                # Desired angles: remove offset; apply optional tilt bias
                desired_pan = cur_pan - pan_off
                desired_tilt = cur_tilt + tilt_off + float(head_tilt_bias)
                # Error to desired
                head_pan_err = desired_pan - cur_pan
                head_tilt_err = desired_tilt - cur_tilt
                # Aggressive gains and wider velocity limits for fast alignment
                k_pan_eff = max(0.8, float(k_pan)) * 1.4
                k_tilt_eff = max(0.8, float(k_tilt)) * 1.6
                v_pan = float(np.clip(k_pan_eff * head_pan_err, -1.2, 1.2))
                v_tilt = float(np.clip(k_tilt_eff * head_tilt_err, -1.0, 1.0))
            else:
                # Marker not visible: sweep pan to search; hold tilt steady.
                desired_pan = cur_pan
                desired_tilt = cur_tilt
                head_pan_err = 0.0
                head_tilt_err = 0.0
                v_tilt = 0.0
                sweep_speed = 0.22
                try:
                    if abs(cur_pan) > 1.25:
                        sweep_dir = -sweep_dir
                except Exception:
                    pass
                v_pan = sweep_speed * sweep_dir

            # Stop conditions and grasp
            # Centering is relative to the gripper ArUco: small relative pixel offset
            centered = (abs(dx_norm) < 0.05) and (abs(dy_norm) < 0.05)
            # Close when at or inside grasp depth threshold (or white nearly at gripper range)
            close_enough = (float(z_m) <= float(grasp_z_m)) or (valid_d405 and abs(dist_err) < 0.03)

            if centered and close_enough:
                done_hold += 1
                # Stop motion and attempt close once
                try:
                    robot.base.set_velocity(0.0, 0.0)
                    robot.arm.set_velocity(0.0)
                    robot.lift.set_velocity(0.0)
                    if hasattr(robot, 'head'):
                        robot.head.set_velocity('head_tilt', 0.0)
                        robot.head.set_velocity('head_pan', 0.0)
                    # Close gripper once
                    if (not close_sent) and hasattr(robot, 'end_of_arm'):
                        try:
                            g = robot.end_of_arm.get_joint('stretch_gripper')
                            if g is not None:
                                g.move_to(0.0)
                            close_sent = True
                        except Exception:
                            pass
                    robot.push_command()
                except Exception:
                    pass
                if done_hold >= 12:
                    break
            else:
                done_hold = 0
                try:
                    # Command base yaw, arm, lift, and head stabilization
                    robot.base.set_velocity(0.0, w_cmd)
                    robot.arm.set_velocity(v_arm)
                    robot.lift.set_velocity(v_lift)
                    if hasattr(robot, 'head'):
                        robot.head.set_velocity('head_tilt', v_tilt)
                        robot.head.set_velocity('head_pan', v_pan)
                    # Keep wrist yaw facing forward
                    try:
                        if hasattr(robot, 'end_of_arm'):
                            jy = robot.end_of_arm.get_joint('wrist_yaw')
                            if jy is not None:
                                jy.move_to(math.pi / 2.0)
                    except Exception:
                        pass
                    robot.push_command()
                except Exception:
                    pass

            time.sleep(0.06)

    finally:
        try:
            robot.base.set_velocity(0.0, 0.0)
            robot.arm.set_velocity(0.0)
            robot.lift.set_velocity(0.0)
            if hasattr(robot, 'head'):
                try:
                    robot.head.set_velocity('head_tilt', 0.0)
                    robot.head.set_velocity('head_pan', 0.0)
                except Exception:
                    pass
            robot.push_command()
        except Exception:
            pass
        try:
            robot.stop()
        except Exception:
            pass


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Reach with arm to a white point using D435i')
    p.add_argument('-x', type=int, required=True, help='Pixel x in original D435i image coordinates')
    p.add_argument('-y', type=int, required=True, help='Pixel y in original D435i image coordinates')
    p.add_argument('-r', '--remote', action='store_true', help='Subscribe to remote robot D435i stream')
    p.add_argument('--stop-z-m', type=float, default=0.38, help='Depth to stop extension before grasp (meters)')
    p.add_argument('--grasp-z-m', type=float, default=0.28, help='Depth at which to close gripper (meters)')
    p.add_argument('--max-time-s', type=float, default=600.0, help='Safety timeout (seconds)')
    p.add_argument('--k-ext', type=float, default=0.8, help='Arm extension gain (m/s per meter)')
    p.add_argument('--k-lift', type=float, default=0.6, help='Lift gain (m/s per normalized pixel)')
    p.add_argument('--k-pan', type=float, default=0.8, help='Head pan gain (vel per normalized pixel)')
    p.add_argument('--k-tilt', type=float, default=0.8, help='Head tilt gain (vel per normalized pixel)')
    p.add_argument('--aruco-weight', type=float, default=0.65,
                   help='Blend weight (0-1) for D405 ArUco alignment in head control.')
    p.add_argument('--aruco-update-n', type=int, default=3,
                   help='Run ArUco detection every N frames (>=1).')
    p.add_argument('--invert-yaw', action='store_true', help='Invert base yaw direction (default on)')
    p.add_argument('--head-aruco-only', action='store_true', help='Use ArUco-only head tracking (ignore white point).')
    p.set_defaults(invert_yaw=True)
    args = p.parse_args()

    main(args.x, args.y, args.remote,
         stop_z_m=args.stop_z_m, grasp_z_m=args.grasp_z_m,
         max_time_s=args.max_time_s,
         k_ext=args.k_ext, k_lift=args.k_lift,
         k_pan=args.k_pan, k_tilt=args.k_tilt,
         invert_yaw=args.invert_yaw,
         aruco_weight=args.aruco_weight,
         aruco_update_n=args.aruco_update_n,
         head_aruco_only=bool(args.head_aruco_only))
