#!/usr/bin/env python3
"""
Base motion controller to navigate the robot towards a selected white point
seen by the D435i camera. Drives the mobile base and the head (D435i mount)
to keep the point centered and stops at a configurable distance.

Inputs:
- Subscribes to D435i ZMQ stream (same schema as send_d435i_images.py).
- Initializes target from a provided pixel (x, y) in original, unrotated
  D435i image coordinates. Tracks the point over time using template-matching.

Outputs/Control:
- Uses stretch_body APIs to command base linear/angular velocity and head pan/tilt.
  Only base and head are controlled; arm/gripper are untouched.

Notes:
- The GUI rotates D435i by 90° only for display. The provided (x, y) should be
  in the original camera frame. This module works in that original frame.
"""

import argparse
import math
import time
from typing import Optional, Tuple

import numpy as np
import zmq
import cv2

from . import yolo_networking as yn
from . import d435i_helpers_without_pyrealsense as dh
from .gui_rotation import (
    compute_display_offsets,
    normalized_display_position,
    resolve_rotation_degrees,
)

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


def main(x: int, y: int, use_remote: bool, stop_dist_m: float, max_time_s: float,
         k_lin: float, k_ang: float, k_tilt: float, k_pan: float,
         tilt_only: bool = False,
         tilt_down_negative: bool = True,
         invert_yaw: bool = True):
    if rb is None:
        raise RuntimeError("stretch_body not available; cannot control base/head")

    robot = rb.Robot()
    robot.startup()

    # Move to a navigation start pose: head forward, arm/lift/gripper to neutral
    try:
        if hasattr(robot, 'head'):
            # Start with pan at 0.0 to face forward; keep current tilt (no reset) to avoid snapping back
            robot.head.move_to('head_pan', 0.0)
        # Align gripper forward if wrist yaw joint exists: +pi/2 faces forward per request
        try:
            if hasattr(robot, 'end_of_arm'):
                j = robot.end_of_arm.get_joint('wrist_yaw')
                if j is not None:
                    j.move_to(math.pi / 2.0)
        except Exception:
            pass
        try:
            if hasattr(robot, 'arm'):
                robot.arm.move_to(0.01)
        except Exception:
            pass
        try:
            if hasattr(robot, 'lift'):
                robot.lift.move_to(0.7)
        except Exception:
            pass
        try:
            if hasattr(robot, 'end_of_arm'):
                robot.end_of_arm.get_joint('stretch_gripper').move_to(10.46)
        except Exception:
            pass
        robot.push_command(); robot.wait_command()

        # Engage head joints so the D435i mount resists manual rotation even in tilt-only mode.
        # Sending a zero velocity command energizes the motor without changing behavior.
        try:
            if hasattr(robot, 'head'):
                # Ensure tilt is active (it will be commanded in the loop), and energize pan at zero
                try:
                    robot.head.set_velocity('head_tilt', 0.0)
                except Exception:
                    try:
                        robot.head.motors['head_tilt'].set_command('vel', 0.0)
                    except Exception:
                        pass
                try:
                    robot.head.set_velocity('head_pan', 0.0)
                except Exception:
                    try:
                        robot.head.motors['head_pan'].set_command('vel', 0.0)
                    except Exception:
                        pass
                robot.push_command()
        except Exception:
            pass
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
    # When use_remote=True this process is intended to run on the robot; for GUI-hosted runs,
    # we still subscribe to localhost mirror created by DirectCameraThread.
    if use_remote:
        addr = 'tcp://' + yn.robot_ip + ':' + str(yn.d435i_port)
    sub.connect(addr)

    tracker = None
    display_rotation_deg = resolve_rotation_degrees()
    t0 = time.time()
    done_hold = 0

    try:
        while (time.time() - t0) < max_time_s:
            try:
                out = sub.recv_pyobj(flags=0)
            except zmq.error.ZMQError:
                break
            color = out.get('color_image'); depth = out.get('depth_image')
            info = out.get('depth_camera_info', out.get('color_camera_info'))
            dscale = out.get('depth_scale')
            if color is None or depth is None or info is None or dscale is None:
                continue

            h, w = depth.shape[:2]
            # Initialize tracker on first valid frame
            if tracker is None:
                px = int(np.clip(x, 0, w - 1)); py = int(np.clip(y, 0, h - 1))
                tracker = WhitePointTracker(px, py, template_size=41, search_radius=50)
                tracker.initialize(color)

            # Update target pixel
            px, py = tracker.update(color)
            horizontal_offset, vertical_offset, _, _ = compute_display_offsets(
                px, py, w, h, rotation_deg=display_rotation_deg
            )
            disp_u, disp_v = normalized_display_position(
                px, py, w, h, rotation_deg=display_rotation_deg
            )

            # Depth and 3D point
            z_m = robust_depth_at_pixel(depth, px, py, dscale, k=4)
            if z_m is None or not (0.05 < z_m < 10.0):
                # If no valid depth, stop base but keep searching
                robot.base.set_velocity(0.0, 0.0)
                robot.push_command(); time.sleep(0.02)
                continue
            xyz = dh.pixel_to_3d(np.array([px, py], dtype=np.float32), float(z_m), info)
            x_m, y_m, z_m = float(xyz[0]), float(xyz[1]), float(xyz[2])

            # Control laws
            # Angular error for GUI-aligned left/right: camera appears rotated 90° in the GUI,
            # so use the camera Y-axis to represent GUI horizontal. Right-of-center on the GUI
            # corresponds to negative y_m. Use -y_m so that right-of-center -> positive yaw_err.
            # Then with invert_yaw=True, w_cmd becomes negative (clockwise) on GUI-right.
            yaw_err = math.atan2(-y_m, z_m)
            yaw_term = float(np.clip(k_ang * yaw_err, -0.5, 0.5))
            w_cmd_3d = (-yaw_term) if invert_yaw else yaw_term
            # Linear error: move until distance ~ stop_dist_m
            dist_err = z_m - float(stop_dist_m)
            v_cmd = float(np.clip(k_lin * dist_err, -0.30, 0.35))  # m/s forward positive

            # Tilt-first gating based on GUI vertical error - stop driving forward until
            # the white point is near the upper 2/3 of the display and within the deadband.
            if (disp_v > (2.0 / 3.0)) or (abs(vertical_offset) > 0.08):
                # During tilt-first alignment, hold the base translation still so head can align vertically.
                v_cmd = 0.0

            # Head tilt/pan to keep point near image center
            tilt_err = vertical_offset
            pan_err = horizontal_offset

            # Determine sign convention for head tilt: if tilting downward requires
            # negative velocity (default on Stretch), set tilt_down_negative=True.
            # Positive tilt_err means the point is lower in the GUI view -> look down.
            tilt_gain = (-k_tilt) if tilt_down_negative else (k_tilt)
            tilt_vel = float(np.clip(tilt_gain * tilt_err, -0.6, 0.6))
            # Respect tilt-only mode by zeroing pan gain/command. For pan, use a
            # negative gain to rotate the head towards the error (standard image coords).
            pan_vel = 0.0 if tilt_only else float(np.clip(-k_pan * pan_err, -0.8, 0.8))

            # Gentle base yaw to re-center the point horizontally in the GUI.
            # The GUI rotates the original image by 90° clockwise, and for drawing we use:
            #   x_display = (h - 1) - py,  y_display = px
            # So compute the GUI horizontal error directly in display space to avoid sign confusion.
            err_disp_x = (disp_u - 0.5)
            k_gui = 0.6  # small gain for display-based yaw
            # Positive GUI-right error -> rotate base slightly right (clockwise -> negative w)
            w_gui = float(np.clip(-k_gui * err_disp_x, -0.15, 0.15))

            # Combine 3D-based yaw alignment with GUI-horizontal centering.
            w_cmd = float(np.clip(w_cmd_3d + w_gui, -0.25, 0.25))

            # Stop conditions: close and aligned
            close = (abs(dist_err) < max(0.05, 0.2 * stop_dist_m))  # within 5–10 cm or 20% of stop
            centered = (abs(yaw_err) < math.radians(4.0))
            if close and centered:
                done_hold += 1
                robot.base.set_velocity(0.0, 0.0)
                if hasattr(robot, 'head'):
                    robot.head.set_velocity('head_tilt', 0.0)
                    if not tilt_only:
                        try:
                            robot.head.motors['head_pan'].set_command('vel', 0.0)
                        except Exception:
                            pass
                robot.push_command()
                if done_hold >= 10:  # hold for ~0.6s (loop sleeps ~0.06 s)
                    break
            else:
                done_hold = 0
                # Command base and head
                robot.base.set_velocity(v_cmd, w_cmd)
                if hasattr(robot, 'head'):
                    # Drive head joints in velocity mode; prefer set_velocity with motors fallback
                    try:
                        robot.head.set_velocity('head_tilt', tilt_vel)
                    except Exception:
                        try:
                            robot.head.motors['head_tilt'].set_command('vel', tilt_vel)
                        except Exception:
                            pass
                    # Keep head_pan energized even in tilt-only mode by commanding zero velocity
                    if not tilt_only:
                        try:
                            robot.head.set_velocity('head_pan', pan_vel)
                        except Exception:
                            try:
                                robot.head.motors['head_pan'].set_command('vel', pan_vel)
                            except Exception:
                                pass
                    else:
                        try:
                            robot.head.set_velocity('head_pan', 0.0)
                        except Exception:
                            try:
                                robot.head.motors['head_pan'].set_command('vel', 0.0)
                            except Exception:
                                pass
                # Maintain wrist yaw facing forward while moving
                try:
                    if hasattr(robot, 'end_of_arm'):
                        j = robot.end_of_arm.get_joint('wrist_yaw')
                        if j is not None:
                            j.move_to(math.pi / 2.0)
                except Exception:
                    pass
                robot.push_command()

            time.sleep(0.06)

    finally:
        try:
            robot.base.set_velocity(0.0, 0.0)
            if hasattr(robot, 'head'):
                try:
                    robot.head.set_velocity('head_tilt', 0.0)
                except Exception:
                    try:
                        robot.head.motors['head_tilt'].set_command('vel', 0.0)
                    except Exception:
                        pass
                if not tilt_only:
                    try:
                        robot.head.set_velocity('head_pan', 0.0)
                    except Exception:
                        try:
                            robot.head.motors['head_pan'].set_command('vel', 0.0)
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
    p = argparse.ArgumentParser(description='Navigate base to a white point using D435i')
    p.add_argument('-x', type=int, required=True, help='Pixel x in original D435i image coordinates')
    p.add_argument('-y', type=int, required=True, help='Pixel y in original D435i image coordinates')
    p.add_argument('-r', '--remote', action='store_true', help='Subscribe to remote robot D435i stream')
    p.add_argument('--stop-dist-m', type=float, default=0.5, help='Target stopping distance (meters)')
    p.add_argument('--max-time-s', type=float, default=600.0, help='Safety timeout (seconds)')
    p.add_argument('--k-lin', type=float, default=0.4, help='Linear gain (m/s per meter)')
    p.add_argument('--k-ang', type=float, default=1.2, help='Angular gain (rad/s per rad)')
    p.add_argument('--k-tilt', type=float, default=0.8, help='Head tilt gain (vel per normalized pixel)')
    p.add_argument('--k-pan', type=float, default=0.8, help='Head pan gain (vel per normalized pixel)')
    p.add_argument('--tilt-only', action='store_true', help='Do not command head pan; tilt only')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--tilt-down-negative', dest='tilt_down_negative', action='store_true', help='Downward tilt is negative velocity (default)')
    g.add_argument('--tilt-down-positive', dest='tilt_down_negative', action='store_false', help='Downward tilt is positive velocity')
    p.set_defaults(tilt_down_negative=True)
    p.add_argument('--invert-yaw', action='store_true', help='Invert base yaw direction (default on)')
    p.set_defaults(invert_yaw=True)
    args = p.parse_args()

    main(args.x, args.y, args.remote, args.stop_dist_m, args.max_time_s,
         args.k_lin, args.k_ang, args.k_tilt, args.k_pan,
         tilt_only=args.tilt_only,
         tilt_down_negative=args.tilt_down_negative,
         invert_yaw=args.invert_yaw)
