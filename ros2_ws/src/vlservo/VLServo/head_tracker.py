#!/usr/bin/env python3
"""
Head pan/tilt tracker for D405 ArUco in D435i image.

Goals
- Keep the gripper-mounted D405 ArUco centered in the D435i view at all times.
- If the marker is lost, sweep head pan left/right until it is found; hold tilt steady.

I/O
- SUB: D435i ZMQ stream (same schema as send_d435i_images.py)
- Control: stretch_body.head set_velocity('head_pan'/'head_tilt', v)

Notes
- For LLM grasping, the integrated controller in arm_motion also performs this head tracking.
  This module provides a standalone process if you wish to run head tracking separately.
"""

import math
import time
import argparse
import numpy as np
import zmq

from . import yolo_networking as yn
from .white_point_d435i import _load_aruco_marker_info
from . import aruco_detector as ad
from . import d435i_helpers_without_pyrealsense as dh

try:
    import stretch_body.robot as rb
except Exception:
    rb = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Head tracker: center D405 ArUco in D435i view.')
    p.add_argument('-r', '--remote', action='store_true', help='Subscribe to remote robot D435i stream.')
    p.add_argument('--k-pan', type=float, default=1.0, help='Head pan gain.')
    p.add_argument('--k-tilt', type=float, default=1.0, help='Head tilt gain.')
    p.add_argument('--sweep-speed', type=float, default=0.3, help='Pan sweep speed (rad/s) when marker lost.')
    p.add_argument('--aruco-every-n', type=int, default=3, help='Run ArUco detection every N frames.')
    p.add_argument('--tilt-floor-deg', type=float, default=None,
                   help='Minimum downward tilt (deg, negative) the tracker will not move above.')
    return p


def main():
    if rb is None:
        raise RuntimeError('stretch_body not available on this machine')

    args = build_parser().parse_args()
    robot = rb.Robot(); robot.startup()

    # Subscribe to D435i
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b'')
    sub.setsockopt(zmq.SNDHWM, 1)
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.setsockopt(zmq.CONFLATE, 1)
    addr = 'tcp://' + (yn.robot_ip if args.remote else '127.0.0.1') + ':' + str(yn.d435i_port)
    sub.connect(addr)

    marker_info = _load_aruco_marker_info()
    detector = ad.ArucoDetector(marker_info=marker_info, show_debug_images=False,
                                use_apriltag_refinement=False, brighten_images=False)

    frame_idx = 0
    missing_count = 0
    sweep_dir = -1.0
    last_d405_uv = None
    # Keep tracking for a short window after detection so the head does not
    # immediately fall back into sweep mode if a frame is dropped.
    hold_frames = max(18, int(args.aruco_every_n) * 6)
    reacquire_delay = hold_frames + max(30, int(args.aruco_every_n) * 10)
    tracking_active = False

    try:
        floor_rad = None
        if args.tilt_floor_deg is not None:
            try:
                floor_deg = float(args.tilt_floor_deg)
                floor_rad = float(np.clip(floor_deg, -90.0, 0.0)) * math.pi / 180.0
            except Exception:
                floor_rad = None
        while True:
            frame_idx += 1
            try:
                out = sub.recv_pyobj(flags=0)
            except zmq.error.ZMQError:
                break
            color = out.get('color_image'); info = out.get('depth_camera_info', out.get('color_camera_info'))
            if color is None or info is None:
                time.sleep(0.03)
                continue

            # Detect D405 marker occasionally
            d405_uv = None
            if (frame_idx % max(1, int(args.aruco_every_n))) == 0:
                try:
                    detector.update(color, info)
                    markers = detector.get_detected_marker_dict() or {}
                    d405_entry = markers.get(135)
                    if d405_entry is None:
                        for _mid, mdata in markers.items():
                            mi = mdata.get('info', {}) if isinstance(mdata, dict) else {}
                            if mi.get('name') == 'd405_back':
                                d405_entry = mdata; break
                    if d405_entry is not None:
                        center_uv = d405_entry.get('center_uv')
                        if center_uv is not None:
                            try:
                                d405_uv = (float(center_uv[0]), float(center_uv[1]))
                            except Exception:
                                d405_uv = None
                        if d405_uv is None:
                            pos = d405_entry.get('pos')
                            if pos is not None:
                                try:
                                    uv = dh.pixel_from_3d(np.array(pos, dtype=np.float32), info)
                                    d405_uv = (float(uv[0]), float(uv[1]))
                                except Exception:
                                    d405_uv = None
                except Exception:
                    pass

            if d405_uv is not None:
                last_d405_uv = d405_uv
                missing_count = 0
                tracking_active = True
            else:
                missing_count = min(missing_count + 1, 10**9)
                if last_d405_uv is not None and missing_count <= hold_frames:
                    # Re-use the most recent detection for a short period.
                    d405_uv = last_d405_uv
                elif missing_count > reacquire_delay:
                    tracking_active = False

            # Determine control targets
            h, w = color.shape[:2]
            cx = 0.5 * (w - 1.0)
            cy = 0.5 * (h - 1.0)
            fx = float(info.get('camera_matrix')[0, 0]) if isinstance(info, dict) else 580.0
            fy = float(info.get('camera_matrix')[1, 1]) if isinstance(info, dict) else 580.0

            if tracking_active and d405_uv is not None:
                sweep_dir = -1.0
                # Horizontal pixel error -> head pan, vertical pixel error -> head tilt.
                # The D435i image is rotated 90° clockwise in the GUI/world frame, so
                # GUI-horizontal offsets (right-left) correspond to the original image
                # Y axis, and GUI-vertical offsets (up-down) correspond to the original
                # image X axis. Swap axes accordingly when forming the errors.
                pan_err = (cy - d405_uv[1]) / max(1e-6, fy)
                tilt_err = (d405_uv[0] - cx) / max(1e-6, fx)
                v_pan = float(np.clip(-args.k_pan * pan_err, -1.2, 1.2))
                v_tilt_raw = float(np.clip(-args.k_tilt * tilt_err, -1.0, 1.0))
                if abs(pan_err) < 0.01:
                    v_pan = 0.0
                if abs(tilt_err) < 0.012:
                    v_tilt_raw = 0.0
                v_tilt = v_tilt_raw
                if floor_rad is not None:
                    try:
                        jtilt = robot.head.get_joint('head_tilt') if hasattr(robot, 'head') else None
                        cur = 0.0
                        if jtilt is not None and jtilt.status is not None:
                            cur = float(jtilt.status.get('pos', 0.0))
                        tol_rad = math.radians(0.5)
                        # Above the floor (less downward): allow only downward velocity to return toward the floor
                        if cur > floor_rad + tol_rad and v_tilt > 0.0:
                            v_tilt = 0.0
                            if abs(v_tilt_raw) > 0.01:
                                print(f"[HeadTracker] Blocking upward tilt to respect floor: current={cur:.3f} rad ({cur*180/math.pi:.1f}°), floor={floor_rad:.3f} rad ({floor_rad*180/math.pi:.1f}°)")
                        # Below the floor (more downward): allow only upward motion back toward the floor
                        elif cur < floor_rad - tol_rad and v_tilt < 0.0:
                            v_tilt = 0.0
                            if abs(v_tilt_raw) > 0.01:
                                print(f"[HeadTracker] Blocking further downward tilt: current={cur:.3f} rad ({cur*180/math.pi:.1f}°), floor={floor_rad:.3f} rad ({floor_rad*180/math.pi:.1f}°)")
                        else:
                            # Within tolerance of the floor: only move in the direction that keeps us at/below it
                            if cur >= floor_rad and v_tilt > 0.0:
                                v_tilt = 0.0
                            elif cur <= floor_rad and v_tilt < 0.0:
                                v_tilt = 0.0
                    except Exception as e:
                        print(f"[HeadTracker] Tilt floor check error: {e}")
                        pass
            else:
                if tracking_active:
                    # Should not command motion while we wait for an updated detection.
                    v_pan = 0.0
                    v_tilt = 0.0
                else:
                    # Sweep pan to reacquire; hold tilt
                    v_pan = float(np.clip(args.sweep_speed * sweep_dir, -0.6, 0.6))
                    v_tilt = 0.0
                    if floor_rad is not None:
                        try:
                            jtilt = robot.head.get_joint('head_tilt') if hasattr(robot, 'head') else None
                            cur = 0.0
                            if jtilt is not None and jtilt.status is not None:
                                cur = float(jtilt.status.get('pos', 0.0))
                            tol_rad = math.radians(0.5)
                            if cur > floor_rad + tol_rad:
                                v_tilt = float(np.clip(-0.4 * (cur - floor_rad), -0.4, -0.05))
                        except Exception:
                            pass
                    # Flip direction periodically if missing for a while
                    pan_min = -math.pi / 2.0
                    pan_max = 0.0
                    bound_eps = math.radians(3.0)
                    try:
                        if hasattr(robot, 'head'):
                            j_pan = robot.head.get_joint('head_pan')
                            cur_pan = 0.0
                            if j_pan is not None and j_pan.status is not None:
                                cur_pan = float(j_pan.status.get('pos', 0.0))
                            if cur_pan <= pan_min + bound_eps:
                                sweep_dir = 1.0
                            elif cur_pan >= pan_max - bound_eps:
                                sweep_dir = -1.0
                    except Exception:
                        pass
                    if (missing_count % 120) == 0:
                        sweep_dir = -sweep_dir

            # Apply commands
            try:
                if hasattr(robot, 'head'):
                    robot.head.set_velocity('head_pan', v_pan)
                    robot.head.set_velocity('head_tilt', v_tilt)
                robot.push_command()
            except Exception:
                pass
            time.sleep(0.05)

    finally:
        try:
            if hasattr(robot, 'head'):
                robot.head.set_velocity('head_pan', 0.0)
                robot.head.set_velocity('head_tilt', 0.0)
            robot.push_command()
        except Exception:
            pass
        try:
            robot.stop()
        except Exception:
            pass


if __name__ == '__main__':
    main()
