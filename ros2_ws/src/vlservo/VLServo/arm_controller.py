#!/usr/bin/env python3
"""
Arm + Lift + Base-Yaw controller (LLM White-Point)

Overview
- Subscribes to the D435i ZMQ camera stream (see send_d435i_images or GUI direct publisher).
- Tracks a user-selected white point (pixel x,y in original D435i coords).
- Computes white-point relative offset to the gripper’s D405 ArUco marker in the same D435i image:
  dx (left/right), dy (up/down), and dz (depth difference) and applies:
  - base yaw: correct horizontal dx (no base translation)
  - lift: correct vertical dy
  - arm extension: correct dz (extend if white point is farther; retract if closer)
  - gripper close: when dx, dy within thresholds and depth near target, close to grasp

Notes
- This module is a thin entrypoint that reuses VLServo.arm_motion for the full control logic.
- Use this script if you prefer a named binary that aligns with system diagrams.

CLI
-r / --remote   subscribe to robot-hosted stream; otherwise use localhost
-x, -y          initial D435i pixel to track (original, unrotated camera frame)
Other tuning args are forwarded to arm_motion.
"""

import argparse

from . import arm_motion


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Arm + Lift + Yaw controller for white-point visual servo.')
    p.add_argument('-x', '--pixel-x', dest='x', type=int, required=True,
                   help='Target pixel column (original D435i frame).')
    p.add_argument('-y', '--pixel-y', dest='y', type=int, required=True,
                   help='Target pixel row (original D435i frame).')
    p.add_argument('-r', '--remote', action='store_true',
                   help='Subscribe to remote robot D435i stream.')
    p.add_argument('--stop-z-m', type=float, default=0.38, help='Depth to stop extension (m).')
    p.add_argument('--grasp-z-m', type=float, default=0.28, help='Depth to trigger gripper close (m).')
    p.add_argument('--max-time-s', type=float, default=600.0, help='Safety timeout.')
    p.add_argument('--k-ext', type=float, default=0.8, help='Arm extension gain.')
    p.add_argument('--k-lift', type=float, default=0.6, help='Lift gain.')
    p.add_argument('--k-pan', type=float, default=0.8, help='Head pan gain (rad/s per error).')
    p.add_argument('--k-tilt', type=float, default=0.8, help='Head tilt gain (rad/s per error).')
    p.add_argument('--head-aruco-only', action='store_true', help='Force ArUco-only head tracking.')
    p.add_argument('--aruco-update-n', type=int, default=3, help='Run ArUco detection every N frames.')
    p.add_argument('--no-invert-yaw', dest='invert_yaw', action='store_false',
                   help='Use camera frame yaw directly; default is inverted.')
    p.set_defaults(invert_yaw=True)
    return p


def main():
    args = build_parser().parse_args()

    arm_motion.main(
        x=args.x,
        y=args.y,
        use_remote=args.remote,
        stop_z_m=args.stop_z_m,
        grasp_z_m=args.grasp_z_m,
        max_time_s=args.max_time_s,
        k_ext=args.k_ext,
        k_lift=args.k_lift,
        k_pan=args.k_pan,
        k_tilt=args.k_tilt,
        invert_yaw=args.invert_yaw,
        aruco_update_n=args.aruco_update_n,
        head_aruco_only=bool(args.head_aruco_only),
    )


if __name__ == '__main__':
    main()
