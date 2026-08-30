#!/usr/bin/env python3
"""D435i-based arm visual servo for LLM grasping.

This module replaces the original D405-centric visual_servoing_demo with a
controller that drives the Stretch arm toward a white-point target selected in
the GUI. It reuses :mod:`VLServo.arm_motion` so the same logic can be invoked
either directly (``python -m VLServo.arm_motion``) or via this wrapper.

Key behavior:
- Subscribe to the D435i ZMQ stream to track the selected white point.
- Extend the arm toward the point while lightly adjusting lift height.
- Rotate the base just enough to keep the point centered; no base translation.
- Keep the head aligned with the target throughout the motion.
- Close the gripper when the point is close and centered.

The CLI mirrors the arm_motion parameters so existing scripts can pass the
initial pixel and optional gains/distances. Compatibility flags ``--yolo`` and
``--exposure`` are accepted (but ignored) so legacy launch commands do not fail.
"""

import argparse

from . import arm_motion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='LLM Arm Visual Servo',
        description='Move the Stretch arm toward a white-point using the D435i camera.',
    )
    parser.add_argument('-x', '--pixel-x', dest='x', type=int, required=True,
                        help='Target pixel column (original D435i frame, not GUI-rotated).')
    parser.add_argument('-y', '--pixel-y', dest='y', type=int, required=True,
                        help='Target pixel row (original D435i frame, not GUI-rotated).')
    parser.add_argument('-r', '--remote', action='store_true',
                        help='Subscribe to the robot-hosted D435i stream instead of localhost.')
    parser.add_argument('--stop-z-m', type=float, default=0.38,
                        help='Depth (m) where the arm stops extending before the final close.')
    parser.add_argument('--grasp-z-m', type=float, default=0.28,
                        help='Depth (m) that triggers a gripper close once aligned.')
    parser.add_argument('--max-time-s', type=float, default=600.0,
                        help='Safety timeout for the controller loop.')
    parser.add_argument('--k-ext', type=float, default=0.8,
                        help='Extension gain (m/s per meter of depth error).')
    parser.add_argument('--k-lift', type=float, default=0.6,
                        help='Lift gain (m/s per normalized vertical pixel error).')
    parser.add_argument('--k-pan', type=float, default=0.8,
                        help='Head pan gain (rad/s per normalized horizontal pixel error).')
    parser.add_argument('--k-tilt', type=float, default=0.8,
                        help='Head tilt gain (rad/s per tilt error).')
    parser.add_argument('--head-tilt-bias', type=float, default=0.0,
                        help='Additional downward tilt bias (radians). Set 0.0 to let ArUco drive alignment.')
    parser.add_argument('--aruco-weight', type=float, default=0.85,
                        help='Blend weight (0-1) for D405 ArUco alignment in head control.')
    parser.add_argument('--aruco-update-n', type=int, default=3,
                        help='Run D405 ArUco detection every N frames (>=1).')
    # Head tracking mode: default to ArUco-only for LLM grasping
    parser.add_argument('--no-head-aruco-only', dest='head_aruco_only', action='store_false',
                        help='Disable ArUco-only head tracking (fall back to blend).')
    parser.set_defaults(head_aruco_only=True)
    parser.add_argument('--no-invert-yaw', dest='invert_yaw', action='store_false',
                        help='Disable the default yaw inversion (use camera frame yaw directly).')
    parser.set_defaults(invert_yaw=True)

    # Compatibility flags from the legacy visual_servoing_demo_llm interface.
    parser.add_argument('--yolo', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--exposure', type=str, help=argparse.SUPPRESS)
    return parser


def main(**override_kwargs):
    parser = build_parser()
    args = parser.parse_args()

    # Allow programmatic invocation by passing override kwargs.
    params = {
        'x': args.x,
        'y': args.y,
        'use_remote': args.remote,
        'stop_z_m': args.stop_z_m,
        'grasp_z_m': args.grasp_z_m,
        'max_time_s': args.max_time_s,
        'k_ext': args.k_ext,
        'k_lift': args.k_lift,
        'k_pan': args.k_pan,
        'k_tilt': args.k_tilt,
        'head_tilt_bias': args.head_tilt_bias,
        'aruco_weight': args.aruco_weight,
        'aruco_update_n': args.aruco_update_n,
        'invert_yaw': args.invert_yaw,
        'head_aruco_only': bool(args.head_aruco_only),
    }
    if override_kwargs:
        params.update(override_kwargs)

    arm_motion.main(**params)


if __name__ == '__main__':
    main()
