#!/usr/bin/env python3
"""
Camera white-point tracker (D435i)

Responsibilities
- Subscribe to the D435i ZMQ color+depth stream.
- Track a user-picked pixel (x,y in original D435i coordinates) over time via template matching.
- Compute depth at the tracked pixel and derive a 3D grasp_center_xyz.
- Publish results on the existing YOLO ZMQ channel with the schema used by downstream code:
  send_dict = {
      'fingertips': { ... from ArUco on wrist ... },
      'yolo': [{'grasp_center_xyz': np.array([x,y,z]), 'width_m': w}],
      'd405_pixel': (u, v)  # Optional: D405 ArUco pixel in D435i frame
  }

Implementation
- Thin wrapper that reuses VLServo.white_point_d435i for robust behavior.
- Provided for clarity and future extension if a different publisher API is needed.
"""

import argparse
from . import white_point_d435i


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Track and publish a 3D grasp center from a white-point (D435i).')
    p.add_argument('-x', '--pixel-x', dest='x', type=int, required=True,
                   help='Target pixel column (original D435i frame).')
    p.add_argument('-y', '--pixel-y', dest='y', type=int, required=True,
                   help='Target pixel row (original D435i frame).')
    p.add_argument('-r', '--remote', action='store_true',
                   help='Subscribe to remote robot D435i stream.')
    p.add_argument('--radius-m', type=float, default=0.0, help='Optional push radius toward center (m).')
    p.add_argument('--template-size', type=int, default=41, help='Template patch size (odd).')
    p.add_argument('--search-radius', type=int, default=40, help='Search radius (px).')
    p.add_argument('--extra-push-m', type=float, default=0.02, help='Small forward offset (m).')
    return p


def main():
    args = build_parser().parse_args()
    white_point_d435i.main(
        use_remote_computer=args.remote,
        x=args.x,
        y=args.y,
        push_radius_m=args.radius_m,
        template_size=args.template_size,
        search_radius=args.search_radius,
        extra_push_m=args.extra_push_m,
    )


if __name__ == '__main__':
    main()
