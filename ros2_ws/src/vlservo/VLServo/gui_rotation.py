"""
Helpers for dealing with the 90° GUI rotation applied to RealSense images.

The GUI shows the D435i feed rotated 90° clockwise so the cable exits downward.
Any control logic that reasons about "up/down/left/right" in the GUI must map
between the rotated view and the camera's native pixel coordinates.

This module centralizes that math so all components use the same conventions.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

DEFAULT_ROTATION_DEG = -90.0
ROT_ENV_VAR = 'VL_IMAGE_ROTATION_DEG'


def resolve_rotation_degrees(
    rotation_deg: Optional[float] = None,
    default_deg: float = DEFAULT_ROTATION_DEG,
    env_var: str = ROT_ENV_VAR,
) -> float:
    """Return the configured GUI rotation in degrees (camera -> display)."""
    if rotation_deg is not None:
        try:
            return float(rotation_deg)
        except Exception:
            return float(default_deg)
    raw = os.environ.get(env_var)
    if raw is None:
        return float(default_deg)
    try:
        return float(raw)
    except Exception:
        return float(default_deg)


def _rotation_info(
    width: float,
    height: float,
    rotation_deg: Optional[float] = None,
) -> dict:
    """Pre-compute trigonometric helpers for the requested rotation."""
    width = max(1.0, float(width))
    height = max(1.0, float(height))
    rot_deg = resolve_rotation_degrees(rotation_deg)
    # Camera -> display uses the inverse of the GUI rotation, hence the minus.
    theta = math.radians(-rot_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    disp_width = abs(width * cos_t) + abs(height * sin_t)
    disp_height = abs(width * sin_t) + abs(height * cos_t)
    return {
        'rotation_deg': rot_deg,
        'theta': theta,
        'cos': cos_t,
        'sin': sin_t,
        'cam_width': width,
        'cam_height': height,
        'cam_cx': (width - 1.0) / 2.0,
        'cam_cy': (height - 1.0) / 2.0,
        'disp_width': max(1.0, disp_width),
        'disp_height': max(1.0, disp_height),
        'disp_cx': (disp_width - 1.0) / 2.0,
        'disp_cy': (disp_height - 1.0) / 2.0,
    }


def camera_to_display(
    px: float,
    py: float,
    width: float,
    height: float,
    rotation_deg: Optional[float] = None,
) -> Tuple[float, float, float, float, float, float]:
    """
    Map a pixel from camera coordinates into the GUI/display frame.

    Returns:
        display_x, display_y, display_width, display_height, rot_dx, rot_dy
    """
    info = _rotation_info(width, height, rotation_deg)
    dx = float(px) - info['cam_cx']
    dy = float(py) - info['cam_cy']
    rot_dx = dx * info['cos'] - dy * info['sin']
    rot_dy = dx * info['sin'] + dy * info['cos']
    disp_x = rot_dx + info['disp_cx']
    disp_y = rot_dy + info['disp_cy']
    return disp_x, disp_y, info['disp_width'], info['disp_height'], rot_dx, rot_dy


def display_to_camera(
    disp_x: float,
    disp_y: float,
    width: float,
    height: float,
    rotation_deg: Optional[float] = None,
) -> Tuple[float, float]:
    """Inverse of camera_to_display."""
    info = _rotation_info(width, height, rotation_deg)
    dx = float(disp_x) - info['disp_cx']
    dy = float(disp_y) - info['disp_cy']
    theta_inv = -info['theta']
    cos_i = math.cos(theta_inv)
    sin_i = math.sin(theta_inv)
    cam_dx = dx * cos_i - dy * sin_i
    cam_dy = dx * sin_i + dy * cos_i
    px = cam_dx + info['cam_cx']
    py = cam_dy + info['cam_cy']
    return px, py


def compute_display_offsets(
    px: float,
    py: float,
    width: float,
    height: float,
    rotation_deg: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """
    Return normalized horizontal/vertical offsets relative to the GUI frame.

    horizontal_offset > 0 => point appears on the GUI-right.
    vertical_offset > 0   => point appears lower in the GUI view.
    """
    disp_x, disp_y, disp_w, disp_h, rot_dx, rot_dy = camera_to_display(
        px, py, width, height, rotation_deg=rotation_deg
    )
    horizontal = rot_dx / disp_w
    vertical = rot_dy / disp_h
    return horizontal, vertical, rot_dx, rot_dy


def normalized_display_position(
    px: float,
    py: float,
    width: float,
    height: float,
    rotation_deg: Optional[float] = None,
) -> Tuple[float, float]:
    """Return display-space coordinates normalized to [0, 1]."""
    disp_x, disp_y, disp_w, disp_h, _, _ = camera_to_display(
        px, py, width, height, rotation_deg=rotation_deg
    )
    return disp_x / disp_w, disp_y / disp_h
