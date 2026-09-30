"""
Pose/Geometry helpers (skeleton)

This module documents the basic pose abstractions and provides light re-exports
or shims to existing utilities. For concrete robot pose actions (e.g., setting
head tilt or going to a start pose) see VLServo.pose_utils.

Provided for architectural clarity; extend as needed.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Pixel2D:
    x: int
    y: int


@dataclass
class Offset2D:
    dx: float
    dy: float


@dataclass
class Offset3D:
    dx: float
    dy: float
    dz: float


def image_center(width: int, height: int) -> Pixel2D:
    return Pixel2D(int((width - 1) * 0.5), int((height - 1) * 0.5))


def to_tuple2(p: Pixel2D) -> Tuple[int, int]:
    return int(p.x), int(p.y)
