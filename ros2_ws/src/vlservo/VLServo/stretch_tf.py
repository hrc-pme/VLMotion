"""Utility functions for computing Stretch URDF transforms without ROS tf."""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np


def _parse_floats(text: Optional[str], length: int, default: float = 0.0) -> List[float]:
    if not text:
        return [default] * length
    vals = [float(v) for v in text.strip().split() if v]
    if len(vals) != length:
        raise ValueError(f"Expected {length} values, got {vals}")
    return vals


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _axis_angle_to_matrix(axis: Iterable[float], angle: float) -> np.ndarray:
    axis_arr = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis_arr)
    if norm == 0:
        return np.eye(3)
    axis_arr /= norm
    x, y, z = axis_arr
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _matrix_to_transform(R: np.ndarray, t: Iterable[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array(t, dtype=float)
    return T


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    joint_type: str
    origin_xyz: List[float]
    origin_rpy: List[float]
    axis: List[float]

    def transform(self, position: float = 0.0) -> np.ndarray:
        base = _matrix_to_transform(_rpy_to_matrix(*self.origin_rpy), self.origin_xyz)
        motion = np.eye(4)
        if self.joint_type in ("revolute", "continuous"):
            motion[:3, :3] = _axis_angle_to_matrix(self.axis, position)
        elif self.joint_type == "prismatic":
            motion[:3, 3] = np.array(self.axis, dtype=float) * position
        return base @ motion


class StretchTransforms:
    """Compute link poses using only the Stretch URDF."""

    def __init__(self, urdf_path: str):
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self.urdf_path = urdf_path
        self._child_to_joint: Dict[str, Joint] = {}
        self._parse_urdf()

    def _parse_urdf(self) -> None:
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        for joint_elem in root.findall("joint"):
            name = joint_elem.attrib.get("name")
            joint_type = joint_elem.attrib.get("type", "fixed")
            parent_elem = joint_elem.find("parent")
            child_elem = joint_elem.find("child")
            origin_elem = joint_elem.find("origin")
            axis_elem = joint_elem.find("axis")
            if parent_elem is None or child_elem is None or not name:
                continue
            xyz = _parse_floats(origin_elem.attrib.get("xyz") if origin_elem is not None else None, 3)
            rpy = _parse_floats(origin_elem.attrib.get("rpy") if origin_elem is not None else None, 3)
            axis = _parse_floats(axis_elem.attrib.get("xyz") if axis_elem is not None else None, 3, default=0.0)
            if axis == [0.0, 0.0, 0.0]:
                axis = [1.0, 0.0, 0.0]
            joint = Joint(
                name=name,
                parent=parent_elem.attrib["link"],
                child=child_elem.attrib["link"],
                joint_type=joint_type,
                origin_xyz=xyz,
                origin_rpy=rpy,
                axis=axis,
            )
            self._child_to_joint[joint.child] = joint

    def _chain(self, target_link: str, base_link: str) -> List[Joint]:
        chain: List[Joint] = []
        current = target_link
        while current != base_link:
            joint = self._child_to_joint.get(current)
            if joint is None:
                raise ValueError(f"No joint connects {current} to {base_link}")
            chain.append(joint)
            current = joint.parent
        chain.reverse()
        return chain

    def get_transform(
        self,
        target_link: str,
        joint_positions: Optional[Dict[str, float]] = None,
        base_link: str = "base_link",
    ) -> np.ndarray:
        joint_positions = joint_positions or {}
        T = np.eye(4)
        for joint in self._chain(target_link, base_link):
            pos = joint_positions.get(joint.name, 0.0)
            T = T @ joint.transform(pos)
        return T

    def transform_point(
        self,
        point: Iterable[float],
        from_link: str,
        joint_positions: Optional[Dict[str, float]] = None,
        base_link: str = "base_link",
    ) -> np.ndarray:
        T = self.get_transform(from_link, joint_positions=joint_positions, base_link=base_link)
        vec = np.array(list(point) + [1.0], dtype=float)
        return (T @ vec)[:3]


def matrix_to_translation_quaternion(T: np.ndarray) -> Dict[str, np.ndarray]:
    translation = T[:3, 3]
    R = T[:3, :3]
    qw = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    denom = 4.0 * qw if qw != 0 else 1e-9
    qx = (R[2, 1] - R[1, 2]) / denom
    qy = (R[0, 2] - R[2, 0]) / denom
    qz = (R[1, 0] - R[0, 1]) / denom
    quaternion = np.array([qx, qy, qz, qw])
    return {"translation": translation, "quaternion": quaternion}

