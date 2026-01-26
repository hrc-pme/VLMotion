#!/usr/bin/env python3
"""Convert the tracked white point into base_link coordinates and press an elevator button."""

import argparse
import json
import math
import os
import time
from typing import Dict, Optional

import numpy as np
import zmq

from . import stretch_tf
from . import yolo_networking as yn

try:
    import stretch_body.robot as rb
except Exception:  # pragma: no cover
    rb = None


def _default_urdf() -> str:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pkg_dir, "stretch_uncalibrated.urdf")


class ButtonPressExecutor:
    def __init__(
        self,
        urdf_path: str,
        camera_frame: str,
        approach_distance: float,
        lift_offset: float,
        execute_motion: bool,
        remote_subscriber: bool,
        timeout_s: float,
        head_pan_override: Optional[float] = None,
        head_tilt_override: Optional[float] = None,
        lift_min: float = 0.05,
        lift_max: float = 1.1,
    ) -> None:
        self.transforms = stretch_tf.StretchTransforms(urdf_path)
        self.camera_frame = camera_frame
        self.approach_distance = approach_distance
        self.lift_offset = lift_offset
        self.execute_motion = execute_motion
        self.remote_subscriber = remote_subscriber
        self.timeout_s = timeout_s
        self.head_pan_override = head_pan_override
        self.head_tilt_override = head_tilt_override
        self.lift_min = lift_min
        self.lift_max = lift_max

        self.robot = None
        need_robot = execute_motion or head_pan_override is None or head_tilt_override is None
        if need_robot:
            if rb is None:
                raise RuntimeError("stretch_body is unavailable; cannot read joint states or command motion")
            self.robot = rb.Robot()
            self.robot.startup()

        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        if remote_subscriber:
            address = "tcp://" + yn.robot_ip + ":" + str(yn.yolo_port)
        else:
            address = "tcp://127.0.0.1:" + str(yn.yolo_port)
        self.socket.connect(address)

    def close(self) -> None:
        try:
            self.socket.close(0)
        except Exception:
            pass
        if self.robot is not None:
            try:
                self.robot.stop()
            except Exception:
                pass

    def _current_head_pose(self) -> Dict[str, float]:
        if self.robot is not None:
            try:
                self.robot.pull_status()
            except Exception:
                pass
            pan = self.robot.head.status.get("head_pan", {}).get("pos")
            tilt = self.robot.head.status.get("head_tilt", {}).get("pos")
        else:
            pan = None
            tilt = None
        pan = pan if pan is not None else self.head_pan_override
        tilt = tilt if tilt is not None else self.head_tilt_override
        if pan is None or tilt is None:
            raise RuntimeError("Head pan/tilt unavailable. Provide overrides or run on the robot.")
        return {"joint_head_pan": float(pan), "joint_head_tilt": float(tilt)}

    def _camera_pose(self, joint_positions: Dict[str, float]) -> Dict[str, np.ndarray]:
        T = self.transforms.get_transform(self.camera_frame, joint_positions=joint_positions)
        pose = stretch_tf.matrix_to_translation_quaternion(T)
        return pose

    def _plan_motion(self, button_base_xyz: np.ndarray) -> Dict[str, float]:
        planar = math.hypot(button_base_xyz[0], button_base_xyz[1])
        heading = math.atan2(button_base_xyz[1], button_base_xyz[0])
        forward = planar - self.approach_distance
        lift_goal = np.clip(button_base_xyz[2] + self.lift_offset, self.lift_min, self.lift_max)
        return {
            "heading": heading,
            "heading_deg": math.degrees(heading),
            "forward": forward,
            "planar_distance": planar,
            "button_height": button_base_xyz[2],
            "lift_goal": lift_goal,
        }

    def _execute(self, plan: Dict[str, float]) -> None:
        if self.robot is None:
            raise RuntimeError("Cannot execute motion without a connected robot")
        heading = plan["heading"]
        forward = plan["forward"]
        if abs(heading) > math.radians(1.0):
            self.robot.base.rotate_by(heading)
            self.robot.push_command()
            self.robot.wait_command()
        if abs(forward) > 0.01:
            self.robot.base.translate_by(forward)
            self.robot.push_command()
            self.robot.wait_command()
        lift_goal = plan.get("lift_goal")
        if lift_goal is not None:
            self.robot.lift.move_to(lift_goal)
            self.robot.push_command()
            self.robot.wait_command()

    def run(self) -> Optional[Dict[str, float]]:
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        deadline = time.time() + self.timeout_s
        try:
            while time.time() < deadline:
                socks = dict(poller.poll(timeout=200))
                if self.socket not in socks:
                    continue
                msg = self.socket.recv_pyobj()
                plan = self._handle_message(msg)
                if plan:
                    if self.execute_motion:
                        self._execute(plan)
                    return plan
        finally:
            self.close()
        return None

    def _handle_message(self, msg) -> Optional[Dict[str, float]]:
        if not isinstance(msg, dict):
            return None
        if not msg.get("white_point_visible"):
            return None
        detections = msg.get("yolo") or []
        if not detections:
            return None
        target = detections[0].get("grasp_center_xyz")
        if target is None:
            return None
        target = np.array(target, dtype=float)
        joints = self._current_head_pose()
        button_in_base = self.transforms.transform_point(target, self.camera_frame, joint_positions=joints)
        plan = self._plan_motion(button_in_base)
        plan.update({
            "button_in_base": button_in_base.tolist(),
            "camera_pose": self._camera_pose(joints),
        })
        if self.robot is not None:
            base_status = {
                "x": self.robot.base.status.get("x"),
                "y": self.robot.base.status.get("y"),
                "theta": self.robot.base.status.get("theta"),
            }
            plan["base_pose"] = base_status
        print(json.dumps(plan, indent=2, default=lambda o: o if isinstance(o, (int, float, str, list, dict)) else list(o)))
        return plan


def _deg_to_rad(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value) * math.pi / 180.0


def main():
    parser = argparse.ArgumentParser(description="Drive the Stretch base/lift to an elevator button detected in the head camera.")
    parser.add_argument("--camera-frame", default="camera_depth_optical_frame",
                        help="Camera optical frame to transform from (default assumes depth intrinsics).")
    parser.add_argument("--urdf", default=_default_urdf(), help="Path to the Stretch URDF used for transforms.")
    parser.add_argument("--approach-distance", type=float, default=0.45,
                        help="Desired standoff distance between base_link origin and the button (m).")
    parser.add_argument("--lift-offset", type=float, default=-0.02,
                        help="Offset applied to the button height to compute the lift goal (m).")
    parser.add_argument("--lift-min", type=float, default=0.05, help="Minimum lift position (m).")
    parser.add_argument("--lift-max", type=float, default=1.1, help="Maximum lift position (m).")
    parser.add_argument("--execute", action="store_true", help="Command the robot instead of only printing the plan.")
    parser.add_argument("--remote", action="store_true", help="Subscribe to YOLO data from the robot_ip instead of localhost.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Time to wait for a detection (s).")
    parser.add_argument("--head-pan-deg", type=float, help="Override head pan angle in degrees (used if robot is not connected).")
    parser.add_argument("--head-tilt-deg", type=float, help="Override head tilt angle in degrees (used if robot is not connected).")
    args = parser.parse_args()

    executor = ButtonPressExecutor(
        urdf_path=args.urdf,
        camera_frame=args.camera_frame,
        approach_distance=args.approach_distance,
        lift_offset=args.lift_offset,
        execute_motion=args.execute,
        remote_subscriber=args.remote,
        timeout_s=args.timeout,
        head_pan_override=_deg_to_rad(args.head_pan_deg),
        head_tilt_override=_deg_to_rad(args.head_tilt_deg),
        lift_min=args.lift_min,
        lift_max=args.lift_max,
    )
    plan = executor.run()
    if plan is None:
        raise SystemExit("No white point detection received before timeout.")


if __name__ == "__main__":
    main()
