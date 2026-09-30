#!/usr/bin/env python3
"""
Custom pose utilities for VLServo GUI.

When starting LLM grasping, we want the gripper oriented forward and the head
to look at the gripper. We bias the head further down and to the right so the
D405 ArUco marker stays visible near the middle-right of the D435i view.
"""

import math
import time
from typing import Optional, Tuple

try:
    import stretch_body.robot as rb
except Exception:  # pragma: no cover
    rb = None


def _with_robot():
    if rb is None:
        raise RuntimeError("stretch_body not available on this machine")
    robot = rb.Robot()
    # Retry startup briefly if ports are busy or being released
    last_err = None
    for i in range(3):
        try:
            robot.startup()
            last_err = None
            break
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    if last_err is not None:
        # Re-raise original error after retries
        raise last_err
    return robot


def set_head_tilt_deg(deg: float) -> None:
    """Set head tilt (pitch) angle in degrees immediately; 0 deg looks forward."""
    robot = _with_robot()
    try:
        tilt_rad = float(deg) * math.pi / 180.0
        if hasattr(robot, 'head'):
            robot.head.move_to('head_tilt', tilt_rad)
        robot.push_command(); robot.wait_command()
    finally:
        try:
            robot.stop()
        except Exception:
            pass


def go_to_start_pose(head_tilt_deg: Optional[float] = 0.0, head_pan_rad: Optional[float] = 0.0) -> None:
    """Move to a grasp start pose: head forward, gripper forward.

    - Head pan -> 0.0 rad (forward). If head_pan_rad is None, do not move head pan.
    - Head tilt -> provided degrees (default 0 deg = forward). If None, do not move head tilt.
    - Wrist yaw -> +pi/2 rad (gripper facing forward on Stretch 3)
    - Arm/Lift/Gripper to safe nominal values
    """
    robot = _with_robot()
    try:
        if hasattr(robot, 'head'):
            try:
                if head_pan_rad is not None:
                    robot.head.move_to('head_pan', float(head_pan_rad))
            except Exception:
                pass
            try:
                if head_tilt_deg is not None:
                    tilt_rad = float(head_tilt_deg) * math.pi / 180.0
                    robot.head.move_to('head_tilt', tilt_rad)
            except Exception:
                pass
        # Align gripper forward if wrist yaw joint exists; pitch up to (at least) horizontal
        try:
            if hasattr(robot, 'end_of_arm'):
                jy = robot.end_of_arm.get_joint('wrist_yaw')
                if jy is not None:
                    jy.move_to(math.pi / 2.0)
                jp = robot.end_of_arm.get_joint('wrist_pitch')
                if jp is not None:
                    # Pitch upward a bit to ensure horizontal-or-better
                    jp.move_to(WRIST_PITCH_START_RAD)
                jr = robot.end_of_arm.get_joint('wrist_roll')
                if jr is not None:
                    # Keep roll roughly horizontal
                    jr.move_to(0.0)
        except Exception:
            pass
        # Nominal arm/lift/gripper
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
    finally:
        try:
            robot.stop()
        except Exception:
            pass


# Tunables
# On this robot, right is negative pan. Turn more so gripper is visible.
PAN_OFFSET = -0.68      # rad, right turn (~39°) keeps D405 marker in frame
TILT_BIAS = -0.58       # rad, look further downward than purely geometric
HEAD_PAN_LIMIT = 1.5    # rad
HEAD_TILT_MIN = -1.57   # rad
HEAD_TILT_MAX = 0.40    # rad

# Start pose wrist pitch (positive = up). Adjust to match your robot.
WRIST_PITCH_START_DEG = 10.0
WRIST_PITCH_START_RAD = WRIST_PITCH_START_DEG * math.pi / 180.0


def point_head_to_gripper() -> None:
    """Aim head toward the gripper using a simple kinematic approximation.

    Reads current arm extension and lift height to compute a tilt angle that
    looks at the gripper region in front of the robot. Pan is NOT forced to a
    fixed offset anymore so that downstream controllers (arm_motion) can
    continuously steer pan to keep the D405 ArUco centered. We preserve a
    gentle downward tilt bias to help keep the gripper in view initially.
    """
    robot = _with_robot()
    try:
        robot.pull_status()
        # Defaults in case status is unavailable
        arm_ext = 0.3
        lift = 0.7
        head_pan = 0.0
        try:
            if hasattr(robot, 'arm') and robot.arm.status is not None:
                arm_ext = float(robot.arm.status.get('pos', arm_ext))
        except Exception:
            pass
        try:
            if hasattr(robot, 'lift') and robot.lift.status is not None:
                lift = float(robot.lift.status.get('pos', lift))
        except Exception:
            pass
        try:
            if hasattr(robot, 'head'):
                jpan = robot.head.get_joint('head_pan')
                if jpan is not None and jpan.status is not None:
                    head_pan = float(jpan.status.get('pos', head_pan))
        except Exception:
            pass

        # Simple geometric model (meters)
        X_BASE = 0.30  # base-to-gripper offset when arm=0
        x_rel = max(0.05, X_BASE + arm_ext)
        Z_HEAD_ABOVE_CARRIAGE = 0.25
        Z_GRIP_ABOVE_CARRIAGE = 0.05
        z_head = lift + Z_HEAD_ABOVE_CARRIAGE
        z_grip = lift + Z_GRIP_ABOVE_CARRIAGE
        dz = z_grip - z_head  # typically negative

        tilt = math.atan2(dz, x_rel) + TILT_BIAS
        tilt = float(max(HEAD_TILT_MIN, min(HEAD_TILT_MAX, tilt)))

        if hasattr(robot, 'head'):
            # Do not force a fixed pan offset; keep current pan and only aim tilt.
            try:
                jpan = robot.head.get_joint('head_pan')
                cur_pan = 0.0
                if jpan is not None and jpan.status is not None:
                    cur_pan = float(jpan.status.get('pos', 0.0))
                tgt_pan = float(max(-HEAD_PAN_LIMIT, min(HEAD_PAN_LIMIT, cur_pan)))
                robot.head.move_to('head_pan', tgt_pan)
            except Exception:
                pass
            robot.head.move_to('head_tilt', tilt)
        robot.push_command(); robot.wait_command()
    finally:
        try:
            robot.stop()
        except Exception:
            pass


def get_head_pan_tilt() -> Tuple[float, float]:
    """Return current head pan/tilt in radians as (pan, tilt).

    Falls back to (0.0, 0.0) if status is unavailable.
    """
    robot = _with_robot()
    try:
        pan = 0.0
        tilt = 0.0
        try:
            robot.pull_status()
            if hasattr(robot, 'head'):
                jpan = robot.head.get_joint('head_pan')
                jtilt = robot.head.get_joint('head_tilt')
                if jpan is not None and jpan.status is not None:
                    pan = float(jpan.status.get('pos', pan))
                if jtilt is not None and jtilt.status is not None:
                    tilt = float(jtilt.status.get('pos', tilt))
        except Exception:
            pass
        return float(pan), float(tilt)
    finally:
        try:
            robot.stop()
        except Exception:
            pass


def get_gripper_forward_xyz() -> Tuple[float, float, float]:
    """Approximate gripper tip position (x,y,z) in meters in the base frame.

    Uses a simple kinematic model consistent with point_head_to_gripper().
    y is assumed to be 0 (centerline). This is an approximation intended for
    aiming the head, not precise manipulation.
    """
    robot = _with_robot()
    try:
        robot.pull_status()
        arm_ext = 0.3
        lift = 0.6
        try:
            if hasattr(robot, 'arm') and robot.arm.status is not None:
                arm_ext = float(robot.arm.status.get('pos', arm_ext))
        except Exception:
            pass
        try:
            if hasattr(robot, 'lift') and robot.lift.status is not None:
                lift = float(robot.lift.status.get('pos', lift))
        except Exception:
            pass
        X_BASE = 0.30
        x_rel = max(0.05, X_BASE + arm_ext)
        Z_GRIP_ABOVE_CARRIAGE = 0.05
        z = float(lift + Z_GRIP_ABOVE_CARRIAGE)
        return float(x_rel), 0.0, float(z)
    finally:
        try:
            robot.stop()
        except Exception:
            pass


def _maybe_enable(obj):
    try:
        if hasattr(obj, 'enable_torque'):
            obj.enable_torque(); return True
        if hasattr(obj, 'set_torque_enable'):
            obj.set_torque_enable(True); return True
        if hasattr(obj, 'set_compliant'):
            obj.set_compliant(False); return True
        if hasattr(obj, 'set_stiffness'):
            obj.set_stiffness(1.0); return True
        if hasattr(obj, 'stiffness'):
            try:
                obj.stiffness = 1.0; return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _maybe_disable(obj):
    try:
        if hasattr(obj, 'disable_torque'):
            obj.disable_torque(); return True
        if hasattr(obj, 'set_torque_enable'):
            obj.set_torque_enable(False); return True
        if hasattr(obj, 'set_compliant'):
            obj.set_compliant(True); return True
        if hasattr(obj, 'set_stiffness'):
            obj.set_stiffness(0.0); return True
        if hasattr(obj, 'stiffness'):
            try:
                obj.stiffness = 0.0; return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def lock_head_and_gripper() -> None:
    """Enable holding torque on head pan/tilt and end-of-arm joints (wrist_yaw, gripper)."""
    robot = _with_robot()
    try:
        # Head motors
        try:
            head_objs = []
            for attr in ['get_joint', 'motors', 'dxl', 'joints']:
                if hasattr(robot.head, attr):
                    obj = getattr(robot.head, attr)
                    if callable(obj) and attr == 'get_joint':
                        for name in ['head_pan', 'head_tilt']:
                            try:
                                j = robot.head.get_joint(name)
                                if j is not None:
                                    head_objs.append(j)
                            except Exception:
                                pass
                    elif isinstance(obj, dict):
                        head_objs.extend(list(obj.values()))
            for j in head_objs:
                _maybe_enable(j)
        except Exception:
            pass

        # End-of-arm: wrist_yaw and gripper
        try:
            if hasattr(robot, 'end_of_arm'):
                for name in ['wrist_yaw', 'stretch_gripper']:
                    try:
                        j = robot.end_of_arm.get_joint(name)
                        if j is not None:
                            _maybe_enable(j)
                    except Exception:
                        pass
        except Exception:
            pass

        robot.push_command(); robot.wait_command()
    finally:
        try:
            robot.stop()
        except Exception:
            pass


def unlock_head_and_gripper() -> None:
    """Disable holding torque (make compliant) on head pan/tilt and end-of-arm joints."""
    robot = _with_robot()
    try:
        try:
            head_objs = []
            for attr in ['get_joint', 'motors', 'dxl', 'joints']:
                if hasattr(robot.head, attr):
                    obj = getattr(robot.head, attr)
                    if callable(obj) and attr == 'get_joint':
                        for name in ['head_pan', 'head_tilt']:
                            try:
                                j = robot.head.get_joint(name)
                                if j is not None:
                                    head_objs.append(j)
                            except Exception:
                                pass
                    elif isinstance(obj, dict):
                        head_objs.extend(list(obj.values()))
            for j in head_objs:
                _maybe_disable(j)
        except Exception:
            pass
        try:
            if hasattr(robot, 'end_of_arm'):
                for name in ['wrist_yaw', 'stretch_gripper']:
                    try:
                        j = robot.end_of_arm.get_joint(name)
                        if j is not None:
                            _maybe_disable(j)
                    except Exception:
                        pass
        except Exception:
            pass
        robot.push_command(); robot.wait_command()
    finally:
        try:
            robot.stop()
        except Exception:
            pass


def nudge_head_pan_tilt(pan_delta: float, tilt_delta: float, max_step: float = 0.2) -> None:
    """Incrementally adjust head pan/tilt by small deltas (radians).

    Clamps deltas to +/- max_step and absolute positions to safe limits.
    """
    robot = _with_robot()
    try:
        cur_pan = 0.0
        cur_tilt = 0.0
        try:
            jpan = robot.head.get_joint('head_pan') if hasattr(robot, 'head') else None
            jtilt = robot.head.get_joint('head_tilt') if hasattr(robot, 'head') else None
            robot.pull_status()
            if jpan is not None and jpan.status is not None:
                cur_pan = float(jpan.status.get('pos', cur_pan))
            if jtilt is not None and jtilt.status is not None:
                cur_tilt = float(jtilt.status.get('pos', cur_tilt))
        except Exception:
            pass

        # Clamp deltas
        pd = float(max(-max_step, min(max_step, pan_delta)))
        td = float(max(-max_step, min(max_step, tilt_delta)))
        tgt_pan = float(max(-HEAD_PAN_LIMIT, min(HEAD_PAN_LIMIT, cur_pan + pd)))
        tgt_tilt = float(max(HEAD_TILT_MIN, min(HEAD_TILT_MAX, cur_tilt + td)))

        if hasattr(robot, 'head'):
            robot.head.move_to('head_pan', tgt_pan)
            robot.head.move_to('head_tilt', tgt_tilt)
        robot.push_command(); robot.wait_command()
    finally:
        try:
            robot.stop()
        except Exception:
            pass
