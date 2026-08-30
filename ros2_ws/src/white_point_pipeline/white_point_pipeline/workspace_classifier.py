#!/usr/bin/env python3
"""Pure workspace classification used by the direct motion state machine."""

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class WorkspaceAssessment:
    """Reachability result for one target observation."""

    scenario: str
    height_reachable: bool
    horizontal_reachable: bool
    height_error: Optional[float]
    horizontal_distance: float
    horizontal_shortfall: float


def choose_parallel_reach_yaw(candidate_yaw, target_bearing, contact_offset):
    """Choose the parallel base yaw whose positive contact axis faces target."""
    def wrap(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    candidate = wrap(candidate_yaw)
    flipped = wrap(candidate + math.pi)
    candidate_contact = wrap(candidate + contact_offset)
    flipped_contact = wrap(flipped + contact_offset)
    candidate_error = abs(wrap(candidate_contact - target_bearing))
    flipped_error = abs(wrap(flipped_contact - target_bearing))
    if flipped_error < candidate_error:
        return flipped, flipped_contact, candidate_error, flipped_error
    return candidate, candidate_contact, candidate_error, flipped_error


def classify_workspace(
    *,
    target_z: float,
    gripper_contact_z: Optional[float],
    horizontal_distance: float,
    height_tolerance: float,
    horizontal_min: float,
    horizontal_max: float,
    horizontal_tolerance: float,
) -> WorkspaceAssessment:
    """Classify a target without mixing its height into planar reachability.

    ``horizontal_max`` is the stage-one preparation radius, not the bare
    telescoping-arm travel. ``horizontal_tolerance`` expands both ends of that
    range so a few centimetres of sensor/TF noise do not trigger base motion.
    """
    min_reach = max(0.0, horizontal_min - max(0.0, horizontal_tolerance))
    max_reach = max(min_reach, horizontal_max + max(0.0, horizontal_tolerance))
    distance = max(0.0, horizontal_distance)

    if distance < min_reach:
        horizontal_shortfall = distance - min_reach
    elif distance > max_reach:
        horizontal_shortfall = distance - max_reach
    else:
        horizontal_shortfall = 0.0
    horizontal_reachable = horizontal_shortfall == 0.0

    height_error = None
    height_reachable = False
    if gripper_contact_z is not None:
        height_error = target_z - gripper_contact_z
        height_reachable = abs(height_error) <= max(0.0, height_tolerance)

    scenarios = {
        (True, True): 'workspace',
        (False, True): 'height_only',
        (True, False): 'base_only',
        (False, False): 'height_and_base',
    }
    return WorkspaceAssessment(
        scenario=scenarios[(height_reachable, horizontal_reachable)],
        height_reachable=height_reachable,
        horizontal_reachable=horizontal_reachable,
        height_error=height_error,
        horizontal_distance=distance,
        horizontal_shortfall=horizontal_shortfall,
    )
