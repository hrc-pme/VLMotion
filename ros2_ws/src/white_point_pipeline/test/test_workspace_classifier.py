import math

from white_point_pipeline.workspace_classifier import (
    choose_parallel_reach_yaw,
    classify_workspace,
)


def assess(target_z, gripper_z, distance):
    return classify_workspace(
        target_z=target_z,
        gripper_contact_z=gripper_z,
        horizontal_distance=distance,
        height_tolerance=0.04,
        horizontal_min=0.08,
        horizontal_max=0.70,
        horizontal_tolerance=0.05,
    )


def test_target_inside_workspace():
    result = assess(0.80, 0.78, 0.40)
    assert result.scenario == 'workspace'


def test_only_height_requires_motion():
    result = assess(0.90, 0.78, 0.40)
    assert result.scenario == 'height_only'


def test_only_base_requires_motion():
    result = assess(0.80, 0.78, 0.90)
    assert result.scenario == 'base_only'


def test_height_and_base_require_motion():
    result = assess(0.90, 0.78, 0.90)
    assert result.scenario == 'height_and_base'


def test_horizontal_margin_avoids_unnecessary_base_motion():
    result = assess(0.80, 0.78, 0.74)
    assert result.horizontal_reachable


def test_logged_robot_distance_is_inside_preparation_radius():
    # Log target Z 0.943 with fixed gripper offset 0.100 gives required
    # lift 0.843. Current lift was about 0.815, so the 2.8 cm correction is
    # inside the 4 cm height tolerance.
    result = assess(0.843, 0.815, 0.660)
    assert result.scenario == 'workspace'


def test_missing_gripper_tf_conservatively_requests_height_motion():
    result = assess(0.80, None, 0.40)
    assert not result.height_reachable


def test_parallel_yaw_faces_side_arm_toward_logged_target():
    chosen_yaw, contact_yaw, candidate_error, flipped_error = (
        choose_parallel_reach_yaw(
            math.radians(179.5),
            math.radians(-88.8),
            math.radians(-90.0),
        )
    )
    assert math.isclose(math.degrees(chosen_yaw), -0.5, abs_tol=0.01)
    assert math.isclose(math.degrees(contact_yaw), -90.5, abs_tol=0.01)
    assert flipped_error < candidate_error
