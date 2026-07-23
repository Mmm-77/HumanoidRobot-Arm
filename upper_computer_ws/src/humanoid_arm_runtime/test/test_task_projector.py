import numpy as np

from humanoid_arm_runtime.task_projector import TaskProjector


def test_large_delta_is_rate_limited_but_eventually_preserved() -> None:
    projector = TaskProjector(
        max_position_step_m=0.05,
        max_yaw_step_rad=0.1,
    )
    desired_position = np.array([0.12, 0.0, 0.0])

    first_position, first_yaw = projector.project(desired_position, 0.25)
    second_position, second_yaw = projector.project(desired_position, 0.25)
    final_position, final_yaw = projector.project(desired_position, 0.25)

    np.testing.assert_allclose(first_position, [0.05, 0.0, 0.0])
    np.testing.assert_allclose(second_position, [0.10, 0.0, 0.0])
    np.testing.assert_allclose(final_position, desired_position)
    assert first_yaw == 0.1
    assert second_yaw == 0.2
    assert final_yaw == 0.25


def test_reset_returns_to_zero_offset_baseline() -> None:
    projector = TaskProjector(0.05, 0.1)
    projector.project(np.array([0.04, 0.0, 0.0]), 0.08)

    projector.reset()
    position, yaw = projector.project(np.zeros(3), 0.0)

    np.testing.assert_allclose(position, np.zeros(3))
    assert yaw == 0.0


def test_rejects_invalid_configuration_and_input() -> None:
    for args in ((0.0, 0.1), (0.05, 0.0)):
        try:
            TaskProjector(*args)
        except ValueError:
            continue
        raise AssertionError(f"invalid projector configuration accepted: {args}")

    projector = TaskProjector()
    try:
        projector.project(np.array([np.nan, 0.0, 0.0]), 0.0)
    except ValueError:
        return
    raise AssertionError("non-finite task delta was accepted")
