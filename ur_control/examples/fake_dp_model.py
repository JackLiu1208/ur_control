"""Fake Diffusion Policy inference: stands in for a real model so the control
loop / interpolator plumbing can be tested without a trained policy.

Shared by dp_control_ros2.py, dp_control_rtde.py, and
test_trajectory_interpolator.py so all three exercise the exact same chunk
shape — swapping this out for a real model later shouldn't require touching
any of them, just this function.
"""

import math

import numpy as np


def fake_dp_inference(t_obs: float, start_position, orientation, radius: float,
                       angular_velocity: float, action_step_dt: float, prediction_horizon: int):
    """Simulate one policy inference at observation time t_obs: returns the
    next `prediction_horizon` predicted (timestamp, position, orientation)
    steps, tracing a circle of the given radius through start_position at
    t=0, base_link frame (position/orientation stay a plain tuple/np.array,
    no ROS types here).

    start_position is a point ON the circle (matches "以當前位置為起點"): the
    circle's center is placed radius meters away along -X so that angle=0
    lands exactly back on start_position.

    Returns: (timestamps: list[float], positions: list[np.ndarray(3)],
              orientations: list[np.ndarray(4)])
    """
    center = np.array(start_position, dtype=float) - np.array([radius, 0.0, 0.0])
    orientation = np.array(orientation, dtype=float)

    timestamps, positions, orientations = [], [], []
    for i in range(1, prediction_horizon + 1):
        t = t_obs + i * action_step_dt
        angle = angular_velocity * t
        position = center + np.array([radius * math.cos(angle), radius * math.sin(angle), 0.0])
        timestamps.append(t)
        positions.append(position)
        orientations.append(orientation.copy())
    return timestamps, positions, orientations
