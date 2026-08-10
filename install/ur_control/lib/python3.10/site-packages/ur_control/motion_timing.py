"""Small motion-timing utility shared by control scripts that need to turn a
"how far" + "how fast/how hard" input into a trajectory duration."""

import math


def trapezoidal_duration(distance: float, max_velocity: float, max_acceleration: float) -> float:
    """Time to cover `distance` under a symmetric trapezoidal (or triangular,
    if too short to reach max_velocity) velocity profile bounded by
    max_velocity and max_acceleration."""
    distance = abs(distance)
    if distance < 1e-9:
        return 0.0
    max_velocity = max(max_velocity, 1e-6)
    max_acceleration = max(max_acceleration, 1e-6)

    accel_distance = (max_velocity * max_velocity) / max_acceleration
    if distance >= accel_distance:
        return distance / max_velocity + max_velocity / max_acceleration
    return 2.0 * math.sqrt(distance / max_acceleration)
