"""PoseTrajectoryInterpolator: a time-queryable end-effector trajectory buffer.

This is the piece that was structurally missing from the earlier
move_pose_waypoints()/move_pose_chunk() approach: instead of building one
discrete multi-point FollowJointTrajectory goal and blocking until it
finishes (or getting preempted wholesale by the next one), this class keeps
a live (timestamp, position, orientation) waypoint buffer that a high-rate
control loop queries by absolute time every tick. New model output gets
merged in with schedule_waypoint(), which truncates the buffer at the
current time (so the pose stays continuous) and stretches the target time
if reaching it would exceed a velocity limit (so it never lurches).

Equivalent in spirit to the PoseTrajectoryInterpolator used in the
diffusion_policy / UMI real-robot deployment code. Pure numpy, no ROS
dependency, so it can be unit-tested offline (see
ur_control/examples/test_trajectory_interpolator.py) and reused by both the
ROS2 (forward_position_controller) and RTDE (servoJ) control scripts.

All positions/orientations are plain (x, y, z) / (x, y, z, w) — frame is
whatever the caller uses consistently (this project uses base_link
internally; convert to/from the UR "Base" rotation-vector convention with
pose_utils.ur_base_to_base_link() / base_link_to_ur_base() at the point
where you talk to the teach pendant or accept model output in that frame).
"""

import numpy as np

from ur_control.pose_utils import quaternion_multiply, quaternion_conjugate, quaternion_to_rotation_vector


def _slerp(q0, q1, alpha):
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def _quaternion_angle(q0, q1):
    """Shortest rotation angle (rad) between two quaternions (xyzw)."""
    dot = abs(np.dot(q0, q1))
    dot = min(dot, 1.0)
    return 2.0 * np.arccos(dot)


class PoseTrajectoryInterpolator:
    """Time-indexed buffer of (timestamp, position, orientation) waypoints.

    Position interpolation is linear; orientation is SLERP. All timestamps
    are absolute (same clock as whatever you pass to interpolate()/
    schedule_waypoint() — this project uses time.time() throughout)."""

    def __init__(self, time0: float, position0, orientation0):
        self._times = [float(time0)]
        self._positions = [np.array(position0, dtype=float)]
        self._orientations = [np.array(orientation0, dtype=float)]

    def __len__(self):
        return len(self._times)

    @property
    def times(self):
        return list(self._times)

    def start_time(self) -> float:
        return self._times[0]

    def end_time(self) -> float:
        return self._times[-1]

    def end_pose(self):
        return self._positions[-1].copy(), self._orientations[-1].copy()

    def trim(self, t: float):
        """Collapse the whole buffer down to a single anchor point: the
        interpolated pose AT t. Discards both the past (already executed)
        AND whatever future the old plan still had queued up — a new chunk
        must fully replace the old one's future, not get appended after its
        leftover tail. This is what keeps the trajectory continuous (same
        pose at the splice) while still guaranteeing the old plan has no
        lingering influence past `t`."""
        if t <= self._times[0]:
            return
        position, orientation = self.interpolate(t)
        self._times = [t]
        self._positions = [position]
        self._orientations = [orientation]

    def schedule_waypoint(self, position, orientation, target_time: float, curr_time: float,
                           max_pos_speed: float = None, max_rot_speed: float = None):
        """Insert a new target, truncating the buffer at curr_time first (so
        the new segment starts from wherever the trajectory actually is
        *right now*, not from some stale future point a previous chunk
        promised). If reaching `position`/`orientation` by `target_time`
        would exceed max_pos_speed (m/s) or max_rot_speed (rad/s), the
        segment's duration is stretched (target time pushed later) instead
        of just letting the interpolator imply an over-speed motion."""
        self.trim(curr_time)
        start_position, start_orientation = self.end_pose()
        start_time = self.end_time()

        position = np.array(position, dtype=float)
        orientation = np.array(orientation, dtype=float)

        duration = max(target_time - start_time, 1e-6)
        if max_pos_speed is not None and max_pos_speed > 0:
            distance = float(np.linalg.norm(position - start_position))
            duration = max(duration, distance / max_pos_speed)
        if max_rot_speed is not None and max_rot_speed > 0:
            angle = _quaternion_angle(start_orientation, orientation)
            duration = max(duration, angle / max_rot_speed)

        self._times.append(start_time + duration)
        self._positions.append(position)
        self._orientations.append(orientation)

    def interpolate(self, t: float):
        """(position, orientation) at absolute time t, clamped to the first/
        last waypoint if t is outside the buffered range."""
        if t <= self._times[0]:
            return self._positions[0].copy(), self._orientations[0].copy()
        if t >= self._times[-1]:
            return self._positions[-1].copy(), self._orientations[-1].copy()
        idx = 0
        while self._times[idx + 1] < t:
            idx += 1
        t0, t1 = self._times[idx], self._times[idx + 1]
        alpha = (t - t0) / max(t1 - t0, 1e-9)
        position = self._positions[idx] + alpha * (self._positions[idx + 1] - self._positions[idx])
        orientation = _slerp(self._orientations[idx], self._orientations[idx + 1], alpha)
        return position, orientation

    def interpolate_velocity(self, t: float):
        """(linear_velocity, angular_velocity) at t — the constant velocity
        of whichever segment contains t (linear segments have constant
        velocity by construction). Angular velocity is the relative-rotation
        rotation-vector between the segment's endpoints divided by its
        duration, i.e. the same finite-difference feedforward the task brief
        asked for, just precomputed per segment instead of live-differenced."""
        if len(self._times) < 2:
            return np.zeros(3), np.zeros(3)
        if t <= self._times[0]:
            idx = 0
        elif t >= self._times[-1]:
            idx = len(self._times) - 2
        else:
            idx = 0
            while self._times[idx + 1] < t:
                idx += 1
        t0, t1 = self._times[idx], self._times[idx + 1]
        dt = max(t1 - t0, 1e-9)
        linear = (self._positions[idx + 1] - self._positions[idx]) / dt
        relative = quaternion_multiply(
            tuple(self._orientations[idx + 1]), quaternion_conjugate(tuple(self._orientations[idx])))
        rot_vec = np.array(quaternion_to_rotation_vector(*relative))
        angular = rot_vec / dt
        return linear, angular


class JointTrajectoryInterpolator:
    """Same idea as PoseTrajectoryInterpolator, but for a plain N-dimensional
    joint vector: linear interpolation only, no orientation/SLERP concept.

    This is what the high-rate streaming loop actually queries. IK is only
    solved once per chunk point (a ROS service round-trip / RTDE call, tens
    of ms — fine at ~1 Hz replan rate, far too slow at CONTROL_HZ); once you
    have (timestamp, joint_positions) pairs, interpolating between them at
    500 Hz is cheap and doesn't need MoveIt/RTDE again at all."""

    def __init__(self, time0: float, positions0):
        self._times = [float(time0)]
        self._positions = [np.array(positions0, dtype=float)]

    def __len__(self):
        return len(self._times)

    @property
    def times(self):
        return list(self._times)

    def start_time(self) -> float:
        return self._times[0]

    def end_time(self) -> float:
        return self._times[-1]

    def end_positions(self):
        return self._positions[-1].copy()

    def trim(self, t: float):
        """See PoseTrajectoryInterpolator.trim(): same collapse-to-one-anchor
        semantics, for the same reason (a new chunk must fully replace the
        old plan's future, not get appended after its leftover tail)."""
        if t <= self._times[0]:
            return
        positions = self.interpolate(t)
        self._times = [t]
        self._positions = [positions]

    def schedule_waypoint(self, positions, target_time: float, curr_time: float,
                           max_joint_speed: float = None):
        self.trim(curr_time)
        start_positions = self.end_positions()
        start_time = self.end_time()

        positions = np.array(positions, dtype=float)
        duration = max(target_time - start_time, 1e-6)
        if max_joint_speed is not None and max_joint_speed > 0:
            max_delta = float(np.max(np.abs(positions - start_positions)))
            duration = max(duration, max_delta / max_joint_speed)

        self._times.append(start_time + duration)
        self._positions.append(positions)

    def interpolate(self, t: float):
        if t <= self._times[0]:
            return self._positions[0].copy()
        if t >= self._times[-1]:
            return self._positions[-1].copy()
        idx = 0
        while self._times[idx + 1] < t:
            idx += 1
        t0, t1 = self._times[idx], self._times[idx + 1]
        alpha = (t - t0) / max(t1 - t0, 1e-9)
        return self._positions[idx] + alpha * (self._positions[idx + 1] - self._positions[idx])
