"""ArmStateMonitor: computes joint/end-effector state plus numerically
differentiated velocity/acceleration and achieved sampling FPS from a
URArmNode. Import this in your own code to pull live arm state:

    from ur_control.ur_arm_node import URArmNode
    from ur_control.state_monitor import ArmStateMonitor

    arm = URArmNode(...)
    monitor = ArmStateMonitor(arm)
    while rclpy.ok():
        sample = monitor.sample()   # call once per loop iteration
        sample.joint_state          # JointState(names, positions) or None
        sample.ee_state             # EndEffectorState(position, rotation_vector) or None
        sample.velocity.joint       # list[float] rad/s, or None
        sample.velocity.ee_linear   # (x, y, z) m/s, or None
        sample.velocity.ee_angular  # (x, y, z) rad/s, or None
        sample.acceleration.joint / .ee_linear / .ee_angular  # same shape, /s^2
        sample.fps                  # achieved sampling rate (Hz), or None

End-effector position/rotation_vector/velocity/acceleration are all expressed
in the UR controller's "Base" frame (matching the teach pendant / URScript),
not ROS's base_link — see pose_utils.base_link_to_ur_base() for why they
differ. Joint values are always in radians.

sample() calls rclpy.spin_once() internally, so just calling it in a loop is
enough to both pump the node and get fresh data.
"""

import time
from typing import List, NamedTuple, Optional, Tuple

import rclpy

from ur_control.ur_arm_node import URArmNode
from ur_control.pose_utils import (
    quaternion_to_rotation_vector,
    quaternion_multiply,
    base_link_to_ur_base,
)

DEFAULT_SMOOTHING_ALPHA = 0.5  # EMA coefficient for velocity/acceleration/FPS (0~1; higher = less smoothing)


class JointState(NamedTuple):
    names: List[str]
    positions: List[float]  # rad


class EndEffectorState(NamedTuple):
    position: Tuple[float, float, float]         # m, UR Base frame
    rotation_vector: Tuple[float, float, float]   # rad, UR Base frame


class Velocity(NamedTuple):
    joint: Optional[List[float]]                      # rad/s
    ee_linear: Optional[Tuple[float, float, float]]    # m/s, UR Base frame
    ee_angular: Optional[Tuple[float, float, float]]   # rad/s, UR Base frame


class Acceleration(NamedTuple):
    joint: Optional[List[float]]                      # rad/s^2
    ee_linear: Optional[Tuple[float, float, float]]    # m/s^2
    ee_angular: Optional[Tuple[float, float, float]]   # rad/s^2


class ArmStateSample(NamedTuple):
    joint_state: Optional[JointState]
    ee_state: Optional[EndEffectorState]
    velocity: Velocity
    acceleration: Acceleration
    fps: Optional[float]


def _quaternion_conjugate(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def _to_ur_base_vector(v):
    """base_link 方向向量（速度/加速度）轉成教導器 Base 座標系：原點相同、
    只差繞 Z 轉 180 度，所以跟位置一樣是 x, y 反號。"""
    return (-v[0], -v[1], v[2])


class _Differentiator:
    """對相鄰兩次數值取樣做差分，回傳變化率。"""

    def __init__(self):
        self._value = None
        self._time = None

    def update(self, new_value, new_time):
        if self._value is None or new_time <= self._time:
            derivative = [0.0] * len(new_value)
        else:
            dt = new_time - self._time
            derivative = [(n - o) / dt for n, o in zip(new_value, self._value)]
        self._value = list(new_value)
        self._time = new_time
        return derivative


class _EmaVector:
    """對一個向量做指數移動平均，讓數值微分出來的雜訊穩定一點。"""

    def __init__(self, alpha):
        self._alpha = alpha
        self._value = None

    def update(self, new_value):
        if self._value is None or len(self._value) != len(new_value):
            self._value = list(new_value)
        else:
            self._value = [self._alpha * n + (1.0 - self._alpha) * o
                            for n, o in zip(new_value, self._value)]
        return self._value


class ArmStateMonitor:
    """Stateful sampler: call sample() once per loop iteration to get a fresh
    ArmStateSample. Keeps the history needed to numerically differentiate
    position -> velocity -> acceleration between calls."""

    def __init__(self, arm: URArmNode, smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA):
        self.arm = arm
        self._alpha = smoothing_alpha

        self._joint_accel_diff = _Differentiator()
        self._joint_accel_ema = _EmaVector(smoothing_alpha)

        self._linear_vel_diff = _Differentiator()
        self._linear_vel_ema = _EmaVector(smoothing_alpha)
        self._linear_accel_diff = _Differentiator()
        self._linear_accel_ema = _EmaVector(smoothing_alpha)

        self._angular_vel_ema = _EmaVector(smoothing_alpha)
        self._angular_accel_diff = _Differentiator()
        self._angular_accel_ema = _EmaVector(smoothing_alpha)

        self._previous_quat = None
        self._previous_tcp_time = None
        self._fps_ema = None
        self._previous_sample_time = None

    def sample(self) -> ArmStateSample:
        rclpy.spin_once(self.arm, timeout_sec=0.0)
        now = time.time()

        joint_state = None
        joint_velocity = None
        joint_accel = None
        positions = self.arm.peek_joint_positions()
        if positions is not None:
            joint_state = JointState(names=list(self.arm.joint_names), positions=positions)
        velocities = self.arm.peek_joint_velocities()
        if velocities is not None:
            joint_velocity = velocities
            raw_accel = self._joint_accel_diff.update(velocities, now)
            joint_accel = self._joint_accel_ema.update(raw_accel)

        ee_state = None
        ee_linear_vel = ee_angular_vel = ee_linear_accel = ee_angular_accel = None
        tcp_pose = self.arm.get_current_tcp_pose()
        if tcp_pose is not None:
            (x, y, z), quat = tcp_pose
            rx, ry, rz = quaternion_to_rotation_vector(*quat)
            base_xyz, base_rotvec = base_link_to_ur_base((x, y, z), (rx, ry, rz))
            ee_state = EndEffectorState(position=base_xyz, rotation_vector=base_rotvec)

            raw_linear_vel = self._linear_vel_diff.update([x, y, z], now)
            ee_linear_vel = tuple(self._linear_vel_ema.update(_to_ur_base_vector(raw_linear_vel)))
            raw_linear_accel = self._linear_accel_diff.update(raw_linear_vel, now)
            ee_linear_accel = tuple(self._linear_accel_ema.update(_to_ur_base_vector(raw_linear_accel)))

            if self._previous_quat is not None and now > self._previous_tcp_time:
                dt = now - self._previous_tcp_time
                relative = quaternion_multiply(quat, _quaternion_conjugate(self._previous_quat))
                wx, wy, wz = quaternion_to_rotation_vector(*relative)
                raw_angular_vel = [wx / dt, wy / dt, wz / dt]
            else:
                raw_angular_vel = [0.0, 0.0, 0.0]
            ee_angular_vel = tuple(self._angular_vel_ema.update(_to_ur_base_vector(raw_angular_vel)))
            raw_angular_accel = self._angular_accel_diff.update(raw_angular_vel, now)
            ee_angular_accel = tuple(self._angular_accel_ema.update(_to_ur_base_vector(raw_angular_accel)))

            self._previous_quat = quat
            self._previous_tcp_time = now

        if self._previous_sample_time is not None:
            loop_dt = now - self._previous_sample_time
            if loop_dt > 0:
                instant_fps = 1.0 / loop_dt
                self._fps_ema = (instant_fps if self._fps_ema is None
                                  else self._alpha * instant_fps + (1.0 - self._alpha) * self._fps_ema)
        self._previous_sample_time = now

        return ArmStateSample(
            joint_state=joint_state,
            ee_state=ee_state,
            velocity=Velocity(joint=joint_velocity, ee_linear=ee_linear_vel, ee_angular=ee_angular_vel),
            acceleration=Acceleration(joint=joint_accel, ee_linear=ee_linear_accel, ee_angular=ee_angular_accel),
            fps=self._fps_ema,
        )
