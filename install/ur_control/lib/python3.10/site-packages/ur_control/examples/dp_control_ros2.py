#!/usr/bin/env python3
"""`ur_control.dp_controller.DPController` API 的示範程式：四種假軌跡（`TRAJECTORY_TYPE`
切換）跑一輪 receding horizon 高頻串流控制，示範怎麼把一個模型包成
`DPController.run()` 要的 `inference_fn` 介面（換成真模型只要換掉對應 class 的
`__call__()` 裡面那段推論邏輯）。控制邏輯本身都在 `dp_controller.py`，這支程式不
含任何控制邏輯。

四種軌跡的用途、啟動前置作業、`inference_fn` 座標系要求，見 README「程式清單」
跟「API」。
"""

import numpy as np
import rclpy

from ur_control.dp_controller import DPControlConfig, DPController
from ur_control.pose_utils import quaternion_multiply, rotation_vector_to_quaternion
from ur_control.ur_arm_node import TCP_OFFSET_ROBOTIQ_2F140

# =============================================================================
# 選要測哪一種軌跡
# =============================================================================
TRAJECTORY_TYPE = "figure_eight"   
# "six_axis_circle" / "figure_eight" / "step_waypoints" / "line_back_and_forth"

# 目前手臂上裝的是 25cm 的 Robotiq 2F-140 夾爪；如果拆掉夾爪、只用裸法蘭面測試，
# 改成 `from ur_control.ur_arm_node import TCP_OFFSET_BARE_FLANGE` 再指派給這個
# 變數。這個值只影響座標轉換（IK/FK 本身不知道 TCP offset 這件事），不會自動
# 偵測你實際裝了什麼工具，換工具一定要手動改這裡。
TCP_OFFSET_XYZ = TCP_OFFSET_ROBOTIQ_2F140


class _SixAxisCoupledCircleInference:
    """XY 圓 + Z 正弦 + RX/RY/RZ 擺動，四軸用不同頻率/相位、刻意不同步（見 README
    「程式清單」說明測試用意）。

    在第一次呼叫（t_obs 最小，也就是 run() 剛開始那一刻）把當下的
    current_tcp_pose 存下來當固定參考點，之後每次都用這個存下來的值，不管
    current_tcp_pose 之後怎麼變。真模型如果是逐次根據當下狀態推論（沒有「固定
    參考點」這種概念），可以每次都直接用傳進來的 current_tcp_pose，不需要這種
    快取——下面四個 class 都是同樣的寫法，之後接真模型時這段可以整個拿掉。"""

    def __init__(self, radius_m: float, xy_period_s: float,
                 z_amplitude_m: float, z_period_s: float,
                 rot_amplitude_deg: float, rx_period_s: float,
                 ry_period_s: float, rz_period_s: float,
                 action_step_dt: float, prediction_horizon: int):
        self._origin = None
        self.radius_m = radius_m
        self.xy_angular_velocity = 2.0 * np.pi / xy_period_s
        self.z_amplitude_m = z_amplitude_m
        self.z_angular_velocity = 2.0 * np.pi / z_period_s
        self.rot_amplitude_rad = np.radians(rot_amplitude_deg)
        self.rx_angular_velocity = 2.0 * np.pi / rx_period_s
        self.ry_angular_velocity = 2.0 * np.pi / ry_period_s
        self.rz_angular_velocity = 2.0 * np.pi / rz_period_s
        self.action_step_dt = action_step_dt
        self.prediction_horizon = prediction_horizon

    def __call__(self, t_obs, current_joint_positions, current_tcp_pose):
        if self._origin is None:
            self._origin = current_tcp_pose
        origin_position, origin_orientation = self._origin
        # 圓心放在起點 -X 方向 radius_m 處，這樣 angle=0 剛好回到起點（跟舊版
        # fake_dp_inference() 的幾何一致）。
        center = np.array(origin_position, dtype=float) - np.array([self.radius_m, 0.0, 0.0])

        timestamps, positions, orientations = [], [], []
        for i in range(1, self.prediction_horizon + 1):
            t = t_obs + i * self.action_step_dt
            xy_angle = self.xy_angular_velocity * t
            position = center + np.array([
                self.radius_m * np.cos(xy_angle),
                self.radius_m * np.sin(xy_angle),
                self.z_amplitude_m * np.sin(self.z_angular_velocity * t),
            ])

            # 三個軸用不同頻率 + 互相錯開 120 度相位，避免同步擺動（同步的話
            # 看起來只是單軸旋轉，測不出真正的軸間耦合）。
            rx = self.rot_amplitude_rad * np.sin(self.rx_angular_velocity * t)
            ry = self.rot_amplitude_rad * np.sin(self.ry_angular_velocity * t + 2.0 * np.pi / 3.0)
            rz = self.rot_amplitude_rad * np.sin(self.rz_angular_velocity * t + 4.0 * np.pi / 3.0)
            delta_quat = rotation_vector_to_quaternion(rx, ry, rz)
            # 世界座標（base_link）疊加，不是工具自己的局部座標——先到 origin
            # 姿態，再疊加 delta，順序: quaternion_multiply(delta, origin)。
            orientation = quaternion_multiply(delta_quat, tuple(origin_orientation))

            timestamps.append(t)
            positions.append(position)
            orientations.append(np.array(orientation))
        return timestamps, positions, orientations


class _FigureEightInference:
    """XY 平面 8 字形（Lissajous：x=sin(w t), y=sin(2w t)），曲率連續變化、中間
    有交叉點。姿態固定不變——曲率變化是這個軌跡要單獨測的變數，姿態變化留給
    six_axis_circle 測。"""

    def __init__(self, amplitude_x_m: float, amplitude_y_m: float, period_s: float,
                 action_step_dt: float, prediction_horizon: int):
        self._origin = None
        self.amplitude_x_m = amplitude_x_m
        self.amplitude_y_m = amplitude_y_m
        self.angular_velocity = 2.0 * np.pi / period_s
        self.action_step_dt = action_step_dt
        self.prediction_horizon = prediction_horizon

    def __call__(self, t_obs, current_joint_positions, current_tcp_pose):
        if self._origin is None:
            self._origin = current_tcp_pose
        origin_position, origin_orientation = self._origin
        origin_position = np.array(origin_position, dtype=float)

        timestamps, positions, orientations = [], [], []
        for i in range(1, self.prediction_horizon + 1):
            t = t_obs + i * self.action_step_dt
            angle = self.angular_velocity * t
            offset = np.array([
                self.amplitude_x_m * np.sin(angle),
                self.amplitude_y_m * np.sin(2.0 * angle),
                0.0,
            ])
            timestamps.append(t)
            positions.append(origin_position + offset)
            orientations.append(np.array(origin_orientation, dtype=float))
        return timestamps, positions, orientations


class _StepWaypointInference:
    """在起點附近幾個固定 waypoint 之間跳動：同一個 chunk 內全部 T_p 步都指向
    同一個目標，每隔 replans_per_waypoint 次 replan 才跳下一個目標，目標本身
    不連續（見 README「程式清單」說明測試用意）。"""

    def __init__(self, waypoint_offsets_m, replans_per_waypoint: int,
                 action_step_dt: float, prediction_horizon: int):
        self._origin = None
        self.waypoint_offsets_m = [np.array(o, dtype=float) for o in waypoint_offsets_m]
        self.replans_per_waypoint = replans_per_waypoint
        self.action_step_dt = action_step_dt
        self.prediction_horizon = prediction_horizon
        self._replan_count = 0

    def __call__(self, t_obs, current_joint_positions, current_tcp_pose):
        if self._origin is None:
            self._origin = current_tcp_pose
        origin_position, origin_orientation = self._origin
        origin_position = np.array(origin_position, dtype=float)

        waypoint_index = (self._replan_count // self.replans_per_waypoint) % len(self.waypoint_offsets_m)
        self._replan_count += 1
        target_position = origin_position + self.waypoint_offsets_m[waypoint_index]

        timestamps, positions, orientations = [], [], []
        for i in range(1, self.prediction_horizon + 1):
            t = t_obs + i * self.action_step_dt
            timestamps.append(t)
            positions.append(target_position.copy())
            orientations.append(np.array(origin_orientation, dtype=float))
        return timestamps, positions, orientations


class _LineBackAndForthInference:
    """沿 X 軸來回直線移動，三角波（不是正弦）：等速度、端點瞬間反向（見 README
    「程式清單」說明測試用意）。t=0 時位移為 0，避免一開始就有跳變。"""

    def __init__(self, amplitude_m: float, period_s: float,
                 action_step_dt: float, prediction_horizon: int):
        self._origin = None
        self.amplitude_m = amplitude_m
        self.period_s = period_s
        self.action_step_dt = action_step_dt
        self.prediction_horizon = prediction_horizon

    def _triangle_wave(self, t: float) -> float:
        """0 -> +1 -> 0 -> -1 -> 0，週期 period_s，t=0 時為 0。"""
        x = (t / self.period_s) % 1.0
        if x < 0.25:
            return 4.0 * x
        elif x < 0.75:
            return 2.0 - 4.0 * x
        return -4.0 + 4.0 * x

    def __call__(self, t_obs, current_joint_positions, current_tcp_pose):
        if self._origin is None:
            self._origin = current_tcp_pose
        origin_position, origin_orientation = self._origin
        origin_position = np.array(origin_position, dtype=float)

        timestamps, positions, orientations = [], [], []
        for i in range(1, self.prediction_horizon + 1):
            t = t_obs + i * self.action_step_dt
            offset = np.array([self.amplitude_m * self._triangle_wave(t), 0.0, 0.0])
            timestamps.append(t)
            positions.append(origin_position + offset)
            orientations.append(np.array(origin_orientation, dtype=float))
        return timestamps, positions, orientations


def _build_inference_fn(trajectory_type: str, action_step_dt: float, prediction_horizon: int):
    if trajectory_type == "six_axis_circle":
        return _SixAxisCoupledCircleInference(
            radius_m=0.05, xy_period_s=5,
            z_amplitude_m=0.05, z_period_s=10,
            rot_amplitude_deg=6.0, rx_period_s=10, ry_period_s=10, rz_period_s=10,
            action_step_dt=action_step_dt, prediction_horizon=prediction_horizon)
    if trajectory_type == "figure_eight":
        return _FigureEightInference(
            amplitude_x_m=0.06, amplitude_y_m=0.06, period_s=5.0,
            action_step_dt=action_step_dt, prediction_horizon=prediction_horizon)
    if trajectory_type == "step_waypoints":
        return _StepWaypointInference(
            waypoint_offsets_m=[
                (0.04, 0.04, 0.0), (-0.04, 0.04, 0.0),
                (-0.04, -0.04, 0.0), (0.04, -0.04, 0.0),
            ],
            replans_per_waypoint=2,
            action_step_dt=action_step_dt, prediction_horizon=prediction_horizon)
    if trajectory_type == "line_back_and_forth":
        return _LineBackAndForthInference(
            amplitude_m=0.05, period_s=3.0,
            action_step_dt=action_step_dt, prediction_horizon=prediction_horizon)
    raise ValueError(f"未知的 TRAJECTORY_TYPE: {trajectory_type!r}")


def main(args=None):
    config = DPControlConfig(
        node_name="dp_control_ros2_demo",

        tcp_offset_xyz=TCP_OFFSET_XYZ,

        # 這個工作站量測過的安全 home pose——通用 UR「手肘向上」姿態，換工作空間
        # 要自己重新確認不會撞到任何東西（見 dp_controller.py 裡的說明）。
        home_joint_positions=[-1.5708, -1.0708, -2.1708, -1.4708, 1.5708, 0.0],
        home_move_time_seconds=4.0,

        control_hz=120.0,
        prediction_horizon=16,
        action_horizon=8,
        action_step_dt=0.1,

        max_pos_speed=1.5,     # 這台手臂設定上限 1.5m/s，這裡保守留一點餘裕
        max_rot_speed=3.33,     # 上限 3.33rad/s (191deg/s)

        # "joint"（預設、已驗證）或 "cartesian"（逐 tick 現場解 IK，見 README）。
        control_space="cartesian",

        # cartesian 模式必填的關節角速度/加速度上限，安全關鍵。以下數字**還沒有
        # 像 max_pos_speed/max_rot_speed 那樣實測確認過**，用之前先核對教導器
        # Installation 設定/payload 限制。
        max_joint_speed=6.2832,          # rad/s (180deg/s)，未實測確認
        max_joint_acceleration=6.0,      # rad/s^2，未實測確認

        run_duration_seconds=30.0,   # 示範用安全上限，跑這麼久自動停止（Ctrl+C 也可隨時中止）

        # 在真實 UR5e 上量出來的延遲前饋補償，見 dp_controller.py／README。換平台
        # 要重新用 analyze_trajectory.py 的對齊掃描量測。
        latency_feedforward_seconds=0.05,

        enable_logging=True,
        experiment_name=f"dp_control_ros2_demo_{TRAJECTORY_TYPE}",
    )

    inference_fn = _build_inference_fn(
        TRAJECTORY_TYPE, action_step_dt=config.action_step_dt,
        prediction_horizon=config.prediction_horizon)

    rclpy.init(args=args)
    controller = DPController(config)
    try:
        controller.run(inference_fn)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
