"""可重用的 Diffusion Policy（receding horizon action chunking）高頻串流控制 API。

`DPController.run()` 把一個 `inference_fn`（模型推論）的 chunk 輸出，變成平順的
高頻關節位置指令串流。完整說明、使用範例、IK 機制、`control_space` 兩種模式的
細節，見 README「API：把你的模型接上去」跟「問題與解決方法」。

## `inference_fn` 的合約

```
inference_fn(t_obs, current_joint_positions, current_tcp_pose) -> (timestamps, positions, orientations)
```

輸入：`t_obs`（這次 replan 的時刻，相對 `run()` 開始執行的秒數）、
`current_joint_positions`（目前 6 個關節角 rad）、`current_tcp_pose`
（目前 TCP 在 `base_link` 座標系的 `(position_xyz, orientation_xyzw)`）。

輸出三個等長 list（長度 = `config.prediction_horizon`）：`timestamps`（每步的
**絕對**目標時間）、`positions`（**base_link** 座標系的 `(x,y,z)`，不是教導器
Base 座標系——需要轉換用 `pose_utils.ur_base_to_base_link()`）、`orientations`
（`base_link` 座標系的四元數 `(x,y,z,w)`）。

`inference_fn` 只在每次 replan（每 `action_horizon * action_step_dt` 秒）被呼叫
一次，不是每個控制 tick——呼叫頻率跟串流平順度都是 `DPController` 的責任。
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import ColorRGBA, Float64MultiArray
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from controller_manager_msgs.srv import SwitchController

from ur_control import analytic_ik
from ur_control.ur_arm_node import URArmNode
from ur_control.trajectory_interpolator import PoseTrajectoryInterpolator, JointTrajectoryInterpolator
from ur_control.trajectory_logger import TrajectoryLogger

Position = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
TcpPose = Tuple[Position, Quaternion]
InferenceFn = Callable[[float, List[float], TcpPose], Tuple[List[float], List[Position], List[Quaternion]]]


@dataclass
class DPControlConfig:
    """`DPController` 的所有可調參數。預設值大多沿用 `dp_control_ros2.py` 在真實
    UR5e 上驗證過的數字（見 README）——`home_joint_positions` 是唯一刻意不給預設值
    的欄位：那是特定工作站量測過的安全姿態，換一個機器人/工作空間直接沿用可能會
    撞到東西，必須自己重新確認、明確填進來。"""

    node_name: str = "dp_control_node"

    streaming_controller_name: str = "forward_position_controller"
    default_controller_name: str = "scaled_joint_trajectory_controller"

    # 執行 run() 之前會先移動到這裡，確保每次都有一致的起點。刻意沒有預設值——
    # 這是安全關鍵參數，必須由呼叫端針對自己的工作空間明確給值並自行確認不會
    # 撞到任何東西。設 enable_homing=False 可以跳過這一步（呼叫端自己保證起始
    # 姿態安全）。
    home_joint_positions: Optional[List[float]] = None
    home_move_time_seconds: float = 4.0
    enable_homing: bool = True

    control_hz: float = 120.0

    prediction_horizon: int = 16     # T_p：inference_fn 每次要回傳未來幾步
    action_horizon: int = 8          # T_a：執行到第幾步才呼叫下一次 inference_fn
    action_step_dt: float = 0.1      # chunk 裡每一步間隔幾秒

    max_pos_speed: float = 1.0       # m/s，interpolator 排新 waypoint 時的自動限速上限
    max_rot_speed: float = 3.0       # rad/s

    # 控制迴圈在哪個空間做 120Hz 插值，"joint"（預設，已驗證）或 "cartesian"
    # （每 tick 現場解 IK）。細節見 README「問題與解決方法」。
    control_space: str = "cartesian"     # "joint" / "cartesian"

    # 兩種 control_space 都會用到的關節角速度上限。joint 模式用來夾
    # joint_interp 排點的時間（IK 在奇異點附近可能讓「Cartesian 移動很小」對應
    # 到「關節角要轉很大」，只靠 max_pos_speed/max_rot_speed 管不到這個）；也是
    # 下面 joint_accel_limit 平滑濾波器的速度上限。安全關鍵，**不提供預設值**
    # （理由同 home_joint_positions）：必須填你在真實 UR5e 上確認過的數字（示
    # 教器 Installation 找不到就用保守值，不要瞎猜），沒填會直接 raise。細節見
    # README。
    max_joint_speed: Optional[float] = None          # rad/s，六軸統一上限

    # 不管 control_space 是哪一種，run() 最後實際發布出去的指令都會經過同一個
    # 加速度限制的平滑濾波器（見 run() 內的實作），把 joint_interp 在 chunk
    # 交界處的斜率不連續、cartesian 模式現場解 IK 在奇異點附近的大位移，都收斂
    # 成速度連續變化的指令——UR 控制器內建一套跟教導器 Installation 設定無關、
    # 不能調、卻一直在監測「指令變化是否過於突然」的低階保護機制，指令加速度太
    # 大一樣會觸發 Protective Stop，不是只有速度超過 Installation 上限才會。
    #
    # 這個值**不是**教導器上能查到的任何真實規格（找不到才需要這個欄位），是
    # 純軟體端自選的保守平滑常數，用來壓低「指令」本身的加速度，不代表手臂
    # 實際動力學上限。安全關鍵，**不提供預設值**：先給保守值（例如 2 rad/s²）
    # 實測，觀察還會不會跳 Protective Stop，穩定後再視情況調高。
    joint_accel_limit: Optional[float] = None        # rad/s^2，六軸統一上限

    # 這台手臂出廠校正檔（教導器 Installation -> Calibration 匯出的格式，見
    # analytic_ik.py「出廠校正」小節）。None（預設）= analytic_ik 用內建的標稱
    # UR5e 參數，不做校正修正。填了路徑，run() 會在進入串流迴圈前呼叫
    # analytic_ik.load_calibration()，之後這個 process 裡所有 IK/FK（包含
    # get_current_tcp_pose_analytic() 等）都會改用校正後的模型。跟其他安全關鍵
    # 欄位不同：這個沒有校正檔也能正常運作（只是少了 ~1mm 等級的精度），所以
    # 允許 None 預設值。
    calibration_file: Optional[Path] = None

    # 在真實 UR5e 上量出來的延遲前饋補償（見 README「問題與解決方法」第 3 項）：
    # 控制迴圈查詢 interpolator 用 now + latency_feedforward，而不是 now，補償
    # 「規劃到指令真正生效」之間的固定延遲。換平台/換 controller 要重新用
    # analyze_trajectory.py 的對齊掃描量測，不能沿用這個數字。
    latency_feedforward_seconds: float = 0.05
    latency_feedforward_min_seconds: float = 0.0
    latency_feedforward_max_seconds: float = 0.15

    run_duration_seconds: Optional[float] = None   # None = 一直跑到 Ctrl+C；測試時可以設安全上限

    fps_smoothing_alpha: float = 0.1
    status_log_interval_seconds: float = 1.0

    # TCP（工具中心點）相對法蘭面（tool0）的偏移，只有平移。None = 沿用
    # URArmNode 自己的預設值（目前是裝了 25cm Robotiq 2F-140 夾爪的設定，見
    # ur_arm_node.py 的 DEFAULT_TCP_OFFSET_XYZ）。換工具（例如拆掉夾爪、換成
    # 別的工具）就在這裡明確填 ur_arm_node.TCP_OFFSET_BARE_FLANGE 或自己量的值
    # ——這個值只影響座標轉換，不會自動知道你實際裝了什麼工具。
    #
    # 只有 DPController 自己建立 URArmNode 時才會用到這個值；如果建構
    # DPController 時傳入了現成的 arm，這裡會被忽略——那個 arm 自己的
    # tcp_offset_xyz 才算數。
    tcp_offset_xyz: Optional[Tuple[float, float, float]] = None

    enable_markers: bool = True
    marker_topic: str = "/dp_trajectory_markers"
    planned_marker_namespace: str = "dp_planned_chunk"
    executed_marker_namespace: str = "dp_executed_trace"
    # 藍色，量測到的真實 TCP 軌跡（跟 get_current_tcp_pose_analytic() /
    # measured.csv 同一個來源，只是這裡直接畫出來，不用等事後跑
    # analyze_trajectory.py 才看得到）。跟 executed_marker_namespace（綠色，
    # pose_interp 插值出的「理論上應該在哪」）不是同一份資料，兩條線分開畫
    # 才看得出「命令軌跡」跟「真實軌跡」到底有沒有對上。
    measured_marker_namespace: str = "dp_measured_trace"
    trace_publish_interval_seconds: float = 0.1

    enable_logging: bool = True
    experiment_name: str = "dp_control_run"
    data_dir: Optional[Path] = None   # None -> 執行時解析成 Path.cwd() / "data"


def _line_strip_marker(frame_id: str, ns: str, marker_id: int, points, color: ColorRGBA,
                        line_width_m: float = 0.003) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = line_width_m
    marker.color = color
    marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]
    return marker


def _switch_controllers(node: Node, activate, deactivate) -> bool:
    client = node.create_client(SwitchController, "/controller_manager/switch_controller")
    if not client.wait_for_service(timeout_sec=10.0):
        node.get_logger().error("等不到 /controller_manager/switch_controller service")
        return False
    request = SwitchController.Request()
    request.activate_controllers = list(activate)
    request.deactivate_controllers = list(deactivate)
    request.strictness = SwitchController.Request.BEST_EFFORT
    request.activate_asap = True
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    result = future.result()
    ok = result is not None and result.ok
    node.get_logger().info(
        f"switch_controller activate={activate} deactivate={deactivate} -> {'ok' if ok else '失敗'}")
    return ok


class DPController:
    """把 inference_fn 的 chunk 輸出變成平順的高頻關節位置指令串流，送給
    `forward_position_controller` 追蹤。一次 `run()` 呼叫是一個完整的
    回家→串流→回家 lifecycle（阻塞，直到 `run_duration_seconds` 到、或
    Ctrl+C、或 inference_fn 拋例外）。

    不會呼叫 `rclpy.init()`/`rclpy.shutdown()`——這是呼叫端的責任（可能還有別的
    node 在同一個 process 裡）。如果建構時沒有傳入 `arm`，`run()` 結束時才會
    `destroy_node()` 自己建立的那個；如果 `arm` 是呼叫端傳進來的，`DPController`
    不會動它的生命週期。"""

    def __init__(self, config: Optional[DPControlConfig] = None, arm: Optional[URArmNode] = None):
        self.config = config or DPControlConfig()
        self._owns_node = arm is None
        if arm is not None:
            self.arm = arm
        else:
            node_kwargs = {"node_name": self.config.node_name}
            if self.config.tcp_offset_xyz is not None:
                node_kwargs["tcp_offset_xyz"] = self.config.tcp_offset_xyz
            self.arm = URArmNode(**node_kwargs)

    def run(self, inference_fn: InferenceFn, wait_for_enter: bool = True) -> None:
        """阻塞執行。`wait_for_enter=True` 時，切完 controller 之後會先等使用者
        按 Enter 才開始串流（手動測試用的安全閘門）；包進更大的系統裡通常會想
        設 `False`。"""
        cfg = self.config
        arm = self.arm

        if cfg.enable_homing and cfg.home_joint_positions is None:
            raise ValueError(
                "enable_homing=True 但 home_joint_positions 沒有設定——這是安全關鍵"
                "參數，必須針對你的工作空間明確給值並自行確認不會撞到任何東西"
                "（或是把 enable_homing 設 False，自己保證起始姿態安全）。")
        if cfg.control_space not in ("joint", "cartesian"):
            raise ValueError(f"control_space 必須是 'joint' 或 'cartesian'，收到: {cfg.control_space!r}")
        if cfg.max_joint_speed is None or cfg.joint_accel_limit is None:
            raise ValueError(
                "max_joint_speed 跟 joint_accel_limit 都需要明確填值（joint/cartesian 兩種"
                "control_space 都要）——這是安全關鍵參數，沒有預設值可以沿用（理由見"
                "DPControlConfig 這兩個欄位的註解）。")

        if cfg.calibration_file is not None:
            analytic_ik.load_calibration(cfg.calibration_file)
            arm.get_logger().info(f"已載入出廠校正檔: {cfg.calibration_file}")

        command_publisher = arm.create_publisher(
            Float64MultiArray, f"/{cfg.streaming_controller_name}/commands", 10)
        marker_publisher = (arm.create_publisher(Marker, cfg.marker_topic, 10)
                             if cfg.enable_markers else None)
        logger = TrajectoryLogger() if cfg.enable_logging else None

        executor = None
        executor_thread = None
        executor_stop_event = None

        try:
            if cfg.enable_homing:
                arm.get_logger().info("回到 home pose...")
                if not arm.move_joint(cfg.home_joint_positions,
                                       time_from_start=cfg.home_move_time_seconds, wait=True):
                    arm.get_logger().error("回 home pose 失敗，中止")
                    return

            joint_positions = arm.get_current_joint_positions()
            if joint_positions is None:
                arm.get_logger().error("讀不到目前關節狀態，中止")
                return
            # 特意不用 wait_for_tcp_pose()（查 TF）：TF buffer 只要裡面有任何一筆
            # transform 就會立刻回傳，不保證是剛剛回完 home pose「之後」的新值——
            # 可能是上一次啟動、甚至更早留下的舊姿態。用跟 joint_positions 同一份
            # /joint_states 算出來的 analytic FK，才能保證 pose_interp 的起點跟
            # inference_fn 拿到的 current_tcp_pose、joint_interp 的起點三者一致，
            # 不會在第一個 replan 就因為起點對不上而跳變。
            origin_pose = arm.get_current_tcp_pose_analytic()
            if origin_pose is None:
                arm.get_logger().error("讀不到目前末端點狀態，中止")
                return
            origin_position, origin_orientation = origin_pose
            arm.get_logger().info(f"起始位置 [base_link]: {origin_position}")

            if not _switch_controllers(arm, [cfg.streaming_controller_name],
                                        [cfg.default_controller_name]):
                arm.get_logger().error("切換到串流用 controller 失敗，中止")
                return

            if wait_for_enter:
                input("按下 Enter 開始串流控制...")

            # 背景 executor：見 examples/dp_control_ros2.py 開頭的完整說明。重點：
            # 只在串流迴圈這段期間存在，回 home/切換 controller 這些阻塞方法用的
            # 是全域 executor，兩者同時 spin 同一個 node 會死鎖。
            executor = SingleThreadedExecutor()
            executor.add_node(arm)
            executor_stop_event = threading.Event()

            def _spin_executor():
                while not executor_stop_event.is_set() and rclpy.ok():
                    executor.spin_once(timeout_sec=0.1)

            executor_thread = threading.Thread(target=_spin_executor, daemon=True)
            executor_thread.start()

            start_time = time.time()
            pose_interp = PoseTrajectoryInterpolator(0.0, origin_position, origin_orientation)
            joint_interp = JointTrajectoryInterpolator(0.0, joint_positions)
            last_joint_solution = list(joint_positions)   # 只用來 seed 每次 replan 的 IK 連續性
            last_published_positions = list(joint_positions)   # 平滑濾波器狀態：實際發布過的位置
            last_published_velocity = [0.0] * len(joint_positions)   # 平滑濾波器狀態：對應速度

            next_replan_time = 0.0
            replan_index = 0
            fps_ema = None
            min_instant_fps = None
            loop_previous_time = time.time()
            last_status_log = 0.0
            last_status_joint_state_count = arm.joint_state_message_count
            last_trace_publish = 0.0
            executed_trace_points = [origin_position]
            measured_trace_points = [origin_position]
            period = 1.0 / cfg.control_hz
            next_tick_time = time.time()
            cartesian_ik_failures_since_log = 0   # 只有 control_space="cartesian" 會用到
            joint_limit_active = False          # 目前這個 tick 是否正被平滑濾波器夾住
            joint_limit_triggers_since_log = 0

            latency_feedforward = min(
                max(cfg.latency_feedforward_seconds, cfg.latency_feedforward_min_seconds),
                cfg.latency_feedforward_max_seconds)

            while rclpy.ok():
                loop_start = time.time()
                now = loop_start - start_time

                if now >= next_replan_time:
                    current_tcp_pose = arm.get_current_tcp_pose_analytic() or origin_pose
                    timestamps, positions, orientations = inference_fn(
                        now, list(last_joint_solution), current_tcp_pose)

                    for target_time, position, orientation in zip(timestamps, positions, orientations):
                        pose_interp.schedule_waypoint(
                            position, orientation, target_time=target_time, curr_time=now,
                            max_pos_speed=cfg.max_pos_speed, max_rot_speed=cfg.max_rot_speed)
                        actual_time = pose_interp.end_time()

                        if cfg.control_space == "joint":
                            # 這條路徑（預設）在 replan 時就把整個 chunk 解完 IK，
                            # 高頻迴圈只在關節空間插值，不再呼叫 IK。
                            joint_solution = arm.compute_ik_analytic(
                                position, orientation, seed_positions=last_joint_solution)
                            if joint_solution is None:
                                arm.get_logger().warning(
                                    f"[replan {replan_index}] IK 無解，跳過這個點: pos={position}")
                                continue
                            last_joint_solution = joint_solution
                            # max_joint_speed 一定要傳：actual_time 只是 pose_interp 用
                            # Cartesian 速度算出來的時間，IK 在接近奇異點/大幅旋轉時可能
                            # 讓「Cartesian 移動很小」對應到「關節角要轉很大」，沒有這個
                            # 夾子的話 joint_interp 會照樣把大位移排進很短的時間，實機上
                            # 就是瞬間高速、觸發保護性停機。
                            joint_interp.schedule_waypoint(
                                joint_solution, target_time=actual_time, curr_time=now,
                                max_joint_speed=cfg.max_joint_speed)
                        # cfg.control_space == "cartesian"：不在這裡解 IK，
                        # pose_interp 排好之後，IK 留給高頻迴圈每個 tick 現場解
                        # （見下面 query_time 那段），這裡只需要排 pose_interp。

                        if logger is not None:
                            logger.log_predicted_point(
                                replan_index, now, target_time, actual_time, position, orientation)

                    if marker_publisher is not None:
                        marker_publisher.publish(_line_strip_marker(
                            arm.base_link, cfg.planned_marker_namespace, 0, positions,
                            ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)))
                    if logger is not None:
                        logger.log_replan_trigger(replan_index, now)
                    arm.get_logger().info(f"[replan {replan_index}] 新 chunk 已排入 interpolator")
                    replan_index += 1
                    next_replan_time = now + cfg.action_horizon * cfg.action_step_dt

                query_time = now + latency_feedforward
                commanded_position, commanded_orientation = pose_interp.interpolate(query_time)

                if cfg.control_space == "joint":
                    raw_target_positions = np.array(joint_interp.interpolate(query_time))
                else:
                    # cartesian 模式：每個 tick 對 Cartesian 插值出來的目標現場解
                    # 一次 IK，seed 用上一個 tick 真正發布出去的關節角（平滑濾波器
                    # 之後的值，不是 IK 原始解），才會貼著手臂實際位置。解不出來
                    # 就當這個 tick 沒有新目標、沿用上次發布的位置，失敗次數累計
                    # 到週期性狀態 log。
                    joint_solution = arm.compute_ik_analytic(
                        commanded_position, commanded_orientation, seed_positions=last_published_positions)
                    if joint_solution is None:
                        cartesian_ik_failures_since_log += 1
                        raw_target_positions = np.array(last_published_positions)
                    else:
                        raw_target_positions = np.array(joint_solution)

                # 平滑濾波器：不管 target_positions 是 joint_interp 插值出來的
                # （chunk 交界處斜率可能不連續）還是 cartesian 現場解 IK 出來的
                # （奇異點附近可能是大位移），實際發布出去的指令都要經過同一組
                # 速度/加速度上限，兩種 control_space 都適用。速度用煞車距離公式
                # （v=sqrt(2·a·剩餘距離)）反推，不能只夾速度上限，那樣目標接近時
                # 會來不及煞車、衝過頭震盪。joint_accel_limit 見 DPControlConfig
                # 註解——這不是教導器上的規格值，是壓低指令加速度本身的軟體端
                # 保守常數，UR 控制器不管教導器有沒有對應設定，都會因為指令加速度
                # 太突然觸發 Protective Stop。
                current_positions = np.array(last_published_positions)
                current_velocity = np.array(last_published_velocity)

                position_error = raw_target_positions - current_positions
                braking_speed_limit = np.sqrt(
                    np.maximum(2.0 * cfg.joint_accel_limit * np.abs(position_error), 0.0))
                target_velocity = np.sign(position_error) * np.minimum(
                    braking_speed_limit, cfg.max_joint_speed)

                desired_acceleration = (target_velocity - current_velocity) / period
                accel_limited_acceleration = np.clip(
                    desired_acceleration, -cfg.joint_accel_limit, cfg.joint_accel_limit)
                new_velocity = current_velocity + accel_limited_acceleration * period
                new_positions = current_positions + new_velocity * period

                # 「有沒有真的被卡住」要看真正套用的速度/加速度是否頂到上限，不能
                # 看原始追蹤誤差（誤差/tick 週期換算出來的瞬間需要速度永遠很大，
                # 正常追蹤本來就會有，不代表卡到上限）。
                speed_saturated_mask = np.abs(target_velocity) >= cfg.max_joint_speed - 1e-6
                accel_saturated_mask = (
                    np.abs(accel_limited_acceleration) >= cfg.joint_accel_limit - 1e-6)
                triggered = bool(np.any(speed_saturated_mask) or np.any(accel_saturated_mask))
                if triggered and not joint_limit_active:
                    # 順便印出卡住的關節 index + 當下 J5 角度：UR 腕部奇異點是
                    # J5≈0（J4/J6 軸線重合），方便事後判斷觸發原因是不是靠近奇異點。
                    stuck_joints = sorted(set(
                        np.nonzero(speed_saturated_mask)[0].tolist()
                        + np.nonzero(accel_saturated_mask)[0].tolist()))
                    arm.get_logger().warning(
                        f"[t={now:.2f}s] 頂到平滑濾波器的關節角速度/加速度上限——被卡住的關節"
                        f"（0-indexed): {stuck_joints}，此刻 J5(index4)="
                        f"{raw_target_positions[4]:.3f}rad（越接近 0 越可能是腕部奇異點）；"
                        f"可能是靠近運動學奇異點，也可能只是這段本來就跑得快，手臂會暫時"
                        f"跟不上目標軌跡")
                joint_limit_active = triggered
                if triggered:
                    joint_limit_triggers_since_log += 1

                last_published_velocity = new_velocity.tolist()
                last_published_positions = new_positions.tolist()
                target_positions = last_published_positions
                command_publisher.publish(Float64MultiArray(data=[float(v) for v in target_positions]))
                if logger is not None:
                    logger.log_commanded_sample(now, commanded_position, commanded_orientation)

                measured_pose = arm.get_current_tcp_pose_analytic()
                if measured_pose is not None and logger is not None:
                    logger.log_measured_sample(now, measured_pose[0], measured_pose[1])

                if (marker_publisher is not None
                        and now - last_trace_publish >= cfg.trace_publish_interval_seconds):
                    executed_trace_points.append(commanded_position)
                    marker_publisher.publish(_line_strip_marker(
                        arm.base_link, cfg.executed_marker_namespace, 0, executed_trace_points,
                        ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)))
                    if measured_pose is not None:
                        measured_trace_points.append(measured_pose[0])
                        marker_publisher.publish(_line_strip_marker(
                            arm.base_link, cfg.measured_marker_namespace, 0, measured_trace_points,
                            ColorRGBA(r=0.2, g=0.4, b=1.0, a=1.0)))
                    last_trace_publish = now

                # 刻意不呼叫 rclpy.spin_once()——callback 服務完全交給背景 executor
                # thread，串流迴圈只做查表+送出 setpoint。

                loop_dt = loop_start - loop_previous_time
                loop_previous_time = loop_start
                if loop_dt > 0:
                    instant_fps = 1.0 / loop_dt
                    fps_ema = (instant_fps if fps_ema is None
                               else cfg.fps_smoothing_alpha * instant_fps
                               + (1 - cfg.fps_smoothing_alpha) * fps_ema)
                    min_instant_fps = (instant_fps if min_instant_fps is None
                                       else min(min_instant_fps, instant_fps))
                if now - last_status_log >= cfg.status_log_interval_seconds:
                    status_interval = now - last_status_log
                    joint_state_count = arm.joint_state_message_count
                    joint_state_hz = (joint_state_count - last_status_joint_state_count) / status_interval
                    cartesian_ik_status = (
                        f" | cartesian tick IK 失敗: {cartesian_ik_failures_since_log} 次"
                        if cfg.control_space == "cartesian" else "")
                    arm.get_logger().info(
                        f"實測串流頻率: 平均 {fps_ema:.1f} Hz / 這 {cfg.status_log_interval_seconds:.0f} "
                        f"秒內最低 {min_instant_fps:.1f} Hz (目標 {cfg.control_hz:.0f} Hz) | "
                        f"/joint_states 實際收到頻率: {joint_state_hz:.1f} Hz（背景 executor 健康度）"
                        f"{cartesian_ik_status} | 平滑濾波器限速/限加速度觸發: "
                        f"{joint_limit_triggers_since_log} 次")
                    last_status_log = now
                    last_status_joint_state_count = joint_state_count
                    min_instant_fps = None
                    cartesian_ik_failures_since_log = 0
                    joint_limit_triggers_since_log = 0

                if cfg.run_duration_seconds is not None and now >= cfg.run_duration_seconds:
                    arm.get_logger().info("執行時間到，停止串流")
                    break

                # 睡到下一個絕對 deadline，而不是固定 sleep(period)：避免長時間跑
                # 下來因為每個 tick 的微小誤差累積，實際頻率系統性偏低。
                next_tick_time += period
                sleep_time = next_tick_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -period:
                    # 掉超過一整個週期，是一次性的大延遲，不要為了追趕排程狂發
                    # 不睡覺的迴圈，直接回到「現在」重新校準。
                    next_tick_time = time.time()

        except KeyboardInterrupt:
            pass

        finally:
            if executor_thread is not None:
                executor_stop_event.set()
                executor_thread.join(timeout=1.0)
                executor.remove_node(arm)
            _switch_controllers(arm, [cfg.default_controller_name], [cfg.streaming_controller_name])
            if logger is not None and logger.has_data():
                data_dir = cfg.data_dir if cfg.data_dir is not None else Path.cwd() / "data"
                predicted_path, _, _, _ = logger.save(data_dir, cfg.experiment_name)
                arm.get_logger().info(
                    f"軌跡記錄已存到: {predicted_path.parent}/{cfg.experiment_name}_*.csv")
                arm.get_logger().info(
                    f"分析指令: ros2 run ur_control analyze_trajectory {cfg.experiment_name}")
            if cfg.enable_homing:
                arm.get_logger().info("回到 home pose...")
                arm.move_joint(cfg.home_joint_positions,
                                time_from_start=cfg.home_move_time_seconds, wait=True)
            if self._owns_node:
                arm.destroy_node()
