#!/usr/bin/env python3
"""Diffusion Policy 部署測試（ROS2 / ros2_control 路徑）。

架構（對照診斷報告的目標架構）：
  1. fake_dp_inference() 每 ACTION_HORIZON*ACTION_STEP_DT 秒模擬一次模型推論，
     回傳未來 PREDICTION_HORIZON 步的 (絕對時間戳, Cartesian 位置, 姿態)。
  2. 每個 chunk 點依序丟進 PoseTrajectoryInterpolator.schedule_waypoint()：
     在目前時刻截斷舊軌跡（保持位姿連續）、超過 MAX_POS_SPEED/MAX_ROT_SPEED
     就自動拉長時間（不會暴衝），絕對不重新排時間讓每段等速。
  3. 對這些「已經照速度上限調整過時間」的 Cartesian 點，各呼叫一次
     URArmNode.compute_ik_analytic()（批次，一個 chunk 16 次，不是每個控制 tick
     一次），結果連同同一個時間戳存進 JointTrajectoryInterpolator。
     這個方法用 ur_control.analytic_ik（PickNik ur-analytic-ik，IKFast 閉式解），
     in-process、微秒等級，不是原本的 MoveIt compute_ik() service round-trip
     ——後者每次十幾到幾十 ms，一個 chunk 16 次會在控制迴圈裡造成明顯的 dt 尖峰
     （實測 24-42ms，100% 落在 replan trigger 附近），換掉之後尖峰直接消失，
     不需要額外把 IK 搬到背景執行緒。細節/座標系轉換見 analytic_ik.py 開頭註解。
     注意：這 16 次 schedule_waypoint() 必須全部用同一個 curr_time（=replan
     當下的 now）呼叫——PoseTrajectoryInterpolator.schedule_waypoint() 內部每次
     都會呼叫 trim(curr_time)，只有第一次真的會裁切/collapse 緩衝區，之後只要
     curr_time 沒變就是no-op；如果把這 16 次呼叫分散到不同 tick（每次 curr_time
     都往前跳一點），trim() 會在幾乎每個點都重新 collapse，把前一個點洗掉，
     軌跡會變得一格一格的不平滑（這是實際踩過的坑，不要再犯）。
  4. 高頻控制迴圈（CONTROL_HZ）只查表：JointTrajectoryInterpolator.interpolate(now
     + LATENCY_FEEDFORWARD_SECONDS) 直接給 forward_position_controller。查詢時間
     往未來推 tau 是延遲前饋補償，見該變數旁邊的說明。

Log 管線：commanded（真的送出去的 setpoint）、measured（TF 量測回來的實際位姿）、
predicted（模型推論輸出）是三條分開記錄的時間序列，各自帶自己的原生時間戳，不在
線上互相插值/對齊——這樣才不會把「量測來源本身比較慢」誤記成「機器人在頓」。
比對/對齊留給 analyze_trajectory.py 事後做。

跟 run_waypoints.py 用的 move_pose_linear()/move_pose_waypoints() 完全是兩條路：
那兩個走 FollowJointTrajectory action、逐段規劃、逐段等待完成；這支走
forward_position_controller 的 topic 串流，程式自己決定每一刻要送什麼 setpoint，
新的 chunk 可以隨時無縫接上還沒送完的舊軌跡，不必等它跑完。

啟動前置作業：教導器 External Control 包 Loop、Remote Control 開啟。**不需要
MoveIt/move_group**——IK 走 compute_ik_analytic()（analytic_ik.py），不呼叫
MoveIt 的 compute_ik service，這是換成解析解之後的副作用（原本需要 move_group
只是因為要用它的 compute_ik service）。**第一次測試務必先用 use_fake_hardware:=true
在 RViz 裡看過一輪、確認軌跡合理再接真實手臂**。

座標系提醒：這支程式全程在 base_link + 四元數座標系下運算（MoveIt/ROS 的慣例），
fake_dp_inference() 產生的座標也直接是 base_link，不需要轉換。**但如果之後接的
是用教導器 Base 座標系（旋轉向量）記錄訓練資料的真實模型**，要在餵進
schedule_waypoint() 之前，用 pose_utils.ur_base_to_base_link() 把模型輸出轉成
base_link 座標系——這支程式目前沒有這一步，是因為 fake 模型本來就直接生成
base_link 座標，接真模型時請自行確認它的動作空間是哪個座標系。

程式結束時會自動把 controller 切回 scaled_joint_trajectory_controller，
其他程式（run_waypoints.py 等）才能正常使用，不用你手動切回去。
"""

import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float64MultiArray, ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from controller_manager_msgs.srv import SwitchController

from ur_control.ur_arm_node import URArmNode
from ur_control.trajectory_interpolator import PoseTrajectoryInterpolator, JointTrajectoryInterpolator
from ur_control.trajectory_logger import TrajectoryLogger
from ur_control.examples.fake_dp_model import fake_dp_inference

# =============================================================================
# 可修改變數
# =============================================================================
NODE_NAME = "dp_control_ros2_node"

STREAMING_CONTROLLER_NAME = "forward_position_controller"   # 高頻串流用
DEFAULT_CONTROLLER_NAME = "scaled_joint_trajectory_controller"  # 結束時切回去這個

# 每次執行都先回到這個關節姿態再開始畫圓，測試才有一致的起點可以比較。
# 這是通用的 UR「手肘向上」姿態，不是針對你工作站量身量測過的——正式接
# 真實手臂前，務必自己先確認這個姿態在你的工作空間裡不會撞到任何東西。
HOME_JOINT_POSITIONS = [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0]
HOME_MOVE_TIME_SECONDS = 4.0

# UR e-Series 原生控制迴圈頻率是 500Hz（官方 RTDE 文件）。這是 ROS2/Python 這條路
# 實際能不能穩定跑到的目標值，迴圈會印出實測 Hz，讓你自己看能撐多少。
CONTROL_HZ = 120

PREDICTION_HORIZON = 16   # T_p：每次「推論」輸出未來幾步
ACTION_HORIZON = 8        # T_a：執行到第幾步才重新推論下一個 chunk
ACTION_STEP_DT = 0.1      # chunk 裡每一步間隔幾秒（模型輸出解析度）

CIRCLE_RADIUS_M = 0.05         # 5cm 半徑 = 10cm 直徑
CIRCLE_PERIOD_SECONDS = 4     # 繞一圈幾秒；數字越大角速度越慢，先保守

MAX_POS_SPEED = 1.0            # 實體機器手臂設置上限為 1.5m/s，Cartesian 速度上限（保守值，先在 RViz 驗證再調大）
MAX_ROT_SPEED = 3.00            # 實體機器手臂設定上限為 191度/s = 3.33rad/s

RUN_DURATION_SECONDS = 30.0     # 測試安全上限，跑這麼久自動停止（Ctrl+C 也可隨時中止）

# 延遲前饋補償：控制迴圈查詢 interpolator 時，查「現在 + LATENCY_FEEDFORWARD_SECONDS」
# 而不是查「現在」，讓送出去的 setpoint 提前一點，抵銷「送出 setpoint」到「機器人
# 實際開始動」之間量測到的系統性延遲。
#
# 這個值不是猜的：用 dp_control_ros2_test 這組資料做時間平移掃描找出來的——把整條
# 預測軌跡沿時間軸平移 50ms 後，跟實際執行路徑的平均誤差從 7.94mm 降到 1.62mm；
# 誤差波形本身是軌跡速度的同頻正弦（不是雜訊），171mm/s 的切線速度 x 50ms ≈ 8.6mm，
# 跟量到的誤差量級吻合，是相位落後不是形狀不對。這組測試是在 RViz/fake hardware
# 上量的，換到真實手臂後這個延遲的組成會不一樣（沒有真實伺服/網路延遲，但可能有
# 別的瓶頸），正式部署前要用 analyze_trajectory.py 的對齊掃描在目標硬體上重新量一次，
# 不要一直沿用這個數字。
#
# 上下限是安全防呆：太大會查到 interpolator 還沒排進去的未來（該點目前的行為是
# clamp 在最後一個 waypoint，不會外插，但太超前實質上等於瞎猜），太小/負值等於
# 沒補償甚至補償方向反過來，兩種都不該讓程式直接照做。
LATENCY_FEEDFORWARD_SECONDS = 0.050
LATENCY_FEEDFORWARD_MIN_SECONDS = 0.0
LATENCY_FEEDFORWARD_MAX_SECONDS = 0.15

FPS_SMOOTHING_ALPHA = 0.1       # 實測迴圈頻率的指數移動平均係數
STATUS_LOG_INTERVAL_SECONDS = 1.0

# RViz 視覺化：在 RViz 裡 Add -> By topic -> /dp_trajectory_markers -> Marker，
# 就能看到黃色「這次 chunk 預測的路徑」和綠色「手臂實際走過的軌跡」。
TRAJECTORY_MARKER_TOPIC = "/dp_trajectory_markers"
PLANNED_MARKER_NAMESPACE = "dp_planned_chunk"
EXECUTED_MARKER_NAMESPACE = "dp_executed_trace"
TRACE_PUBLISH_INTERVAL_SECONDS = 0.1   # 執行軌跡 marker 多久重新發布一次

# 資料紀錄：把每次推論的預測路徑跟機器人實際走過的路徑存成 CSV，跑完用
# `ros2 run ur_control analyze_trajectory <EXPERIMENT_NAME>` 分析、畫圖。
# 每次想保留獨立的實驗結果就換一個名字，同名會直接覆蓋舊檔案。
EXPERIMENT_NAME = "dp_control_ros2_RealRobot_test2"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


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


def main(args=None):
    rclpy.init(args=args)
    arm = URArmNode(node_name=NODE_NAME)

    command_publisher = arm.create_publisher(
        Float64MultiArray, f"/{STREAMING_CONTROLLER_NAME}/commands", 10)
    marker_publisher = arm.create_publisher(Marker, TRAJECTORY_MARKER_TOPIC, 10)
    logger = TrajectoryLogger()

    # 背景 executor 的控制代碼。放在 try 外面先設成 None，這樣不管在串流迴圈
    # 開始之前的哪個環節提早 return/丟例外，finally 都能安全判斷「有沒有真的
    # 啟動過」，不會因為變數還沒定義就整個 NameError。
    executor = None
    executor_thread = None
    executor_stop_event = None

    try:
        arm.get_logger().info("回到 home pose...")
        if not arm.move_joint(HOME_JOINT_POSITIONS, time_from_start=HOME_MOVE_TIME_SECONDS, wait=True):
            arm.get_logger().error("回 home pose 失敗，中止")
            return

        joint_positions = arm.get_current_joint_positions()
        origin_pose = arm.wait_for_tcp_pose(timeout_sec=5.0)
        if joint_positions is None or origin_pose is None:
            arm.get_logger().error("讀不到目前關節/末端點狀態，中止")
            return
        origin_position, origin_orientation = origin_pose
        arm.get_logger().info(f"起始位置 [base_link]: {origin_position}")

        if not _switch_controllers(arm, [STREAMING_CONTROLLER_NAME], [DEFAULT_CONTROLLER_NAME]):
            arm.get_logger().error("切換到串流用 controller 失敗，中止")
            return

        input("按下 Enter 開始串流控制...")

        # 背景 executor：讓 /joint_states 這類 subscription 的 callback 有一個
        # 專門、持續在跑的 executor 服務，不再依賴串流迴圈裡的 rclpy.spin_once()。
        # 後者「一次最多服務一個 ready 的 callback」，多個 subscription（joint_states、
        # TF 內部的 /tf、/tf_static）競爭同一個名額，實測會讓有效更新率掉到個位數
        # Hz，即使 driver 端本身是 500Hz 發布（見 verify_fast_tcp_pose.py 的離線
        # 驗證：同一個 rclpy.spin_once() pattern，~8 秒只服務到 3 次 /joint_states）。
        #
        # 用 SingleThreadedExecutor 就夠：原本會需要 MultiThreadedExecutor 是因為
        # 舊版 compute_ik()（MoveIt service）的 rclpy.spin_until_future_complete()
        # 會跟背景 executor 搶著 spin 同一個 node（經典 rclpy 死鎖組合）；換成
        # compute_ik_analytic() 之後高頻路徑上已經沒有任何 service/action 呼叫，
        # 這個風險不存在了。
        #
        # 順序很重要，只在串流迴圈這段期間啟動：啟動前（home pose、
        # wait_for_tcp_pose、switch_controllers）跟結束後（finally 裡切回預設
        # controller、回 home）用的都是 URArmNode 內建方法，那些方法內部是裸的
        # rclpy.spin_once()/spin_until_future_complete()（走全域 executor）——
        # 如果背景 executor 這時候還在跑，同一個 node 會同時被兩個 executor spin，
        # 一樣會撞上前面提到的死鎖組合。
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
        last_joint_solution = list(joint_positions)

        next_replan_time = 0.0
        replan_index = 0
        fps_ema = None
        min_instant_fps = None   # 這個 status log 區間內看到的最低瞬時頻率，幫助判斷卡頓是不是集中在 replan 附近
        loop_previous_time = time.time()
        last_status_log = 0.0
        last_status_joint_state_count = arm.joint_state_message_count
        last_trace_publish = 0.0
        executed_trace_points = [origin_position]
        period = 1.0 / CONTROL_HZ
        next_tick_time = time.time()

        latency_feedforward = min(max(LATENCY_FEEDFORWARD_SECONDS, LATENCY_FEEDFORWARD_MIN_SECONDS),
                                   LATENCY_FEEDFORWARD_MAX_SECONDS)
        if latency_feedforward != LATENCY_FEEDFORWARD_SECONDS:
            arm.get_logger().warning(
                f"LATENCY_FEEDFORWARD_SECONDS={LATENCY_FEEDFORWARD_SECONDS} 超出安全範圍 "
                f"[{LATENCY_FEEDFORWARD_MIN_SECONDS}, {LATENCY_FEEDFORWARD_MAX_SECONDS}]，"
                f"已 clamp 成 {latency_feedforward}")
        arm.get_logger().info(f"延遲前饋補償 tau = {latency_feedforward * 1000:.1f}ms")

        while rclpy.ok():
            loop_start = time.time()
            now = loop_start - start_time

            if now >= next_replan_time:
                timestamps, positions, orientations = fake_dp_inference(
                    t_obs=now, start_position=origin_position, orientation=origin_orientation,
                    radius=CIRCLE_RADIUS_M,
                    angular_velocity=2.0 * np.pi / CIRCLE_PERIOD_SECONDS,
                    action_step_dt=ACTION_STEP_DT, prediction_horizon=PREDICTION_HORIZON)

                for target_time, position, orientation in zip(timestamps, positions, orientations):
                    pose_interp.schedule_waypoint(
                        position, orientation, target_time=target_time, curr_time=now,
                        max_pos_speed=MAX_POS_SPEED, max_rot_speed=MAX_ROT_SPEED)
                    actual_time = pose_interp.end_time()

                    joint_solution = arm.compute_ik_analytic(
                        position, orientation, seed_positions=last_joint_solution)
                    if joint_solution is None:
                        arm.get_logger().warning(
                            f"[replan {replan_index}] 有一個 chunk 點 IK 求不出來，跳過這個點")
                        continue
                    last_joint_solution = joint_solution
                    joint_interp.schedule_waypoint(joint_solution, target_time=actual_time, curr_time=now)
                    # 只記錄「成功求出 IK、真的被排進 joint_interp」的點：IK 求不出來的
                    # 點從來沒進到關節軌跡，機器人不會走到那裡，記錄了會讓事後分析誤判。
                    logger.log_predicted_point(
                        replan_index, now, target_time, actual_time, position, orientation)

                marker_publisher.publish(_line_strip_marker(
                    arm.base_link, PLANNED_MARKER_NAMESPACE, 0, positions,
                    ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)))

                logger.log_replan_trigger(replan_index, now)
                arm.get_logger().info(f"[replan {replan_index}] 新 chunk 已排入 interpolator")
                replan_index += 1
                next_replan_time = now + ACTION_HORIZON * ACTION_STEP_DT

            # 查詢時間往未來推 tau：interpolate() 本來就會 clamp 在最後一個
            # waypoint（trajectory_interpolator.py 既有行為，不會外插），
            # 所以這裡不需要額外處理「超出範圍」——clamp 已經是內建的。
            query_time = now + latency_feedforward
            target_positions = joint_interp.interpolate(query_time)
            command_publisher.publish(Float64MultiArray(data=[float(v) for v in target_positions]))

            # commanded：真正送出去的 setpoint 換成 Cartesian 表示，跟 joint_interp
            # 是用同一組 (target_time, curr_time) 排程出來的，只是查詢另一個 buffer，
            # 純粹是為了記錄方便跟 predicted/measured 直接比，不影響實際送出的指令。
            commanded_position, commanded_orientation = pose_interp.interpolate(query_time)
            logger.log_commanded_sample(now, commanded_position, commanded_orientation)

            # FK 版本，不是 TF 版本：不依賴 TF 的 /tf subscription 被服務到，只要
            # /joint_states 本身有更新（背景 executor 負責這件事）就是新的。
            measured_pose = arm.get_current_tcp_pose_analytic()
            if measured_pose is not None:
                logger.log_measured_sample(now, measured_pose[0], measured_pose[1])

            if now - last_trace_publish >= TRACE_PUBLISH_INTERVAL_SECONDS:
                executed_trace_points.append(commanded_position)
                marker_publisher.publish(_line_strip_marker(
                    arm.base_link, EXECUTED_MARKER_NAMESPACE, 0, executed_trace_points,
                    ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)))
                last_trace_publish = now

            # 注意：這裡刻意不呼叫 rclpy.spin_once()——callback 服務完全交給上面
            # 啟動的背景 executor thread，串流迴圈只做查表+送出 setpoint，不做
            # 任何可能被拿去服務其他 callback、進而讓自己延遲的事。

            loop_dt = loop_start - loop_previous_time
            loop_previous_time = loop_start
            if loop_dt > 0:
                instant_fps = 1.0 / loop_dt
                fps_ema = (instant_fps if fps_ema is None
                           else FPS_SMOOTHING_ALPHA * instant_fps + (1 - FPS_SMOOTHING_ALPHA) * fps_ema)
                min_instant_fps = instant_fps if min_instant_fps is None else min(min_instant_fps, instant_fps)
            if now - last_status_log >= STATUS_LOG_INTERVAL_SECONDS:
                status_interval = now - last_status_log
                joint_state_count = arm.joint_state_message_count
                joint_state_hz = (joint_state_count - last_status_joint_state_count) / status_interval
                arm.get_logger().info(
                    f"實測串流頻率: 平均 {fps_ema:.1f} Hz / 這 {STATUS_LOG_INTERVAL_SECONDS:.0f} 秒內最低 "
                    f"{min_instant_fps:.1f} Hz (目標 {CONTROL_HZ:.0f} Hz) | "
                    f"/joint_states 實際收到頻率: {joint_state_hz:.1f} Hz（背景 executor 健康度）")
                last_status_log = now
                last_status_joint_state_count = joint_state_count
                min_instant_fps = None

            if now >= RUN_DURATION_SECONDS:
                arm.get_logger().info("測試時間到，停止串流")
                break

            # 睡到下一個絕對 deadline，而不是固定 sleep(period)：後者每次都用
            # 「現在」當基準，每個 tick 累積的微小誤差（sleep 精度、迴圈本身的
            # 執行時間）會不斷疊加，長時間跑下來實際頻率會系統性偏低。
            next_tick_time += period
            sleep_time = next_tick_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -period:
                # 掉超過一整個週期，代表發生了一次性的大延遲（不是慢慢累積的
                # 誤差）——直接回到「現在」重新校準，不要為了追趕排程狂發不睡覺
                # 的迴圈。
                next_tick_time = time.time()

    except KeyboardInterrupt:
        pass

    finally:
        if executor_thread is not None:
            executor_stop_event.set()
            executor_thread.join(timeout=1.0)
            executor.remove_node(arm)
        _switch_controllers(arm, [DEFAULT_CONTROLLER_NAME], [STREAMING_CONTROLLER_NAME])
        if logger.has_data():
            predicted_path, _, _, _ = logger.save(DATA_DIR, EXPERIMENT_NAME)
            arm.get_logger().info(f"軌跡記錄已存到: {predicted_path.parent}/{EXPERIMENT_NAME}_*.csv")
            arm.get_logger().info(f"分析指令: ros2 run ur_control analyze_trajectory {EXPERIMENT_NAME}")
        arm.get_logger().info("回到 home pose...")
        arm.move_joint(HOME_JOINT_POSITIONS, time_from_start=HOME_MOVE_TIME_SECONDS, wait=True)
        arm.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
