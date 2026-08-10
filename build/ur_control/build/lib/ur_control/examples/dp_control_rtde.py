#!/usr/bin/env python3
"""Diffusion Policy 部署測試（原生 RTDE 路徑，跟 dp_control_ros2.py 互相對照）。

跟 dp_control_ros2.py 共用完全相同的 fake_dp_inference()、
PoseTrajectoryInterpolator、JointTrajectoryInterpolator，唯一的差異在最後
「怎麼跟手臂溝通」這一層：
  - IK：不是呼叫 MoveIt 的 compute_ik service，而是 ur_rtde 內建、在本地端計算的
    getInverseKinematics()（用機器人自己校正過的運動學，理論上比 ROS service
    快很多，但一樣是每個 chunk 點呼叫一次，不是每個控制 tick 呼叫一次——設計
    上跟 ROS2 版本刻意保持一致，這樣兩邊比較才公平）。
  - 串流：不是發 ROS topic 給 forward_position_controller，而是直接呼叫 UR
    官方建議的 500Hz servoJ 迴圈（initPeriod/waitPeriod 是 ur_rtde 官方範例
    用來精準抓 2ms 週期的寫法）。
  - 完全不經過 ROS2/ros2_control，是一條獨立的通訊路徑。

**重要：這條路徑不能跟 ur_robot_driver 的 ROS2 external control 連線同時運作**
（兩邊都想搶著送運動指令給同一台機器人）。測試這支程式前，請先確認
ur_control.launch.py／ur_robotiq_bringup 那個 terminal 沒有在跑，或至少確定它
沒有主動送出任何動作指令。

**這條路徑沒辦法像 dp_control_ros2.py 一樣先用 RViz + use_fake_hardware 驗證**
——RTDE 是直接連線真實機器人控制器（或 URSim），沒有等效的「假硬體」可以測。
如果你有裝 URSim，可以先對 URSim 的虛擬 IP 測；沒有的話，第一次測試務必用
很保守的 MAX_POS_SPEED/MAX_ROT_SPEED、確保周圍淨空、備妥急停。

座標系提醒：程式內部（interpolator、fake_dp_inference）一律用 base_link + 四元數
運算，跟 dp_control_ros2.py 完全一樣；只有在讀取 RTDE 目前姿態（
getActualTCPPose，回傳教導器 Base 座標系 + 旋轉向量）、以及要送出 IK / servoJ
目標之前，才用 pose_utils 轉成/轉回教導器 Base 座標系——這就是「輸出給實體
機器人時座標需要轉一下」的那一步，已經處理在 _to_ur_pose()/_from_ur_pose() 這
兩個函式裡。
"""

import time

import numpy as np
import rtde_control
import rtde_receive

from ur_control.trajectory_interpolator import PoseTrajectoryInterpolator, JointTrajectoryInterpolator
from ur_control.pose_utils import (
    ur_base_to_base_link,
    base_link_to_ur_base,
    quaternion_to_rotation_vector,
)
from ur_control.examples.fake_dp_model import fake_dp_inference

# =============================================================================
# 可修改變數
# =============================================================================
ROBOT_IP = "192.168.1.30"

CONTROL_HZ = 500.0        # UR e-Series 官方建議的 servoJ 迴圈頻率（RTDE 原生控制迴圈頻率）

PREDICTION_HORIZON = 16   # T_p
ACTION_HORIZON = 8        # T_a
ACTION_STEP_DT = 0.1      # chunk 裡每一步間隔幾秒

CIRCLE_RADIUS_M = 0.05
CIRCLE_PERIOD_SECONDS = 8.0

MAX_POS_SPEED = 0.15      # m/s，先保守
MAX_ROT_SPEED = 0.5       # rad/s

SERVO_LOOKAHEAD_TIME = 0.1   # servoJ 參數，官方建議範圍 [0.03, 0.2]
SERVO_GAIN = 300.0           # servoJ 參數，官方建議範圍 [100, 2000]
SERVO_STOP_DECELERATION = 5.0  # m/s^2，結束時 servoStop 用的減速度，保守值

IK_MAX_POSITION_ERROR = 1e-5
IK_MAX_ORIENTATION_ERROR = 1e-3

RUN_DURATION_SECONDS = 30.0

FPS_SMOOTHING_ALPHA = 0.1
STATUS_LOG_INTERVAL_SECONDS = 1.0


def _to_ur_pose(position, orientation):
    """base_link (position, quaternion xyzw) -> UR Base 座標系的 [x,y,z,rx,ry,rz]
    （getInverseKinematics / servoJ 要的格式）。"""
    rx, ry, rz = quaternion_to_rotation_vector(*orientation)
    base_position, base_rotvec = base_link_to_ur_base(tuple(position), (rx, ry, rz))
    return list(base_position) + list(base_rotvec)


def _from_ur_pose(ur_pose):
    """UR Base 座標系的 [x,y,z,rx,ry,rz]（getActualTCPPose 回傳的格式）
    -> base_link (position, quaternion xyzw)。"""
    return ur_base_to_base_link(tuple(ur_pose[0:3]), tuple(ur_pose[3:6]))


def main():
    # 注意：RTDEControlInterface 連不上時不會很快丟例外，而是內部重試、可能卡很久
    # 才放棄（實測對一個完全不存在的 IP 卡超過 20 秒還沒結束）。如果卡住太久，
    # Ctrl+C 中止、確認 ROBOT_IP 是否正確、Remote Control 有沒有開、以及有沒有
    # 其他程式（例如 ROS2 external control）正在佔用連線。
    print(f"連線到 {ROBOT_IP} ...（連不上的話不會馬上報錯，卡住太久就 Ctrl+C 檢查 IP）")
    try:
        rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    except RuntimeError as exc:
        print(f"連線失敗: {exc}")
        print("確認機器人 IP 正確、Remote Control 開啟、且沒有其他程式（例如 ROS2 "
              "external control）正在佔用連線。")
        return

    try:
        joint_positions = rtde_r.getActualQ()
        origin_position, origin_orientation = _from_ur_pose(rtde_r.getActualTCPPose())
        print(f"起始位置 [base_link]: {origin_position}")

        input("按下 Enter 開始串流控制...")

        start_time = time.time()
        pose_interp = PoseTrajectoryInterpolator(0.0, origin_position, origin_orientation)
        joint_interp = JointTrajectoryInterpolator(0.0, joint_positions)
        last_joint_solution = list(joint_positions)

        next_replan_time = 0.0
        replan_index = 0
        fps_ema = None
        loop_previous_time = time.time()
        last_status_log = 0.0
        dt = 1.0 / CONTROL_HZ

        while True:
            t_start = rtde_c.initPeriod()
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

                    ur_pose = _to_ur_pose(position, orientation)
                    has_solution = rtde_c.getInverseKinematicsHasSolution(
                        ur_pose, last_joint_solution, IK_MAX_POSITION_ERROR, IK_MAX_ORIENTATION_ERROR)
                    if not has_solution:
                        print(f"[replan {replan_index}] 有一個 chunk 點 IK 求不出來，跳過這個點")
                        continue
                    joint_solution = rtde_c.getInverseKinematics(
                        ur_pose, last_joint_solution, IK_MAX_POSITION_ERROR, IK_MAX_ORIENTATION_ERROR)
                    last_joint_solution = joint_solution
                    joint_interp.schedule_waypoint(joint_solution, target_time=actual_time, curr_time=now)

                print(f"[replan {replan_index}] 新 chunk 已排入 interpolator")
                replan_index += 1
                next_replan_time = now + ACTION_HORIZON * ACTION_STEP_DT

            target_positions = [float(v) for v in joint_interp.interpolate(now)]
            rtde_c.servoJ(target_positions, 0.0, 0.0, dt, SERVO_LOOKAHEAD_TIME, SERVO_GAIN)

            loop_dt = loop_start - loop_previous_time
            loop_previous_time = loop_start
            if loop_dt > 0:
                instant_fps = 1.0 / loop_dt
                fps_ema = (instant_fps if fps_ema is None
                           else FPS_SMOOTHING_ALPHA * instant_fps + (1 - FPS_SMOOTHING_ALPHA) * fps_ema)
            if now - last_status_log >= STATUS_LOG_INTERVAL_SECONDS:
                print(f"實測串流頻率: {fps_ema:.1f} Hz (目標 {CONTROL_HZ:.0f} Hz)")
                last_status_log = now

            if now >= RUN_DURATION_SECONDS:
                print("測試時間到，停止串流")
                break

            rtde_c.waitPeriod(t_start)

    except KeyboardInterrupt:
        pass

    finally:
        rtde_c.servoStop(SERVO_STOP_DECELERATION)
        rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()


if __name__ == "__main__":
    main()
