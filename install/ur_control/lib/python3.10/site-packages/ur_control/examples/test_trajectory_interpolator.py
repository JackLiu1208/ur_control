#!/usr/bin/env python3
"""離線測試 PoseTrajectoryInterpolator：不需要 ROS、不需要機器手臂。

餵入兩段時間上有重疊的假 action chunk（模擬 receding horizon：第一段在
t_obs=0 推論、執行到 T_a 步時，第二段在 t_obs=T_a*action_step_dt 才推論，
兩段在中間那一小段時間內是重疊的），用 CONTROL_HZ 高頻取樣整條插值後的軌跡，
印出位置/速度/加速度隨時間的變化，並在接縫時刻（第二段 schedule 進來、trim()
被呼叫的那個時間點）檢查位置、速度有沒有跳變。

跑法：
    python3 test_trajectory_interpolator.py
    # 或
    ros2 run ur_control test_trajectory_interpolator
"""

import numpy as np

from ur_control.trajectory_interpolator import PoseTrajectoryInterpolator
from ur_control.examples.fake_dp_model import fake_dp_inference

# =============================================================================
# 可修改變數（跟 dp_control_ros2.py / dp_control_rtde.py 用同一組預設值）
# =============================================================================
CONTROL_HZ = 500.0          # 取樣/串流頻率，UR e-Series 原生控制迴圈頻率
PREDICTION_HORIZON = 16     # T_p
ACTION_HORIZON = 8          # T_a：執行到第幾步才重新推論下一個 chunk
ACTION_STEP_DT = 0.1        # chunk 裡每一步間隔幾秒（模型輸出解析度）
CIRCLE_RADIUS_M = 0.05      # 5cm 半徑 = 10cm 直徑
ANGULAR_VELOCITY_RAD_S = 2.0 * np.pi / 8.0  # 繞一圈 8 秒，不會太快

MAX_POS_SPEED = 0.25        # m/s，schedule_waypoint 的保守速度上限
MAX_ROT_SPEED = 1.0         # rad/s

# 判定「有沒有跳變」的容忍值
POSITION_JUMP_TOLERANCE_M = 0.005
VELOCITY_JUMP_TOLERANCE_M_S = 0.05


def main():
    start_position = np.array([0.0, -0.4, 0.3])
    orientation = np.array([0.0, 0.0, 0.0, 1.0])

    interp = PoseTrajectoryInterpolator(0.0, start_position, orientation)

    # --- 第一段 chunk：在 t_obs=0 推論 ---
    t1, p1, o1 = fake_dp_inference(
        t_obs=0.0, start_position=start_position, orientation=orientation,
        radius=CIRCLE_RADIUS_M, angular_velocity=ANGULAR_VELOCITY_RAD_S,
        action_step_dt=ACTION_STEP_DT, prediction_horizon=PREDICTION_HORIZON)
    for t, p, o in zip(t1, p1, o1):
        interp.schedule_waypoint(p, o, target_time=t, curr_time=0.0,
                                  max_pos_speed=MAX_POS_SPEED, max_rot_speed=MAX_ROT_SPEED)

    # --- 執行到第 T_a 步的時刻，第二段 chunk 才進來（跟第一段後半段時間重疊）---
    replan_time = ACTION_HORIZON * ACTION_STEP_DT
    t2, p2, o2 = fake_dp_inference(
        t_obs=replan_time, start_position=start_position, orientation=orientation,
        radius=CIRCLE_RADIUS_M, angular_velocity=ANGULAR_VELOCITY_RAD_S,
        action_step_dt=ACTION_STEP_DT, prediction_horizon=PREDICTION_HORIZON)
    print(f"第一段涵蓋時間: [{t1[0]:.3f}, {t1[-1]:.3f}]s")
    print(f"第二段涵蓋時間: [{t2[0]:.3f}, {t2[-1]:.3f}]s  (在 t={replan_time:.3f}s 插入)")
    print(f"重疊區間: [{t2[0]:.3f}, {t1[-1]:.3f}]s\n")

    # 位置/速度在接縫前一瞬間的樣子（trim() 之前查詢）
    pre_position, _ = interp.interpolate(replan_time)
    pre_velocity, _ = interp.interpolate_velocity(replan_time)

    for t, p, o in zip(t2, p2, o2):
        interp.schedule_waypoint(p, o, target_time=t, curr_time=replan_time,
                                  max_pos_speed=MAX_POS_SPEED, max_rot_speed=MAX_ROT_SPEED)

    # 接縫後同一時刻查詢，應該跟接縫前幾乎一樣（trim 保證位置連續）
    post_position, _ = interp.interpolate(replan_time)
    post_velocity, _ = interp.interpolate_velocity(replan_time)

    position_jump = float(np.linalg.norm(post_position - pre_position))
    velocity_jump = float(np.linalg.norm(post_velocity - pre_velocity))

    # --- 高頻取樣整條軌跡 ---
    dt = 1.0 / CONTROL_HZ
    sample_times = np.arange(interp.start_time(), interp.end_time(), dt)
    positions = np.array([interp.interpolate(t)[0] for t in sample_times])
    velocities = np.array([interp.interpolate_velocity(t)[0] for t in sample_times])
    accelerations = np.diff(velocities, axis=0, prepend=velocities[:1]) / dt

    print(f"取樣點數: {len(sample_times)} (CONTROL_HZ={CONTROL_HZ:.0f})")
    print(f"位置範圍: x[{positions[:,0].min():.4f}, {positions[:,0].max():.4f}] "
          f"y[{positions[:,1].min():.4f}, {positions[:,1].max():.4f}]")
    print(f"速度大小範圍: [{np.linalg.norm(velocities, axis=1).min():.4f}, "
          f"{np.linalg.norm(velocities, axis=1).max():.4f}] m/s")
    print(f"加速度大小最大值: {np.linalg.norm(accelerations, axis=1).max():.4f} m/s^2\n")

    # 每 0.2 秒印一行，方便肉眼掃過去看有沒有異常尖峰
    print(f"{'t(s)':>8} {'x':>9} {'y':>9} {'|v|(m/s)':>10} {'|a|(m/s^2)':>11}")
    step = max(int(0.2 / dt), 1)
    for i in range(0, len(sample_times), step):
        v_norm = float(np.linalg.norm(velocities[i]))
        a_norm = float(np.linalg.norm(accelerations[i]))
        marker = "  <-- 接縫附近" if abs(sample_times[i] - replan_time) < 0.15 else ""
        print(f"{sample_times[i]:8.3f} {positions[i,0]:9.4f} {positions[i,1]:9.4f} "
              f"{v_norm:10.4f} {a_norm:11.4f}{marker}")

    print(f"\n接縫處位置跳變: {position_jump * 1000:.3f} mm "
          f"(容忍值 {POSITION_JUMP_TOLERANCE_M * 1000:.1f} mm)")
    print(f"接縫處速度跳變: {velocity_jump:.4f} m/s "
          f"(容忍值 {VELOCITY_JUMP_TOLERANCE_M_S:.2f} m/s)")

    position_ok = position_jump <= POSITION_JUMP_TOLERANCE_M
    velocity_ok = velocity_jump <= VELOCITY_JUMP_TOLERANCE_M_S
    print(f"\n位置連續: {'PASS' if position_ok else 'FAIL'}")
    print(f"速度連續: {'PASS' if velocity_ok else 'FAIL'}")
    print(f"\n整體結果: {'PASS' if (position_ok and velocity_ok) else 'FAIL'}")


if __name__ == "__main__":
    main()
