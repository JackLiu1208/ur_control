#!/usr/bin/env python3
"""終端機即時顯示 UR 手臂狀態：關節角度、末端點姿態（教導器 Base 座標系）、
關節與末端點的速度/加速度，以及顯示迴圈本身的處理速度 (FPS)。

這支程式只負責「顯示」；實際的狀態計算（讀值、微分算速度/加速度、FPS）都在
ur_control.state_monitor.ArmStateMonitor 裡，其他程式要直接拿
joint_state/ee_state/velocity/acceleration/FPS 使用，不需要跑這支程式，
直接這樣用就好：

    from ur_control.ur_arm_node import URArmNode
    from ur_control.state_monitor import ArmStateMonitor

    arm = URArmNode(...)
    monitor = ArmStateMonitor(arm)
    sample = monitor.sample()   # 每個迴圈呼叫一次，內部會自己 spin_once

純讀取、不會對手臂下任何移動指令，不需要等 action server。
"""

import math
import sys
import time

import rclpy

from ur_control.ur_arm_node import URArmNode
from ur_control.state_monitor import ArmStateMonitor

# =============================================================================
# 可修改變數
# =============================================================================
NODE_NAME = "ur_state_monitor_node"

TARGET_FPS = 30.0                # 目標刷新率 (Hz)，可調整
JOINT_DISPLAY_IN_DEGREES = True  # 關節角度/速度/加速度顯示用度數(True)或弧度(False)
SMOOTHING_ALPHA = 0.5            # 速度/加速度/FPS 的指數移動平均係數 (0~1；越大越貼近最新值、越小越平滑)


def _build_lines(joint_names, sample):
    unit = "deg" if JOINT_DISPLAY_IN_DEGREES else "rad"
    to_display = math.degrees if JOINT_DISPLAY_IN_DEGREES else (lambda v: v)

    lines = []
    fps_text = f"{sample.fps:.1f} / 目標 {TARGET_FPS:.1f}" if sample.fps is not None else "--"
    lines.append(f"=== UR Arm State Monitor ===  FPS: {fps_text}")
    lines.append("")

    # 每一幀輸出的行數固定（缺資料時印佔位行），這樣固定位置重繪才不會在資料
    # 忽有忽無時，把上一幀比較長的殘留內容留在畫面上。
    name_width = max(len(name) for name in joint_names)
    lines.append(f"[關節]  (單位: {unit}, {unit}/s, {unit}/s^2)")
    velocities = sample.velocity.joint
    accelerations = sample.acceleration.joint
    for i, name in enumerate(joint_names):
        if sample.joint_state is None:
            lines.append(f"  {name:<{name_width}s} (尚未收到 /joint_states)")
            continue
        pos = to_display(sample.joint_state.positions[i])
        vel = to_display(velocities[i]) if velocities is not None else float("nan")
        accel = to_display(accelerations[i]) if accelerations is not None else float("nan")
        lines.append(f"  {name:<{name_width}s} pos={pos:9.3f}  vel={vel:9.3f}  accel={accel:9.3f}")
    lines.append("")

    lines.append("[末端點 - 教導器 Base 座標系]  (單位: m, rad, m/s, rad/s, m/s^2, rad/s^2)")
    if sample.ee_state is None:
        lines.append("  position     (尚未取得末端點姿態，tf base_link -> tool0)")
        lines.append("  rotvec")
        lines.append("  linear vel")
        lines.append("  angular vel")
        lines.append("  linear acc")
        lines.append("  angular acc")
    else:
        p = sample.ee_state.position
        r = sample.ee_state.rotation_vector
        lv = sample.velocity.ee_linear or (0.0, 0.0, 0.0)
        av = sample.velocity.ee_angular or (0.0, 0.0, 0.0)
        la = sample.acceleration.ee_linear or (0.0, 0.0, 0.0)
        aa = sample.acceleration.ee_angular or (0.0, 0.0, 0.0)
        lines.append(f"  position     x={p[0]:8.4f}  y={p[1]:8.4f}  z={p[2]:8.4f}")
        lines.append(f"  rotvec      rx={r[0]:8.4f} ry={r[1]:8.4f} rz={r[2]:8.4f}")
        lines.append(f"  linear vel  vx={lv[0]:8.4f} vy={lv[1]:8.4f} vz={lv[2]:8.4f}")
        lines.append(f"  angular vel wx={av[0]:8.4f} wy={av[1]:8.4f} wz={av[2]:8.4f}")
        lines.append(f"  linear acc  ax={la[0]:8.4f} ay={la[1]:8.4f} az={la[2]:8.4f}")
        lines.append(f"  angular acc ax={aa[0]:8.4f} ay={aa[1]:8.4f} az={aa[2]:8.4f}")
    lines.append("")
    lines.append("(Ctrl+C 結束)")
    return lines


def _render(joint_names, sample, is_first_frame):
    lines = _build_lines(joint_names, sample)
    if is_first_frame:
        sys.stdout.write("\033[2J")  # 只在第一幀清整個畫面一次
    sys.stdout.write("\033[H")       # 游標歸位到固定位置，不再每幀清畫面
    for line in lines:
        sys.stdout.write("\033[K" + line + "\n")  # 清掉這行舊內容再蓋上新的，避免殘影
    sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    arm = URArmNode(node_name=NODE_NAME)
    monitor = ArmStateMonitor(arm, smoothing_alpha=SMOOTHING_ALPHA)

    period = 1.0 / TARGET_FPS
    is_first_frame = True
    try:
        while rclpy.ok():
            loop_start = time.time()
            sample = monitor.sample()
            _render(arm.joint_names, sample, is_first_frame)
            is_first_frame = False

            elapsed = time.time() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    finally:
        arm.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
