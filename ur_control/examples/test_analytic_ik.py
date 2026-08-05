#!/usr/bin/env python3
"""離線迴歸測試：ur_control.analytic_ik 的 round-trip 正確性。不需要 ROS、
MoveIt、機器人——純數學，跟 test_trajectory_interpolator.py 同一個風格。

這不是「座標系轉換一定對」的原始證明——那件事已經在真實機器人上做過一次
（2026-08，見 analytic_ik.py 開頭註解：拿當下真實關節角同時算
ur_analytic_ik.forward_kinematics() 跟呼叫 MoveIt /compute_fk 比對，完全吻合）。
這支的目的是擋住「以後改東西改壞了沒發現」的回歸：隨機灑一堆姿態，each 都做
「用已知關節角算出目標姿態 -> 用 seed（刻意加擾動）反解 -> 解出來的角度再正向
算一次 -> 應該要回到原本的目標姿態」，任何一步不吻合就算失敗。
"""

import numpy as np

from ur_control import analytic_ik
from ur_analytic_ik import ur5e

NUM_RANDOM_CONFIGS = 500
SEED_PERTURBATION_RAD = 0.1
POSITION_TOLERANCE_M = 0.001
ORIENTATION_TOLERANCE_DEG = 0.1
RANDOM_SEED = 42


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    failures = []

    for i in range(NUM_RANDOM_CONFIGS):
        q_true = rng.uniform(-np.pi, np.pi, size=6)
        target_matrix = ur5e.forward_kinematics(*q_true)
        position, quat = analytic_ik._ur_matrix_to_base_link_pose(target_matrix)

        seed = q_true + rng.uniform(-SEED_PERTURBATION_RAD, SEED_PERTURBATION_RAD, size=6)
        solution = analytic_ik.solve(position, quat, seed)

        if solution is None:
            failures.append((i, "solve() 回傳 None（無解或內部 round-trip 驗證失敗）", q_true))
            continue

        check_matrix = ur5e.forward_kinematics(*solution)
        pos_err = float(np.linalg.norm(check_matrix[:3, 3] - target_matrix[:3, 3]))
        rot_err_deg = float(np.degrees(2.0 * np.arccos(min(1.0, max(-1.0, abs(float(
            np.dot(analytic_ik._matrix_to_quaternion(check_matrix[:3, :3]),
                   analytic_ik._matrix_to_quaternion(target_matrix[:3, :3])))))))))

        if pos_err > POSITION_TOLERANCE_M:
            failures.append((i, f"position round-trip 誤差 {pos_err * 1000:.4f}mm", q_true))
        elif rot_err_deg > ORIENTATION_TOLERANCE_DEG:
            failures.append((i, f"orientation round-trip 誤差 {rot_err_deg:.4f}deg", q_true))

    print(f"測了 {NUM_RANDOM_CONFIGS} 組隨機關節姿態（seed 加了 up to "
          f"{np.degrees(SEED_PERTURBATION_RAD):.1f}deg 的擾動，模擬真實使用情境）")
    print(f"失敗: {len(failures)} / {NUM_RANDOM_CONFIGS}")
    for idx, reason, q in failures[:20]:
        print(f"  #{idx}: {reason}  q_true={q}")

    if failures:
        raise SystemExit(
            f"analytic_ik round-trip 測試失敗（{len(failures)} 筆），"
            "在部署到控制迴圈前必須先查清楚原因")
    print("全部通過")


if __name__ == "__main__":
    main()
