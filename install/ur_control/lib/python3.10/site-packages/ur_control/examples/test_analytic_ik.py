#!/usr/bin/env python3
"""離線迴歸測試：`ur_control.analytic_ik` 的 round-trip 正確性。不需要 ROS、
MoveIt、機器人——純數學，跟 `test_trajectory_interpolator.py` 同一個風格。每個
姿態都做「用已知關節角算出目標姿態 -> 用 seed（刻意加擾動）反解 -> 解出來的角度
再正向算一次 -> 應該要回到原本的目標姿態」，任何一步不吻合就算失敗。

兩輪測試：(1) 隨機取樣，抓「改壞了」的意外回歸；(2) 結構化網格取樣，直接枚舉
「高度變化」（肩/肘關節 J2/J3）× 「大幅旋轉」（腕關節 J4/J5/J6）的交叉組合，
涵蓋隨機取樣不一定會碰到的極端姿態（背景見 README「問題與解決方法」）。
"""

import itertools

import numpy as np

from ur_control import analytic_ik
from ur_analytic_ik import ur5e

NUM_RANDOM_CONFIGS = 500
SEED_PERTURBATION_RAD = 0.1
POSITION_TOLERANCE_M = 0.001
ORIENTATION_TOLERANCE_DEG = 0.1
RANDOM_SEED = 42

# 結構化網格：J2/J3（肩/肘，主要決定末端點高度跟手臂伸展形狀）取幾個涵蓋
# 「手肘上/手肘下/伸直/彎折」的離散角度；J4/J5/J6（腕部三軸，決定末端點姿態）
# 各自取幾個涵蓋大幅旋轉/翻轉的離散角度；J1（底座偏航，只轉整隻手臂的朝向，
# 不影響高度或局部姿態的解算難度）另外取幾個值代表，不需要跟其他軸做全交叉。
_HEIGHT_J2_DEG = [-150, -120, -90, -60, -30]
_HEIGHT_J3_DEG = [-150, -90, -30, 30, 90, 150]
_WRIST_J4_DEG = [-180, -90, 0, 90, 180]
_WRIST_J5_DEG = [-90, -45, 0, 45, 90]
_WRIST_J6_DEG = [-180, -90, 0, 90, 180]
_BASE_J1_DEG = [-120, 0, 120]


def _check_round_trip(q_true: np.ndarray, rng: np.random.Generator):
    """單一姿態的 IK round-trip 驗證。回傳 (ok, reason)，`reason` 只有失敗時有意義。
    `rng` 只用來產生 seed 擾動（模擬真實使用情境：seed 不會剛好等於答案），呼叫端
    自己控制要不要固定種子以求可重現。"""
    target_matrix = ur5e.forward_kinematics(*q_true)
    position, quat = analytic_ik._ur_matrix_to_base_link_pose(target_matrix)

    seed = q_true + rng.uniform(-SEED_PERTURBATION_RAD, SEED_PERTURBATION_RAD, size=6)
    solution = analytic_ik.solve(position, quat, seed)

    if solution is None:
        return False, "solve() 回傳 None（無解或內部 round-trip 驗證失敗）"

    check_matrix = ur5e.forward_kinematics(*solution)
    pos_err = float(np.linalg.norm(check_matrix[:3, 3] - target_matrix[:3, 3]))
    rot_err_deg = float(np.degrees(2.0 * np.arccos(min(1.0, max(-1.0, abs(float(
        np.dot(analytic_ik._matrix_to_quaternion(check_matrix[:3, :3]),
               analytic_ik._matrix_to_quaternion(target_matrix[:3, :3])))))))))

    if pos_err > POSITION_TOLERANCE_M:
        return False, f"position round-trip 誤差 {pos_err * 1000:.4f}mm"
    if rot_err_deg > ORIENTATION_TOLERANCE_DEG:
        return False, f"orientation round-trip 誤差 {rot_err_deg:.4f}deg"
    return True, ""


def _run_random_sweep():
    rng = np.random.default_rng(RANDOM_SEED)
    failures = []
    for i in range(NUM_RANDOM_CONFIGS):
        q_true = rng.uniform(-np.pi, np.pi, size=6)
        ok, reason = _check_round_trip(q_true, rng)
        if not ok:
            failures.append((i, reason, q_true))

    print(f"[隨機取樣] 測了 {NUM_RANDOM_CONFIGS} 組隨機關節姿態（seed 加了 up to "
          f"{np.degrees(SEED_PERTURBATION_RAD):.1f}deg 的擾動，模擬真實使用情境）")
    print(f"[隨機取樣] 失敗: {len(failures)} / {NUM_RANDOM_CONFIGS}")
    for idx, reason, q in failures[:20]:
        print(f"  #{idx}: {reason}  q_true={q}")
    return failures


def _run_structured_workspace_sweep():
    height_configs = list(itertools.product(
        np.radians(_HEIGHT_J2_DEG), np.radians(_HEIGHT_J3_DEG)))
    wrist_configs = list(itertools.product(
        np.radians(_WRIST_J4_DEG), np.radians(_WRIST_J5_DEG), np.radians(_WRIST_J6_DEG)))
    base_values = list(np.radians(_BASE_J1_DEG))

    configs = []
    for j1, (j2, j3), (j4, j5, j6) in itertools.product(base_values, height_configs, wrist_configs):
        configs.append(np.array([j1, j2, j3, j4, j5, j6]))

    rng = np.random.default_rng(RANDOM_SEED + 1)   # 跟隨機取樣那輪用不同種子，避免湊巧掩蓋問題
    failures = []
    for i, q_true in enumerate(configs):
        ok, reason = _check_round_trip(q_true, rng)
        if not ok:
            failures.append((i, reason, q_true))

    print(f"[結構化網格] 測了 {len(configs)} 組「高度 x 旋轉」交叉姿態 "
          f"(J2 x{len(_HEIGHT_J2_DEG)} x J3 x{len(_HEIGHT_J3_DEG)} 涵蓋高度/手肘形狀，"
          f"J4 x{len(_WRIST_J4_DEG)} x J5 x{len(_WRIST_J5_DEG)} x J6 x{len(_WRIST_J6_DEG)} "
          f"涵蓋大幅旋轉，J1 x{len(_BASE_J1_DEG)} 涵蓋底座朝向)")
    print(f"[結構化網格] 失敗: {len(failures)} / {len(configs)}")
    for idx, reason, q in failures[:20]:
        print(f"  #{idx}: {reason}  q_true={q}")
    return failures


def main():
    random_failures = _run_random_sweep()
    structured_failures = _run_structured_workspace_sweep()

    total_failures = len(random_failures) + len(structured_failures)
    if total_failures:
        raise SystemExit(
            f"analytic_ik round-trip 測試失敗（隨機 {len(random_failures)} 筆 + "
            f"結構化 {len(structured_failures)} 筆），在部署到控制迴圈前必須先查清楚原因")
    print("全部通過")


if __name__ == "__main__":
    main()
