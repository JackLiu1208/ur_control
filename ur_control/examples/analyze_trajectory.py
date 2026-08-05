#!/usr/bin/env python3
"""事後分析：dp_control_*.py 錄下的「模型推論路徑」跟「機器人實際執行路徑」是否相符。

只看時序上「有推論、後來也真的被走到」的點：一個 chunk 有 PREDICTION_HORIZON 個
預測點，但通常只有前 ACTION_HORIZON 步左右會在下一次 replan 蓋掉這個 chunk 之前
真的被機器人走到，其餘的點會被新 chunk 取代、機器人永遠不會經過那裡。判斷方式：
用 <experiment>_replan_boundaries.csv 記錄的「下一次 replan 觸發時間」當分界，
一個預測點的 actual_time（已經照 MAX_POS_SPEED/MAX_ROT_SPEED 調整過的排程時間，
不是原始 target_time）如果晚於下一次 replan 觸發的時間，代表它被蓋掉了、
機器人從來沒有真的去過那裡，直接丟棄、不列入誤差計算。

對每個「有被執行」的預測點，用它的 actual_time 去 <experiment>_measured.csv
記錄的實際軌跡（TF 量測到的 TCP 位姿，原生時間戳、原生更新頻率，不是補值出來
的）上線性插值，找出機器人在那個時刻真正的位置，兩者相減就是這個點的追蹤誤差。

姿態不直接比較四元數分量（q 和 -q 代表同一個旋轉，逐分量比較沒有物理意義），
而是先轉成 UR 慣用的旋轉向量 (rx, ry, rz)（pose_utils.quaternion_to_rotation_vector），
degrees 表示，所以最終是 X/Y/Z/RX/RY/RZ 六軸都可以直接畫時序曲線比較。

指標：
  - 六軸（X/Y/Z 用 mm，RX/RY/RZ 用 deg）個別的 MAE / MSE / RMSE / R²。
  - 整體位置追蹤誤差（歐氏距離）跟姿態追蹤誤差（四元數測地線角度）的
    mean/std/max，當作 diffusion policy／UMI 真實機器人部署常用的整體指標參考。

輸出兩張圖（圖上文字全部英文，中文字型顯示不出來）：
  - <experiment>_axes_overlay.png：3x2，X/Y/Z/RX/RY/RZ 六軸各一張子圖，
    predicted 跟 actual 疊在同一張子圖上比較。
  - <experiment>_axes_error.png：3x2，六軸各一張子圖，畫 predicted-actual 的
    誤差隨時間變化，子圖標題附上該軸的 MAE/RMSE/MSE/R2。

用法：
  ros2 run ur_control analyze_trajectory <experiment_name>
或不帶參數，直接分析下面 EXPERIMENT_NAME 這個預設值。
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # 只存檔不開視窗，避免沒有 display 的環境（例如純終端機）跳錯
import matplotlib.pyplot as plt

from ur_control.pose_utils import quaternion_to_rotation_vector

AXIS_NAMES = ("X", "Y", "Z", "RX", "RY", "RZ")
AXIS_UNITS = ("mm", "mm", "mm", "deg", "deg", "deg")

# =============================================================================
# 可修改變數
# =============================================================================
# 沒有從命令列傳參數時，預設分析這個實驗名稱（要跟 dp_control_ros2.py 裡
# EXPERIMENT_NAME 對上）。
EXPERIMENT_NAME = "dp_control_ros2_test"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _load_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [{k: float(v) for k, v in row.items()} for row in reader]


def _quat_geodesic_angle_deg(q1, q2) -> float:
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _quat_to_rotvec_deg(q):
    rx, ry, rz = quaternion_to_rotation_vector(*q)
    return np.degrees((rx, ry, rz))


def _six_axis_matrix(positions_m, orientations):
    """(x,y,z) [m] + 四元數 -> (X,Y,Z) [mm] + (RX,RY,RZ) [deg]，六欄的矩陣，
    方便逐軸畫圖/算指標。"""
    positions_mm = np.asarray(positions_m) * 1000.0
    rotvecs_deg = np.array([_quat_to_rotvec_deg(q) for q in orientations])
    return np.hstack([positions_mm, rotvecs_deg])


def _interp_measured(measured_rows, times, t: float):
    """在依 t 排序好的 measured_rows 裡線性插值出時間 t 當下的位置/姿態。"""
    idx = np.searchsorted(times, t)
    if idx <= 0:
        row = measured_rows[0]
    elif idx >= len(measured_rows):
        row = measured_rows[-1]
    else:
        t0, t1 = times[idx - 1], times[idx]
        row0, row1 = measured_rows[idx - 1], measured_rows[idx]
        ratio = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        position = tuple(
            row0[f"pos_{axis}"] + ratio * (row1[f"pos_{axis}"] - row0[f"pos_{axis}"])
            for axis in ("x", "y", "z"))
        # 姿態不做插值，取時間上比較近的樣本就好：measured.csv 的原生更新頻率
        # 通常還是遠比 ACTION_STEP_DT（通常 0.1s）密，最近點跟真正插值的姿態
        # 差異可以忽略。
        nearest_row = row0 if (t - t0) <= (t1 - t) else row1
        orientation = (nearest_row["quat_x"], nearest_row["quat_y"],
                        nearest_row["quat_z"], nearest_row["quat_w"])
        return position, orientation
    position = (row["pos_x"], row["pos_y"], row["pos_z"])
    orientation = (row["quat_x"], row["quat_y"], row["quat_z"], row["quat_w"])
    return position, orientation


# ground truth 的標準差如果比這個還小（不管是 mm 還是 deg），代表這條軸實際上
# 幾乎沒在動，R2 的分母趨近 0，算出來的值不是 nan 就是隨機發散的巨大數字
# （浮點誤差主導），兩種都不是「這條軸很難預測」的意思，直接當作 N/A 比較誠實。
R2_MIN_STD = 1e-3


def _regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    std_true = float(np.std(y_true))
    r2 = (1.0 - ss_res / ss_tot) if std_true >= R2_MIN_STD else float("nan")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def main():
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else EXPERIMENT_NAME

    predicted_rows = _load_csv(DATA_DIR / f"{experiment_name}_predicted.csv")
    boundary_rows = _load_csv(DATA_DIR / f"{experiment_name}_replan_boundaries.csv")
    measured_rows = _load_csv(DATA_DIR / f"{experiment_name}_measured.csv")
    measured_rows.sort(key=lambda row: row["t"])
    measured_times = [row["t"] for row in measured_rows]

    if not predicted_rows or not measured_rows:
        print(f"'{experiment_name}' 底下的 CSV 是空的，檢查實驗有沒有真的跑完、"
              f"EXPERIMENT_NAME 有沒有打對。(DATA_DIR={DATA_DIR})")
        return

    next_trigger_time = {int(row["replan_index"]): row["trigger_time"] for row in boundary_rows}
    run_end_time = measured_times[-1]

    matched = []
    for row in predicted_rows:
        replan_index = int(row["replan_index"])
        next_time = next_trigger_time.get(replan_index + 1, run_end_time)
        if row["actual_time"] >= next_time:
            continue  # 這個點在被走到之前就被下一次 replan 蓋掉了，機器人沒真的去過
        predicted_position = (row["pos_x"], row["pos_y"], row["pos_z"])
        predicted_orientation = (row["quat_x"], row["quat_y"], row["quat_z"], row["quat_w"])
        actual_position, actual_orientation = _interp_measured(
            measured_rows, measured_times, row["actual_time"])
        matched.append((predicted_position, predicted_orientation,
                         actual_position, actual_orientation, row["actual_time"]))

    if not matched:
        print("沒有任何「有推論且有被執行」的點可以比對——通常代表 replan 太頻繁、"
              "action horizon 太短，或是實驗中斷得太早。")
        return

    predicted_positions = np.array([m[0] for m in matched])
    predicted_orientations = [m[1] for m in matched]
    actual_positions = np.array([m[2] for m in matched])
    actual_orientations = [m[3] for m in matched]
    actual_times = np.array([m[4] for m in matched])

    print(f"共 {len(predicted_rows)} 個成功求解 IK 的預測點，其中 {len(matched)} 個"
          f"（{100.0 * len(matched) / len(predicted_rows):.1f}%）有被實際執行到，"
          f"其餘 {len(predicted_rows) - len(matched)} 個在被走到之前就被下一次 replan 蓋掉了")

    predicted_six = _six_axis_matrix(predicted_positions, predicted_orientations)
    actual_six = _six_axis_matrix(actual_positions, actual_orientations)

    axis_metrics = {}
    for i, axis in enumerate(AXIS_NAMES):
        axis_metrics[axis] = _regression_metrics(actual_six[:, i], predicted_six[:, i])

    position_errors = np.linalg.norm(predicted_positions - actual_positions, axis=1)
    orientation_errors_deg = np.array([
        _quat_geodesic_angle_deg(p, a) for p, a in zip(predicted_orientations, actual_orientations)])

    print("\n=== 逐軸指標（實際值當作 ground truth，預測值當作 prediction；X/Y/Z 用 mm，RX/RY/RZ 用 deg） ===")
    for axis, metrics in axis_metrics.items():
        unit = AXIS_UNITS[AXIS_NAMES.index(axis)]
        # 圓是平面的，某些軸（例如 z、姿態）幾乎不動時 ground truth 幾乎沒有變異數，
        # R2 數學上沒有意義（分母趨近 0），這種情況印出來提醒一下，而不是丟一個
        # 看起來像 bug 的 nan。
        r2_text = (f"{metrics['R2']:.4f}" if metrics['R2'] == metrics['R2']
                   else "N/A（這軸幾乎不變，R2 沒有意義）")
        print(f"  {axis}: MAE={metrics['MAE']:.4f}{unit}  MSE={metrics['MSE']:.4f}{unit}^2  "
              f"RMSE={metrics['RMSE']:.4f}{unit}  R2={r2_text}")

    print("\n=== 整體位置追蹤誤差（歐氏距離，DP/UMI 真實機器人部署常用指標） ===")
    print(f"  mean={position_errors.mean() * 1000:.3f}mm  std={position_errors.std() * 1000:.3f}mm  "
          f"max={position_errors.max() * 1000:.3f}mm")

    print("\n=== 整體姿態追蹤誤差（四元數測地線角度） ===")
    print(f"  mean={orientation_errors_deg.mean():.3f}deg  std={orientation_errors_deg.std():.3f}deg  "
          f"max={orientation_errors_deg.max():.3f}deg")

    summary_path = DATA_DIR / f"{experiment_name}_metrics.txt"
    with open(summary_path, "w") as f:
        f.write(f"matched_points: {len(matched)} / {len(predicted_rows)}\n")
        for axis, metrics in axis_metrics.items():
            f.write(f"axis_{axis} [{AXIS_UNITS[AXIS_NAMES.index(axis)]}]: {metrics}\n")
        f.write(f"position_error_m: mean={position_errors.mean():.6f} std={position_errors.std():.6f} "
                f"max={position_errors.max():.6f}\n")
        f.write(f"orientation_error_deg: mean={orientation_errors_deg.mean():.4f} "
                f"std={orientation_errors_deg.std():.4f} max={orientation_errors_deg.max():.4f}\n")
    print(f"\n指標已存到: {summary_path}")

    # ------------------------------------------------------------------
    # Figure 1: 每軸 predicted vs actual 疊在一起的時序曲線
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    for i, (axis, unit) in enumerate(zip(AXIS_NAMES, AXIS_UNITS)):
        ax = axes1.flat[i]
        ax.plot(actual_times, predicted_six[:, i], "y.-", label="predicted", markersize=3, linewidth=1)
        ax.plot(actual_times, actual_six[:, i], "g.-", label="actual", markersize=3, linewidth=1)
        ax.set_ylabel(f"{axis} [{unit}]")
        ax.set_title(f"{axis}: predicted vs actual")
        ax.grid(True)
        ax.legend(fontsize=8)
        if i >= 4:
            ax.set_xlabel("time [s]")
    fig1.suptitle(f"{experiment_name}: predicted vs actual trajectory, per axis")
    fig1.tight_layout()
    overlay_plot_path = DATA_DIR / f"{experiment_name}_axes_overlay.png"
    fig1.savefig(overlay_plot_path, dpi=150)
    print(f"六軸疊圖已存到: {overlay_plot_path}")

    # ------------------------------------------------------------------
    # Figure 2: 每軸的追蹤誤差（predicted - actual）時序曲線
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    for i, (axis, unit) in enumerate(zip(AXIS_NAMES, AXIS_UNITS)):
        ax = axes2.flat[i]
        axis_error = predicted_six[:, i] - actual_six[:, i]
        metrics = axis_metrics[axis]
        r2_text = f"{metrics['R2']:.3f}" if metrics["R2"] == metrics["R2"] else "N/A"
        ax.plot(actual_times, axis_error, "r.-", markersize=3, linewidth=1)
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_ylabel(f"error [{unit}]")
        ax.set_title(f"{axis}: MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} "
                      f"MSE={metrics['MSE']:.4f} R2={r2_text}", fontsize=9)
        ax.grid(True)
        if i >= 4:
            ax.set_xlabel("time [s]")
    fig2.suptitle(f"{experiment_name}: tracking error (predicted - actual), per axis")
    fig2.tight_layout()
    error_plot_path = DATA_DIR / f"{experiment_name}_axes_error.png"
    fig2.savefig(error_plot_path, dpi=150)
    print(f"六軸誤差圖已存到: {error_plot_path}")


if __name__ == "__main__":
    main()
