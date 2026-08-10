#!/usr/bin/env python3
"""事後分析：`dp_control_*.py` 錄下的三條時間序列（predicted/commanded/measured）
互相比對，拆成兩組獨立比較——predicted vs actual（整條 pipeline 的總和誤差）跟
commanded vs measured（純低階追蹤誤差，排除規劃面因素）。兩組分開看才不會把
「規劃誤差」跟「追蹤誤差」混為一談，細節跟輸出檔案說明見 README「軌跡分析輸出」。

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
    """標準轉換：quaternion_to_rotation_vector() 內部 force w>=0，永遠回傳角度在
    [0, 180°] 的最短旋轉代表。單獨看一個姿態時很好；但逐點比較兩條序列、或畫
    時間序列曲線時不能直接用——旋轉角接近 180° 時，這個「force w>=0」的獨立正規
    化會讓物理上幾乎沒變的姿態在相鄰時刻之間突然翻到相反的軸/角度代表，逐軸相
    減冒出幾百度的假跳變。要處理這個問題見 _unwrap_quaternions()。"""
    rx, ry, rz = quaternion_to_rotation_vector(*q)
    return np.degrees((rx, ry, rz))


def _unwrap_quaternions(orientations):
    """依時間序列展開四元數的正負號連續性（q 和 -q 代表同一個旋轉）。

    記錄下來的原始四元數正負號是任意的（旋轉矩陣轉四元數這類運算內部依矩陣元素
    大小挑分支，物理上幾乎沒動的相鄰時刻，量測端可能因浮點雜訊就翻了正負號）。
    不展開的話，後續轉旋轉向量會在翻面那一刻冒出幾百度的假跳變。

    做法：整條序列依序走一遍，每一筆都跟前一筆展開後的結果比 dot，dot < 0 就翻
    正負號，處理完之後任兩個相鄰時刻的四元數保證連續（dot >= 0）。"""
    if not orientations:
        return []
    unwrapped = [tuple(orientations[0])]
    for q in orientations[1:]:
        prev = unwrapped[-1]
        dot = sum(a * b for a, b in zip(prev, q))
        unwrapped.append(tuple(q) if dot >= 0.0 else tuple(-c for c in q))
    return unwrapped


def _quat_to_rotvec_deg_unforced(q):
    """跟 _quat_to_rotvec_deg() 不同：不強制 w>=0，角度用原始 atan2(|xyz|, w)
    算，範圍 (0°, 360°)，跟著 q 本來的正負號走。**只能用在已經先用
    _unwrap_quaternions() 展開過時間連續性的序列上**——沒展開過的原始資料直接
    用這個，反而會讓 180° 附近的任意翻面問題更嚴重（那正是 force w>=0 原本要
    擋掉的事）。"""
    x, y, z, w = q
    axis_norm = float(np.sqrt(x * x + y * y + z * z))
    if axis_norm < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(axis_norm, w)
    return np.degrees(angle * np.array([x, y, z]) / axis_norm)


def _prepare_comparable_orientations(true_orientations, pred_orientations):
    """把 true/pred 兩條姿態序列變成可以直接逐點相減比較的形式：各自先在時間軸
    上展開連續性（見 _unwrap_quaternions()），再把 pred 整條序列一次性對齊到跟
    true 同一側（用所有配對點的平均 dot 判斷，不是逐點判斷——逐點判斷等於還沒
    展開就先比，會被還沒展開的雜訊帶偏）。回傳 (true_unwrapped, pred_unwrapped)，
    直接丟給 _quat_to_rotvec_deg_unforced() 轉旋轉向量即可安全逐軸相減。"""
    true_unwrapped = _unwrap_quaternions(true_orientations)
    pred_unwrapped = _unwrap_quaternions(pred_orientations)
    mean_dot = float(np.mean([
        sum(a * b for a, b in zip(t, p)) for t, p in zip(true_unwrapped, pred_unwrapped)]))
    if mean_dot < 0.0:
        pred_unwrapped = [tuple(-c for c in q) for q in pred_unwrapped]
    return true_unwrapped, pred_unwrapped


def _six_axis_matrix(positions_m, orientations, unforced: bool = False):
    """(x,y,z) [m] + 四元數 -> (X,Y,Z) [mm] + (RX,RY,RZ) [deg]，六欄的矩陣，
    方便逐軸畫圖/算指標。`unforced=True` 時用 _quat_to_rotvec_deg_unforced()
    （給已經展開過連續性的序列用）；預設用標準的 force w>=0 轉換。"""
    positions_mm = np.asarray(positions_m) * 1000.0
    convert = _quat_to_rotvec_deg_unforced if unforced else _quat_to_rotvec_deg
    rotvecs_deg = np.array([convert(q) for q in orientations])
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


def _analyze_pair(label: str, name_true: str, name_pred: str,
                   true_positions, true_orientations, pred_positions, pred_orientations,
                   times, experiment_name: str, plot_prefix: str):
    """對一組 (ground truth, prediction) 序列算六軸指標、印出來、存進
    metrics.txt、畫 overlay + error 兩張圖。回傳要寫進 metrics.txt 的文字區塊。"""
    true_positions = np.asarray(true_positions)
    pred_positions = np.asarray(pred_positions)

    # RX/RY/RZ 逐軸比較/畫圖前，兩條序列都先在時間軸上展開四元數連續性、互相
    # 對齊，見 _prepare_comparable_orientations() 說明——只影響逐軸指標/畫圖，
    # 不影響下面用 geodesic angle 算的整體姿態誤差（那個本來就不怕 q vs -q）。
    true_orientations_cmp, pred_orientations_cmp = _prepare_comparable_orientations(
        true_orientations, pred_orientations)
    true_six = _six_axis_matrix(true_positions, true_orientations_cmp, unforced=True)
    pred_six = _six_axis_matrix(pred_positions, pred_orientations_cmp, unforced=True)

    axis_metrics = {}
    for i, axis in enumerate(AXIS_NAMES):
        axis_metrics[axis] = _regression_metrics(true_six[:, i], pred_six[:, i])

    position_errors = np.linalg.norm(pred_positions - true_positions, axis=1)
    orientation_errors_deg = np.array([
        _quat_geodesic_angle_deg(p, a) for p, a in zip(pred_orientations, true_orientations)])

    print(f"\n=== [{label}] 逐軸指標（{name_true} 當 ground truth，{name_pred} 當 prediction；"
          f"X/Y/Z 用 mm，RX/RY/RZ 用 deg） ===")
    for axis, metrics in axis_metrics.items():
        unit = AXIS_UNITS[AXIS_NAMES.index(axis)]
        r2_text = (f"{metrics['R2']:.4f}" if metrics['R2'] == metrics['R2']
                   else "N/A（這軸幾乎不變，R2 沒有意義）")
        print(f"  {axis}: MAE={metrics['MAE']:.4f}{unit}  MSE={metrics['MSE']:.4f}{unit}^2  "
              f"RMSE={metrics['RMSE']:.4f}{unit}  R2={r2_text}")

    print(f"\n=== [{label}] 整體位置追蹤誤差（歐氏距離） ===")
    print(f"  mean={position_errors.mean() * 1000:.3f}mm  std={position_errors.std() * 1000:.3f}mm  "
          f"max={position_errors.max() * 1000:.3f}mm")

    print(f"\n=== [{label}] 整體姿態追蹤誤差（四元數測地線角度） ===")
    print(f"  mean={orientation_errors_deg.mean():.3f}deg  std={orientation_errors_deg.std():.3f}deg  "
          f"max={orientation_errors_deg.max():.3f}deg")

    metrics_text_lines = [f"[{label}] {name_true} vs {name_pred}, matched_points: {len(true_positions)}"]
    for axis, metrics in axis_metrics.items():
        metrics_text_lines.append(f"  axis_{axis} [{AXIS_UNITS[AXIS_NAMES.index(axis)]}]: {metrics}")
    metrics_text_lines.append(
        f"  position_error_m: mean={position_errors.mean():.6f} std={position_errors.std():.6f} "
        f"max={position_errors.max():.6f}")
    metrics_text_lines.append(
        f"  orientation_error_deg: mean={orientation_errors_deg.mean():.4f} "
        f"std={orientation_errors_deg.std():.4f} max={orientation_errors_deg.max():.4f}")

    # Figure 1: 每軸 true vs pred 疊在一起的時序曲線
    fig1, axes1 = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    for i, (axis, unit) in enumerate(zip(AXIS_NAMES, AXIS_UNITS)):
        ax = axes1.flat[i]
        ax.plot(times, pred_six[:, i], "y.-", label=name_pred, markersize=3, linewidth=1)
        ax.plot(times, true_six[:, i], "g.-", label=name_true, markersize=3, linewidth=1)
        ax.set_ylabel(f"{axis} [{unit}]")
        ax.set_title(f"{axis}: {name_pred} vs {name_true}")
        ax.grid(True)
        ax.legend(fontsize=8)
        if i >= 4:
            ax.set_xlabel("time [s]")
    fig1.suptitle(f"{experiment_name}: {name_pred} vs {name_true} trajectory, per axis")
    fig1.tight_layout()
    overlay_plot_path = DATA_DIR / f"{experiment_name}_{plot_prefix}_overlay.png"
    fig1.savefig(overlay_plot_path, dpi=150)
    plt.close(fig1)
    print(f"[{label}] 六軸疊圖已存到: {overlay_plot_path}")

    # Figure 2: 每軸的追蹤誤差（pred - true）時序曲線
    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    for i, (axis, unit) in enumerate(zip(AXIS_NAMES, AXIS_UNITS)):
        ax = axes2.flat[i]
        axis_error = pred_six[:, i] - true_six[:, i]
        metrics = axis_metrics[axis]
        r2_text = f"{metrics['R2']:.3f}" if metrics["R2"] == metrics["R2"] else "N/A"
        ax.plot(times, axis_error, "r.-", markersize=3, linewidth=1)
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_ylabel(f"error [{unit}]")
        ax.set_title(f"{axis}: MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} "
                      f"MSE={metrics['MSE']:.4f} R2={r2_text}", fontsize=9)
        ax.grid(True)
        if i >= 4:
            ax.set_xlabel("time [s]")
    fig2.suptitle(f"{experiment_name}: tracking error ({name_pred} - {name_true}), per axis")
    fig2.tight_layout()
    error_plot_path = DATA_DIR / f"{experiment_name}_{plot_prefix}_error.png"
    fig2.savefig(error_plot_path, dpi=150)
    plt.close(fig2)
    print(f"[{label}] 六軸誤差圖已存到: {error_plot_path}")

    return "\n".join(metrics_text_lines)


def main():
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else EXPERIMENT_NAME

    predicted_rows = _load_csv(DATA_DIR / f"{experiment_name}_predicted.csv")
    boundary_rows = _load_csv(DATA_DIR / f"{experiment_name}_replan_boundaries.csv")
    commanded_rows = _load_csv(DATA_DIR / f"{experiment_name}_commanded.csv")
    measured_rows = _load_csv(DATA_DIR / f"{experiment_name}_measured.csv")
    measured_rows.sort(key=lambda row: row["t"])
    measured_times = [row["t"] for row in measured_rows]

    if not predicted_rows or not measured_rows or not commanded_rows:
        print(f"'{experiment_name}' 底下的 CSV 是空的，檢查實驗有沒有真的跑完、"
              f"EXPERIMENT_NAME 有沒有打對。(DATA_DIR={DATA_DIR})")
        return

    metrics_blocks = []

    # -------------------------------------------------------------------
    # 比較 1：predicted vs actual（模型想要的 vs 機器人真正走到的，整條 pipeline
    # 的總和誤差）——只看「有推論、後來也真的被走到」的點。
    # -------------------------------------------------------------------
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
        print("沒有任何「有推論且有被執行」的點可以比對 predicted vs actual——"
              "通常代表 replan 太頻繁、action horizon 太短，或是實驗中斷得太早。")
    else:
        print(f"[predicted vs actual] 共 {len(predicted_rows)} 個成功求解 IK 的預測點，其中 "
              f"{len(matched)} 個（{100.0 * len(matched) / len(predicted_rows):.1f}%）有被實際執行到，"
              f"其餘 {len(predicted_rows) - len(matched)} 個在被走到之前就被下一次 replan 蓋掉了")
        predicted_positions = [m[0] for m in matched]
        predicted_orientations = [m[1] for m in matched]
        actual_positions = [m[2] for m in matched]
        actual_orientations = [m[3] for m in matched]
        actual_times = [m[4] for m in matched]
        metrics_blocks.append(_analyze_pair(
            "predicted vs actual", "actual", "predicted",
            actual_positions, actual_orientations, predicted_positions, predicted_orientations,
            actual_times, experiment_name, "predicted_vs_actual"))

    # -------------------------------------------------------------------
    # 比較 2：commanded vs measured（控制迴圈真的送出去的 setpoint vs 機器人
    # 真正走到的位置，純粹的低階追蹤誤差，排除規劃面因素）——每一筆 commanded
    # 都是真的送出去的指令，不需要用 replan boundary 過濾。
    # -------------------------------------------------------------------
    commanded_positions = [(row["pos_x"], row["pos_y"], row["pos_z"]) for row in commanded_rows]
    commanded_orientations = [
        (row["quat_x"], row["quat_y"], row["quat_z"], row["quat_w"]) for row in commanded_rows]
    commanded_times = [row["t"] for row in commanded_rows]
    measured_at_commanded = [
        _interp_measured(measured_rows, measured_times, t) for t in commanded_times]
    measured_positions_for_commanded = [m[0] for m in measured_at_commanded]
    measured_orientations_for_commanded = [m[1] for m in measured_at_commanded]

    print(f"\n[commanded vs measured] 共 {len(commanded_rows)} 筆送出去的 setpoint（控制迴圈每個 "
          f"tick 一筆，實際送出的指令，非規劃階段的預測點）")
    metrics_blocks.append(_analyze_pair(
        "commanded vs measured", "measured", "commanded",
        measured_positions_for_commanded, measured_orientations_for_commanded,
        commanded_positions, commanded_orientations,
        commanded_times, experiment_name, "commanded_vs_measured"))

    summary_path = DATA_DIR / f"{experiment_name}_metrics.txt"
    with open(summary_path, "w") as f:
        f.write("\n\n".join(metrics_blocks) + "\n")
    print(f"\n指標已存到: {summary_path}")


if __name__ == "__main__":
    main()
