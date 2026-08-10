"""記錄 dp_control_*.py 的三條獨立時間序列，供 examples/analyze_trajectory.py
事後比對「模型推論輸出」跟「機器人實際執行軌跡」是否相符。

三條序列刻意分開、各自帶自己的原生時間戳，不在這裡對齊/插值/補值：
  - predicted：模型（或 fake 模型）的推論輸出，已經成功求解 IK 的點才記錄。
  - commanded：控制迴圈算出來、真的送給 controller 的 setpoint（Cartesian，
    從 PoseTrajectoryInterpolator 查出來，跟送給 joint_interp 的關節目標是
    同一組排程算出來的，只是換一個空間表示，方便跟 predicted/measured 直接比）。
  - measured：從機器人（TF/joint_states）量測回來的實際位姿。

之前 commanded 跟 measured 混在同一份 executed.csv 裡、用高頻迴圈的取樣時刻
硬記成一筆，如果來源本身更新比取樣慢，就會在 log 裡出現重複值的階梯（ZOH），
看起來像機器人在頓、其實只是記錄方式造成的假象。拆開之後，每份 CSV 誠實記錄
自己來源真正的更新頻率，不足的地方就是不足，不用 logger 自己補。
"""

import csv
from pathlib import Path


class TrajectoryLogger:
    def __init__(self):
        self._predicted_rows = []
        self._replan_boundary_rows = []
        self._commanded_rows = []
        self._measured_rows = []

    def log_predicted_point(self, replan_index: int, t_obs: float, target_time: float,
                             actual_time: float, position, orientation) -> None:
        """記錄一個「已經成功求出 IK、真的被排進 joint_interp」的預測點。
        IK 求不出來的點不要呼叫這個——那種點從來沒被排進關節軌跡，之後也不會
        被機器人走到，混進來會讓分析誤判成「有推論但沒執行」。"""
        self._predicted_rows.append((
            replan_index, t_obs, target_time, actual_time,
            position[0], position[1], position[2],
            orientation[0], orientation[1], orientation[2], orientation[3]))

    def log_replan_trigger(self, replan_index: int, trigger_time: float) -> None:
        """記錄第 replan_index 次 replan 是在 trigger_time 這個時刻觸發的。
        分析時用「下一次 replan 的 trigger_time」當作分界，判斷某個預測點是否
        在被走到之前就被新的 chunk 蓋掉了。"""
        self._replan_boundary_rows.append((replan_index, trigger_time))

    def log_commanded_sample(self, t: float, position, orientation) -> None:
        """記錄 t 這個時刻（控制迴圈算出這個值、真的發布出去的當下）送給
        controller 的 setpoint。t 是這個值產生的時刻，不是寫檔時刻。"""
        self._commanded_rows.append((
            t, position[0], position[1], position[2],
            orientation[0], orientation[1], orientation[2], orientation[3]))

    def log_measured_sample(self, t: float, position, orientation) -> None:
        """記錄 t 這個時刻從機器人量測回來（TF/joint_states）的實際 TCP 位姿。"""
        self._measured_rows.append((
            t, position[0], position[1], position[2],
            orientation[0], orientation[1], orientation[2], orientation[3]))

    def has_data(self) -> bool:
        return bool(self._measured_rows)

    def save(self, output_dir, experiment_name: str):
        """存成四個 CSV：<experiment_name>_predicted.csv / _replan_boundaries.csv /
        _commanded.csv / _measured.csv。回傳這四個檔案的路徑。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        predicted_path = output_dir / f"{experiment_name}_predicted.csv"
        with open(predicted_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["replan_index", "t_obs", "target_time", "actual_time",
                              "pos_x", "pos_y", "pos_z",
                              "quat_x", "quat_y", "quat_z", "quat_w"])
            writer.writerows(self._predicted_rows)

        boundaries_path = output_dir / f"{experiment_name}_replan_boundaries.csv"
        with open(boundaries_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["replan_index", "trigger_time"])
            writer.writerows(self._replan_boundary_rows)

        commanded_path = output_dir / f"{experiment_name}_commanded.csv"
        with open(commanded_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "pos_x", "pos_y", "pos_z",
                              "quat_x", "quat_y", "quat_z", "quat_w"])
            writer.writerows(self._commanded_rows)

        measured_path = output_dir / f"{experiment_name}_measured.csv"
        with open(measured_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "pos_x", "pos_y", "pos_z",
                              "quat_x", "quat_y", "quat_z", "quat_w"])
            writer.writerows(self._measured_rows)

        return predicted_path, boundaries_path, commanded_path, measured_path
