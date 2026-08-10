"""UR 閉式解析 IK 包裝，取代 dp_control_ros2.py 高頻路徑上原本每個 chunk 呼叫最多
16 次的 MoveIt `/compute_ik` service round-trip。

背景：實測（`dp_control_ros2_RealRobot_test`）顯示控制迴圈 dt 的 24-42ms 尖峰
100% 落在 replan trigger 附近，根因是這些 service round-trip 同步卡在控制迴圈裡。
PickNik 的 `ur-analytic-ik`（pip 套件，IKFast 產生的解析解）把同一次求解從
「service round-trip、上限 50ms」降到「in-process 函式呼叫、微秒等級」，直接
讓尖峰的根因消失，不需要把 IK 搬到背景執行緒。

**這個模組存在的全部理由是座標系轉換**：`ur_analytic_ik` 操作在 UR 控制器原生的
"Base" 座標系（跟教導器/RTDE 一致），不是 ROS 這邊全程使用的 `base_link`
（REP-103，繞 Z 跟 Base 差 180 度——跟這個專案稍早在 `pose_utils.py` 處理過的
旋轉向量正負號問題是同一組轉換，不是重新發明）。已經用真實機器人（2026-08）
拿目前關節角同時算 `ur_analytic_ik.forward_kinematics()` 和呼叫 MoveIt
`/compute_fk`（`base_link -> tool0`）比對過：套用既有、已驗證過的
`pose_utils.ur_base_to_base_link()` 之後兩者完全吻合（position diff 0.0mm，
orientation diff < 0.001 度）。同時也驗證過 `inverse_kinematics_closest()`
在 8 組解裡正確選出離 seed 最近的分支、且每組解都能透過 `forward_kinematics()`
round-trip 回原本的目標姿態（誤差 0.000000mm）。

安全機制：`solve()` 每次求解後都在本地端用 `forward_kinematics()` round-trip
驗證解真的能到達目標姿態，容許誤差之外的解直接視為求解失敗（回傳 None）——
呼叫端（`URArmNode.compute_ik_analytic()`）沿用既有的「這個點跳過」邏輯，不會
把不可信的關節值送給機器人。

依賴：`pip3 install --user ur-analytic-ik`（不是 ROS package，見 package.xml 註解）。
目前只支援 ur5e——如果之後換機型，把 UR_TYPE 改掉、換一顆 ur_analytic_ik 底下對應
的子模組即可，介面不用動。
"""

import numpy as np
from ur_analytic_ik import ur5e as _ur_model

from ur_control.pose_utils import (
    base_link_to_ur_base,
    ur_base_to_base_link,
    quaternion_to_rotation_vector,
    rotation_vector_to_quaternion,
)

UR_TYPE = "ur5e"

# solve() 內建的 round-trip 驗證容許誤差：解出來的關節角透過 forward_kinematics()
# 算回去的姿態，跟原本要求的目標姿態差距要在這個範圍內，才會被當成有效解回傳。
POSITION_TOLERANCE_M = 0.0005
ORIENTATION_TOLERANCE_DEG = 0.1


def _quaternion_to_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_quaternion(R) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw])


def _base_link_pose_to_ur_matrix(position_xyz, orientation_xyzw) -> np.ndarray:
    """base_link 座標系的 (position, quaternion) -> UR Base 座標系的 4x4 齊次矩陣
    （ur_analytic_ik 要的輸入格式）。"""
    rx, ry, rz = quaternion_to_rotation_vector(*orientation_xyzw)
    ur_position, ur_rotvec = base_link_to_ur_base(tuple(position_xyz), (rx, ry, rz))
    T = np.eye(4)
    T[:3, :3] = _quaternion_to_matrix(rotation_vector_to_quaternion(*ur_rotvec))
    T[:3, 3] = ur_position
    return T


def _ur_matrix_to_base_link_pose(T):
    """UR Base 座標系的 4x4 齊次矩陣 -> base_link 座標系的 (position, quaternion)。"""
    ur_quat = _matrix_to_quaternion(T[:3, :3])
    ur_rotvec = quaternion_to_rotation_vector(*ur_quat)
    return ur_base_to_base_link(tuple(T[:3, 3]), ur_rotvec)


def forward_kinematics(joint_positions):
    """6 個關節角（跟 self.joint_names 同樣順序）-> base_link 座標系的
    (position, quaternion)，tip_link 姿態（不含 TCP offset，呼叫端視需要自己疊加，
    比照 URArmNode._tip_pose_to_tcp_pose()）。微秒等級，跟 solve() 用同一組已經
    驗證過的座標系轉換（見本檔開頭註解）。"""
    T = _ur_model.forward_kinematics(*joint_positions)
    return _ur_matrix_to_base_link_pose(T)


def solve(position_xyz, orientation_xyzw, seed_positions):
    """base_link 座標系下的目標姿態（tip_link，TCP offset 由呼叫端先處理好）+
    seed（6 個關節角，跟 self.joint_names 同樣順序）-> 6 個關節角，或 None
    （無解，或 round-trip 驗證沒過，兩種都代表這個目標不可信）。

    不做碰撞檢查——這是刻意的取捨，碰撞檢查需要 MoveIt，跟這個函式存在的目的
    （避免 service round-trip）互相矛盾。高頻路徑上使用這個函式的呼叫端要自行
    確保軌跡本來就在安全空間內（例如已知的測試圓、事先驗證過的工作區域）。
    """
    target_matrix = _base_link_pose_to_ur_matrix(position_xyz, orientation_xyzw)
    solutions = _ur_model.inverse_kinematics_closest(target_matrix, *seed_positions)
    if not solutions:
        return None
    solution = solutions[0]

    check_matrix = _ur_model.forward_kinematics(*solution)
    position_error = float(np.linalg.norm(check_matrix[:3, 3] - target_matrix[:3, 3]))
    check_quat = _matrix_to_quaternion(check_matrix[:3, :3])
    target_quat = _matrix_to_quaternion(target_matrix[:3, :3])
    dot = min(1.0, max(-1.0, abs(float(np.dot(check_quat, target_quat)))))
    orientation_error_deg = float(np.degrees(2.0 * np.arccos(dot)))

    if position_error > POSITION_TOLERANCE_M or orientation_error_deg > ORIENTATION_TOLERANCE_DEG:
        return None

    return [float(v) for v in solution]
