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

## 出廠校正（`load_calibration()`）

`ur_analytic_ik` 是 IKFast 針對「標稱」UR5e 參數（每一型號共用的理論值）預先
產生的封閉解，沒有介面可以餵入這台手臂實際出廠校正過的 DH 參數——每支 UR
手臂出廠時都會量測、寫入一份跟理論值有微小差異的校正檔（教導器 Installation
分頁可以匯出），MoveIt 用的 KDL/TRAC-IK 是拿這份校正檔即時建構運動學模型，
所以比較準；這個模組預設用的標稱參數沒有這份資訊。

實測量過這個差異的量級（拿這個模組自己重建的 DH 鏈分別代入標稱/校正參數比較，
不是用 MoveIt 側面猜的）：home pose 附近、六軸圓測試會用到的姿態範圍，位置差距
穩定在 0.7~0.75mm；全關節範圍隨機取樣，mean 1.05mm、max 1.87mm。這個量級不需要
重新產生一份 IKFast（那需要 OpenRAVE toolchain，離線、耗時，而且每次校正檔更新
都要重跑），用線性化修正就足夠：`load_calibration()` 載入後，`forward_kinematics()`
改用校正參數直接算（`_chain_forward_kinematics()`，跟 IKFast 用同一套 URDF 鏈
公式，已驗證在標稱參數下跟 `ur_analytic_ik.ur5e.forward_kinematics()` 逐位元
吻合），`solve()` 則是先用標稱 IKFast 解出一個很接近的初始解，再用校正模型的
數值 Jacobian 做幾次 Newton 修正貼齊目標姿態——因為初始解跟真正答案本來就只
差不到 2mm，這個修正兩三次迭代就能收斂到微米等級，不需要真的重新解一次非線性
IK。沒呼叫 `load_calibration()` 就完全沿用原本的標稱行為，向後相容。
"""

import numpy as np
import yaml
from ur_analytic_ik import ur5e as _ur_model

from ur_control.pose_utils import (
    base_link_to_ur_base,
    ur_base_to_base_link,
    quaternion_to_rotation_vector,
    quaternion_multiply,
    quaternion_conjugate,
    rotation_vector_to_quaternion,
)

UR_TYPE = "ur5e"

# solve() 內建的 round-trip 驗證容許誤差：解出來的關節角透過 forward_kinematics()
# 算回去的姿態，跟原本要求的目標姿態差距要在這個範圍內，才會被當成有效解回傳。
POSITION_TOLERANCE_M = 0.0005
ORIENTATION_TOLERANCE_DEG = 0.1

# UR 出廠校正檔的 6 個關節 frame，由 base 到 tool 的順序（跟 URDF 運動學鏈一致）。
_CALIBRATION_FRAMES = ["shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3"]

# Newton 修正的收斂設定：誤差量級本來就只有 mm/mrad 等級，線性化非常準確，
# 給寬鬆的迭代上限只是防呆，正常應該 1~2 次就收斂到遠小於容許誤差。
_CALIBRATION_MAX_ITERS = 5
_CALIBRATION_POSITION_TOLERANCE_M = 1e-8
_CALIBRATION_ORIENTATION_TOLERANCE_RAD = 1e-8
_CALIBRATION_JACOBIAN_EPS = 1e-6

_calibration = None                       # None = 沿用 ur_analytic_ik 內建標稱參數
_calibration_static_transforms = None     # load_calibration() 後：6 個快取好的 4x4 靜態轉換矩陣


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


def _rpy_to_matrix(roll, pitch, yaw) -> np.ndarray:
    """URDF 的 extrinsic RPY 慣例：R = Rz(yaw) * Ry(pitch) * Rx(roll)。"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _static_transform(frame) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(frame['roll'], frame['pitch'], frame['yaw'])
    T[:3, 3] = [frame['x'], frame['y'], frame['z']]
    return T


def _joint_rotation_z(theta) -> np.ndarray:
    T = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


def _chain_forward_kinematics(static_transforms, joint_positions) -> np.ndarray:
    """用校正參數重建的 UR 運動學鏈（UR Base 座標系），逐一 frame 疊乘：
    每個 frame 是「固定 offset（校正檔的 x/y/z/roll/pitch/yaw，`load_calibration()`
    時就算好、快取起來的 4x4 矩陣，這裡不重算三角函數)再繞自己局部 Z 軸轉關節
    角」。跟 `ur_analytic_ik.ur5e.forward_kinematics()` 用同一套 UR/URDF 慣例，
    已經在標稱參數下驗證過逐位元吻合（見本檔開頭「出廠校正」小節），代入校正
    參數後就是這台手臂真正的運動學模型。

    `static_transforms` 固定用預先快取好的 6 個 4x4 矩陣（`_calibration_static_
    transforms`），不是每次重算——這個函式在 Newton 修正的數值 Jacobian 裡一次
    solve() 會被呼叫十幾次，重算三角函數是主要的效能瓶頸（實測快取後單次呼叫
    從 83us 降到個位數 us），快取後才跟得上 cartesian 模式的高頻 per-tick 呼叫。
    """
    T = np.eye(4)
    for static, theta in zip(static_transforms, joint_positions):
        T = T @ static @ _joint_rotation_z(theta)
    return T


def load_calibration(yaml_path) -> None:
    """載入 UR 出廠校正檔（教導器 Installation -> Calibration 匯出的格式：
    `kinematics.{shoulder,upper_arm,forearm,wrist_1,wrist_2,wrist_3}.{x,y,z,
    roll,pitch,yaw}`）。載入後 forward_kinematics()/solve() 都改用校正後的
    模型，細節見本檔開頭「出廠校正」小節。"""
    global _calibration, _calibration_static_transforms
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    kinematics = data["kinematics"]
    _calibration = {name: kinematics[name] for name in _CALIBRATION_FRAMES}
    _calibration_static_transforms = [_static_transform(_calibration[name]) for name in _CALIBRATION_FRAMES]


def is_calibrated() -> bool:
    """load_calibration() 是否已經成功載入過一份校正檔。"""
    return _calibration is not None


def _numeric_jacobian(joint_positions) -> np.ndarray:
    """6x6 數值 Jacobian：前三列是位置對關節角的偏導，後三列是姿態誤差（轉成
    旋轉向量）對關節角的偏導，跟 _refine_solution_with_calibration() 的誤差
    向量定義一致，才能直接拿去解 Newton 步。"""
    jacobian = np.zeros((6, 6))
    base_matrix = _chain_forward_kinematics(_calibration_static_transforms, joint_positions)
    base_quat = _matrix_to_quaternion(base_matrix[:3, :3])
    for i in range(6):
        perturbed = list(joint_positions)
        perturbed[i] += _CALIBRATION_JACOBIAN_EPS
        perturbed_matrix = _chain_forward_kinematics(_calibration_static_transforms, perturbed)
        jacobian[:3, i] = (perturbed_matrix[:3, 3] - base_matrix[:3, 3]) / _CALIBRATION_JACOBIAN_EPS
        perturbed_quat = _matrix_to_quaternion(perturbed_matrix[:3, :3])
        delta_quat = quaternion_multiply(tuple(perturbed_quat), quaternion_conjugate(tuple(base_quat)))
        jacobian[3:, i] = np.array(quaternion_to_rotation_vector(*delta_quat)) / _CALIBRATION_JACOBIAN_EPS
    return jacobian


def _refine_solution_with_calibration(target_matrix, seed_solution):
    """標稱 IKFast 解出來的 seed_solution 離真正答案通常只差不到 2mm 等效的
    關節角（見本檔開頭「出廠校正」小節量測結果），用校正模型的數值 Jacobian
    做幾次 Newton 修正貼齊 target_matrix。收斂不了（數值奇異、或超過迭代
    上限）就回傳 None，呼叫端視同無解，不送出不可信的關節角。"""
    q = np.array(seed_solution, dtype=float)
    target_position = target_matrix[:3, 3]
    target_quat = _matrix_to_quaternion(target_matrix[:3, :3])
    for _ in range(_CALIBRATION_MAX_ITERS):
        current_matrix = _chain_forward_kinematics(_calibration_static_transforms, q)
        position_error = target_position - current_matrix[:3, 3]
        current_quat = _matrix_to_quaternion(current_matrix[:3, :3])
        error_quat = quaternion_multiply(tuple(target_quat), quaternion_conjugate(tuple(current_quat)))
        rotation_error = np.array(quaternion_to_rotation_vector(*error_quat))
        if (np.linalg.norm(position_error) < _CALIBRATION_POSITION_TOLERANCE_M
                and np.linalg.norm(rotation_error) < _CALIBRATION_ORIENTATION_TOLERANCE_RAD):
            return q.tolist()
        error = np.concatenate([position_error, rotation_error])
        jacobian = _numeric_jacobian(q)
        try:
            delta = np.linalg.solve(jacobian, error)
        except np.linalg.LinAlgError:
            return None
        q = q + delta
    return None


def forward_kinematics(joint_positions):
    """6 個關節角（跟 self.joint_names 同樣順序）-> base_link 座標系的
    (position, quaternion)，tip_link 姿態（不含 TCP offset，呼叫端視需要自己疊加，
    比照 URArmNode._tip_pose_to_tcp_pose()）。微秒等級，跟 solve() 用同一組已經
    驗證過的座標系轉換（見本檔開頭註解）。載入過 load_calibration() 就用校正後的
    模型算，否則用 ur_analytic_ik 內建的標稱參數。"""
    if _calibration is not None:
        T = _chain_forward_kinematics(_calibration_static_transforms, joint_positions)
    else:
        T = _ur_model.forward_kinematics(*joint_positions)
    return _ur_matrix_to_base_link_pose(T)


def solve(position_xyz, orientation_xyzw, seed_positions):
    """base_link 座標系下的目標姿態（tip_link，TCP offset 由呼叫端先處理好）+
    seed（6 個關節角，跟 self.joint_names 同樣順序）-> 6 個關節角，或 None
    （無解，或 round-trip 驗證沒過，兩種都代表這個目標不可信）。

    不做碰撞檢查——這是刻意的取捨，碰撞檢查需要 MoveIt，跟這個函式存在的目的
    （避免 service round-trip）互相矛盾。高頻路徑上使用這個函式的呼叫端要自行
    確保軌跡本來就在安全空間內（例如已知的測試圓、事先驗證過的工作區域）。

    載入過 load_calibration() 的話，這裡先用標稱 IKFast 解出初始解，再用校正
    模型的 Newton 修正貼齊目標（見本檔開頭「出廠校正」小節），round-trip 驗證
    也改用校正後的 forward kinematics 檢查，確保回傳的關節角是真的能讓這台
    手臂（不是理論上的標稱手臂）到達目標姿態。
    """
    target_matrix = _base_link_pose_to_ur_matrix(position_xyz, orientation_xyzw)
    solutions = _ur_model.inverse_kinematics_closest(target_matrix, *seed_positions)
    if not solutions:
        return None
    solution = solutions[0]

    if _calibration is not None:
        refined = _refine_solution_with_calibration(target_matrix, solution)
        if refined is None:
            return None
        solution = refined
        check_matrix = _chain_forward_kinematics(_calibration_static_transforms, solution)
    else:
        check_matrix = _ur_model.forward_kinematics(*solution)

    position_error = float(np.linalg.norm(check_matrix[:3, 3] - target_matrix[:3, 3]))
    check_quat = _matrix_to_quaternion(check_matrix[:3, :3])
    target_quat = _matrix_to_quaternion(target_matrix[:3, :3])
    dot = min(1.0, max(-1.0, abs(float(np.dot(check_quat, target_quat)))))
    orientation_error_deg = float(np.degrees(2.0 * np.arccos(dot)))

    if position_error > POSITION_TOLERANCE_M or orientation_error_deg > ORIENTATION_TOLERANCE_DEG:
        return None

    return [float(v) for v in solution]
