#!/usr/bin/env python3
"""範例：手動輸入路徑點控制 UR 手臂（關節空間 + 末端點兩種方式）。

這支程式只負責「決定目標點」，實際通訊/控制都透過 URArmNode 完成。

重要注意事項（實測踩過的坑）：
  - 教導器上的 External Control 程式，External Control 節點必須包在一個
    Loop（無限迴圈）節點裡面並保持執行中，且面板要切到 Remote Control。
    沒包 Loop 的話，程式會在初始連線後自己跑完結束，driver 的
    controller_stopper 偵測到「程式沒在跑」就會停用 scaled_joint_trajectory_controller，
    之後所有目標都會被拒絕（log 會看到 "Controller is not running"）。
  - 座標系：base_link（本程式 / MoveIt 用的 ROS 慣例座標系）跟教導器 / URScript
    顯示的「Base」座標系原點相同，但繞 Z 軸差 180 度。用 pose_utils 的
    ur_base_to_base_link() / base_link_to_ur_base() 互轉，不要直接把教導器上的
    數字貼進 move_pose()。
  - 小範圍/直線移動用 move_pose_linear()（走 compute_cartesian_path 直線路徑）；
    move_pose()（自由空間 OMPL 規劃）沒設碰撞環境時，即使起訖姿態差異不大，
    也可能規劃出大幅度、不直覺的路徑。
  - 單位：關節角度/旋轉向量是 rad，位置是 m，時間是秒，速度/加速度縮放是 0~1。
"""

import math

import rclpy

from ur_control.ur_arm_node import URArmNode
from ur_control.pose_utils import (
    ur_base_to_base_link,
    base_link_to_ur_base,
    quaternion_to_rotation_vector,
)

# =============================================================================
# 可修改變數
# =============================================================================
NODE_NAME = "ur_waypoint_control_node"  # 這支程式使用的 node 名稱

# 手臂移動速度/加速度縮放 (0~1)，套用到本程式所有移動呼叫
# （若某次呼叫需要不同速度，也可以在該次呼叫用 velocity_scaling=... 個別覆寫）
SPEED_SCALING = 0.5
ACCELERATION_SCALING = 0.3


def main(args=None):
    rclpy.init(args=args)
    arm = URArmNode(
        node_name=NODE_NAME,
        velocity_scaling=SPEED_SCALING,
        acceleration_scaling=ACCELERATION_SCALING,
    )

    try:
        if not arm.wait_for_servers():
            arm.get_logger().error("action server 未就緒，中止")
            return

        # 開機立即讀取並顯示目前狀態（關節角度 + 末端點姿態，末端點另外換算成
        # 教導器 Base 座標系，方便直接對照教導器畫面）。下面的範例都會重複用到
        # 這裡讀到的 base_xyz / base_rotvec，不用再重新查一次。
        joint_positions = arm.get_current_joint_positions()
        if joint_positions is None:
            arm.get_logger().error("讀不到目前關節狀態，中止")
            return
        arm.get_logger().info(
            "目前關節角度 (rad): " + ", ".join(f"{p:.4f}" for p in joint_positions)
        )
        arm.get_logger().info(
            "目前關節角度 (deg): "
            + ", ".join(f"{math.degrees(p):.2f}" for p in joint_positions)
        )

        current_pose = arm.wait_for_tcp_pose(timeout_sec=5.0)
        if current_pose is None:
            arm.get_logger().error("讀不到末端點姿態 (tf base_link -> tool0)，中止")
            return
        (x, y, z), (qx, qy, qz, qw) = current_pose
        rx, ry, rz = quaternion_to_rotation_vector(qx, qy, qz, qw)
        base_xyz, base_rotvec = base_link_to_ur_base((x, y, z), (rx, ry, rz))
        arm.get_logger().info(
            f"目前末端點 [教導器 Base]: xyz=({base_xyz[0]:.4f}, {base_xyz[1]:.4f}, "
            f"{base_xyz[2]:.4f}) rxryrz=({base_rotvec[0]:.4f}, {base_rotvec[1]:.4f}, "
            f"{base_rotvec[2]:.4f})"
        )
        # 等代按下enter再繼續，避免程式一開就馬上動作
        input("按下 Enter 繼續...")
        # ------------------------------------------------------------------
        # 以下才是真正會讓手臂動作的範例，預設全部註解，要跑哪個自己打開一段
        # ------------------------------------------------------------------

        # 範例 1：關節空間單點目標（相對目前姿態的小幅度移動，示範用）
        # target = list(joint_positions)
        # target[5] += 0.2  # wrist_3 +0.2 rad
        # arm.move_joint(target, time_from_start=3.0)

        # 範例 2：關節空間多點軌跡（一次送出多個路徑點）
        # arm.move_joint_waypoints([
        #     {"positions": [0.3, -1.4, 1.4, -1.5, -1.5, 0.0], "time_from_start": 3.0},
        #     {"positions": [0.6, -1.2, 1.2, -1.4, -1.4, 0.0], "time_from_start": 6.0},
        # ])

        # 範例 3： 末端點空間單點目標（相對目前姿態的小幅度移動）
        # target_base_xyz = (-0.185, -0.353, 0.356)      # 目標位置 (m)
        # target_base_rotvec = (2.8754, 1.2564, 0.0358)  # 目標姿態（rad)
        # target_position, target_orientation = ur_base_to_base_link(
        #     target_base_xyz, target_base_rotvec
        # )
        # success = arm.move_pose_linear(
        #     target_position, target_orientation, velocity_scaling=SPEED_SCALING
        # )
        # arm.get_logger().info("末端點移動完成" if success else "末端點移動失敗")

        # 範例 4：末端點空間多點連續軌跡——直接寫教導器 Base 座標系的 xyz + rxryrz
        # （time_from_start 是從整條軌跡開始算的秒數，必須遞增；全部路徑點會規劃成
        # 「一條連續軌跡」一次送出，中途不會停頓）
        # ee_trajectory = [
        #     # (x, y, z, rx, ry, rz, time_from_start)
        #     (-0.185, -0.353, 0.356, 2.8754, 1.2564, 0.0358, 3.0),
        #     (-0.021, -0.352, 0.356, 2.8860, 1.2600, 0.0320, 6.0),
        #     (-0.021, -0.432, 0.306, 2.8860, 1.2600, 0.0320, 9.0),
        #     (-0.241, -0.432, 0.355, 2.8860, 1.2600, 0.0320, 12.0),
        # ]
        # TrajectoryPoints = []
        # for x, y, z, rx, ry, rz, t in ee_trajectory:
        #     position, orientation = ur_base_to_base_link((x, y, z), (rx, ry, rz))
        #     TrajectoryPoints.append(
        #         {"position": position, "orientation": orientation, "time_from_start": t}
        #     )

        # success = arm.move_pose_waypoints(TrajectoryPoints)
        # arm.get_logger().info("末端點軌跡完成" if success else "末端點軌跡失敗")

        # 範例 5：把範例 4 的路徑點當成「模型一次只吐一個點」來測試——不規劃成一條
        # 連續軌跡，而是每個點各自呼叫 move_pose_linear() 單點移動，等這個點真的執行
        # 完成（wait=True）才送下一個點。之後真的接 DP 模型時，把 ee_trajectory 這個
        # list 換成「跟模型要下一個點」的呼叫即可，迴圈邏輯不用改。
        ee_trajectory = [
            # (x, y, z, rx, ry, rz)  教導器 Base 座標系
            (-0.185, -0.353, 0.356, 2.8754, 1.2564, 0.0358),
            (-0.021, -0.352, 0.356, 2.8860, 1.2600, 0.0320),
            (-0.021, -0.432, 0.306, 2.8860, 1.2600, 0.0320),
            (-0.241, -0.432, 0.355, 2.8860, 1.2600, 0.0320),
        ]
        for i, (x, y, z, rx, ry, rz) in enumerate(ee_trajectory):
            # 模型輸出的末端點位置/姿態是教導器 Base 座標系，先轉成 base_link 座標系再送給 move_pose_linear()
            # 真的會用到的只有以下兩行，其他只是示範用的 log 顯示。
            arm.get_logger().info(
                f"action target: xyz=({x:.4f}, {y:.4f}, {z:.4f}) ,rxryrz=({rx:.4f}, {ry:.4f}, {rz:.4f})"
            )
            position, orientation = ur_base_to_base_link((x, y, z), (rx, ry, rz))
            success = arm.move_pose_linear(
                position, orientation, velocity_scaling=SPEED_SCALING, wait=True
            )

            arm.get_logger().info(f"第 {i} 點{'完成' if success else '失敗'}")
            if not success:
                break

    except KeyboardInterrupt:
        pass

    finally:
        arm.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
