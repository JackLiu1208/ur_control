#!/usr/bin/env python3
"""範例：開合 Robotiq 2F-140 夾爪。

需要先用 ur_robotiq_bringup 的 ur_robotiq_control.launch.py 啟動手臂+夾爪
（不是原本純手臂的 ur_control.launch.py），這樣 robotiq_gripper_controller
才會存在。夾爪跟手臂用同一個 URArmNode，跟關節/末端點控制共用同一套架構。
"""

import rclpy

from ur_control.ur_arm_node import URArmNode

# =============================================================================
# 可修改變數
# =============================================================================
NODE_NAME = "ur_gripper_demo_node"


def main(args=None):
    rclpy.init(args=args)
    arm = URArmNode(node_name=NODE_NAME)

    try:
        arm.get_logger().info("開爪...")
        if not arm.open_gripper():
            arm.get_logger().error("開爪失敗")
            return

        arm.get_logger().info("閉爪...")
        if not arm.close_gripper():
            arm.get_logger().error("閉爪失敗")
            return

        arm.get_logger().info("完成")

    finally:
        arm.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
