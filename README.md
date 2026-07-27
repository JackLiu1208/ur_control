# ur_control 專案筆記

## 待處理事項

- [ ] **Robotiq_Gripper URCap 跟 rs485 URCap（tool communication forwarding）衝突**：教導器上要移除
      `Robotiq_Gripper` URCap、改裝 `rs485-1.0.urcap`（檔案在
      `~/ur_ws/src/Universal_Robots_ROS2_Driver/ur_robot_driver/resources/rs485-1.0.urcap`），
      `use_tool_communication` 才會正常。裝完用 `nc -zv <robot_ip> 54321` 先確認連得上再開 launch。
- [ ] 上面那步做完後，還沒重新驗證過 `ur_robotiq_bringup` 完整啟動 + 夾爪開合成功。
- [ ] DP 模型還沒真的接上。`run_waypoints.py` 範例 5 只驗證過「逐點送、等這一點執行完才送下一
      點」這個流程能動，`ee_trajectory` 目前是寫死的測試點，之後接模型時把那個 list 換成跟模型
      要下一個點的呼叫即可。

## 程式清單（`ur_control` package，在 `~/ros2_ws`）

Build: `cd ~/ros2_ws && colcon build --packages-select ur_control --symlink-install`
Source: `source ~/ros2_ws/install/setup.bash`

| 檔案 | 功能 | 指令 |
| --- | --- | --- |
| `run_waypoints.py` | 開機顯示目前姿態；範例 1~5：關節/末端點單點、多點連續軌跡、逐點等待執行 | `ros2 run ur_control run_waypoints` |
| `monitor_state.py` | 終端機即時監控關節/末端點/速度/加速度/FPS，純讀取不會動手臂 | `ros2 run ur_control monitor_state` |
| `gripper_demo.py` | 開合夾爪一次（需要 `ur_robotiq_bringup` 啟動、夾爪硬體正常） | `ros2 run ur_control gripper_demo` |

共用模組（不直接執行，被上面程式 import）：`ur_arm_node.py`（核心通訊 node）、`pose_utils.py`
（座標/旋轉換算）、`motion_timing.py`、`state_monitor.py`。

## 啟動手臂 / 夾爪（在 `~/ur_ws`）

Source: `source ~/ur_ws/install/setup.bash`

**純手臂**（兩個 terminal）：

```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.1.30
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e
```

**手臂 + Robotiq 2F-140 夾爪**（兩個 terminal，取代上面兩行）：

```bash
ros2 launch ur_robotiq_bringup ur_robotiq_control.launch.py ur_type:=ur5e robot_ip:=192.168.1.30
ros2 launch ur_robotiq_bringup ur_robotiq_moveit.launch.py ur_type:=ur5e
```

啟動前檢查：教導器 External Control 包 Loop、保持執行中；面板切 Remote Control；夾爪模式額外
需要 rs485 URCap（見上方待處理事項）。
