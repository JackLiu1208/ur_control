# ur_control

UR5e ROS 2 控制套件：關節/末端點控制、Robotiq 夾爪整合，以及一套用來部署 Diffusion
Policy（receding horizon action chunking）到真實手臂的高頻串流控制與延遲診斷工具。

## 快速開始

```bash
# build（在 ~/ros2_ws）
cd ~/ros2_ws && colcon build --packages-select ur_control --symlink-install
source ~/ros2_ws/install/setup.bash
```

先啟動手臂（見文末「啟動手臂 / 夾爪」），確認教導器 External Control 包在 Loop 裡執行中、
面板切到 Remote Control，再執行下面任一支程式。

## 專案架構

- **`ur_arm_node.py`** — 核心通訊 node（`URArmNode`）。關節控制走 `FollowJointTrajectory`
  action（跟官方 external-control 範例同一條路），Cartesian 規劃走 MoveIt2
  （`move_pose`/`move_pose_linear`/`compute_ik`），另外提供 `compute_ik_analytic()`（解析 IK，
  見下方優化紀錄）、`get_current_tcp_pose()`（TF）/`get_current_tcp_pose_analytic()`（FK，
  高頻路徑用）、夾爪控制。
- **`pose_utils.py`** — 座標系轉換。ROS 的 `base_link`（REP-103）跟 UR 教導器/RTDE 的
  `Base` 座標系差一個繞 Z 180° 的旋轉；UR 的旋轉向量代表 `q_tool0_base`（tool→base），
  是「工具在基座下的姿態」這個常見假設的反向——兩個都是實測到真實機器人上才抓出來的坑，
  已經包成 `ur_base_to_base_link()`/`base_link_to_ur_base()`，全專案共用同一份轉換，
  不要在別的地方重新推導。
- **`trajectory_interpolator.py`** — 時間查詢式軌跡緩衝區（`PoseTrajectoryInterpolator`/
  `JointTrajectoryInterpolator`），概念上對應 diffusion_policy/UMI 真實機器人部署代碼裡的
  同名元件：高頻控制迴圈只查表，新的 model chunk 進來時用 `schedule_waypoint()` 在目前時刻
  截斷舊軌跡（保持連續）、超速自動拉長時間（不暴衝）。純 numpy，無 ROS 依賴，可離線測試
  （`test_trajectory_interpolator.py`）。
- **`analytic_ik.py`** — UR 閉式解析 IK（PickNik `ur-analytic-ik`，IKFast 產生），
  `URArmNode.compute_ik_analytic()` 的底層實作，取代高頻路徑上原本的 MoveIt
  `compute_ik` service round-trip。
- **`trajectory_logger.py` / `examples/analyze_trajectory.py`** — 記錄 + 事後分析。
  `predicted`/`commanded`/`measured` 三條獨立時間序列存成 CSV，分析腳本負責比對/畫圖，
  兩者刻意分開（見下方優化紀錄「log 管線拆分」）。

## 程式清單與啟動指令

Build/source 見上方「快速開始」。

| 檔案 | 功能 | 指令 |
| --- | --- | --- |
| `run_waypoints.py` | 開機顯示目前姿態；範例 1~5：關節/末端點單點、多點連續軌跡、逐點等待執行 | `ros2 run ur_control run_waypoints` |
| `monitor_state.py` | 終端機即時監控關節/末端點/速度/加速度/FPS，純讀取不會動手臂 | `ros2 run ur_control monitor_state` |
| `gripper_demo.py` | 開合夾爪一次（需要 `ur_robotiq_bringup` 啟動、夾爪硬體正常） | `ros2 run ur_control gripper_demo` |
| `test_trajectory_interpolator.py` | 離線測試 interpolator 接縫連續性，不需要 ROS/手臂 | `ros2 run ur_control test_trajectory_interpolator` |
| `test_analytic_ik.py` | 離線迴歸測試 analytic_ik 的 round-trip 正確性，不需要 ROS/手臂 | `ros2 run ur_control test_analytic_ik` |
| `dp_control_ros2.py` | DP 部署測試（走 `forward_position_controller` 高頻串流 + 背景 executor），需先 `use_fake_hardware:=true` 在 RViz 驗證 | `ros2 run ur_control dp_control_ros2` |
| `dp_control_rtde.py` | DP 部署測試（走原生 RTDE `servoJ`），跟 ROS2 driver 不能同時連線，**完全沒測過** | `ros2 run ur_control dp_control_rtde` |
| `analyze_trajectory.py` | 分析 `dp_control_*.py` 錄下的三條時間序列，X/Y/Z/RX/RY/RZ 六軸疊圖 + 誤差圖，MAE/MSE/RMSE/R² | `ros2 run ur_control analyze_trajectory <experiment_name>` |

共用模組（不直接執行，被上面程式 import）：`ur_arm_node.py`、`pose_utils.py`、
`motion_timing.py`、`state_monitor.py`、`trajectory_interpolator.py`、`trajectory_logger.py`、
`analytic_ik.py`、`examples/fake_dp_model.py`（假模型輸出，`dp_control_*`/
`test_trajectory_interpolator.py` 共用，之後接真模型時只要把 `fake_dp_inference()`
換成呼叫真模型、回傳同樣格式即可，其餘串流/插值邏輯不用改）。

### 軌跡分析（模型推論 vs 實際執行）

`dp_control_ros2.py` 每次執行都會存三條分開記錄的時間序列成 CSV（各自帶原生時間戳，
不在線上互相插值/對齊，避免把「量測來源本身比較慢」誤記成「機器人在頓」）：

- `<EXPERIMENT_NAME>_predicted.csv` — 模型推論輸出（只記錄成功求出 IK、真的被排進關節
  軌跡的點）
- `<EXPERIMENT_NAME>_commanded.csv` — 真的送給 controller 的 setpoint（Cartesian）
- `<EXPERIMENT_NAME>_measured.csv` — 機器人量測回來的實際位姿

放在 `data/<EXPERIMENT_NAME>_*.csv`（`.gitignore` 掉了，不會進 git）。`EXPERIMENT_NAME`
在 `dp_control_ros2.py` 檔案最上面，每次想保留獨立的實驗結果就改個名字，同名會直接覆蓋。

跑完之後分析：

```bash
ros2 run ur_control analyze_trajectory <experiment_name>
```

只會比較「有推論、後來也真的被走到」的點——一個 chunk 預測 `T_p=16` 步，但只有前
`T_a=8` 步左右會在下一次 replan 蓋掉這個 chunk 之前真的被機器人走到（receding horizon
的定義，`T_a/T_p=50%` 是正常現象不是 bug），其餘的點會自動被排除，不列入誤差計算。
輸出：

- `<experiment_name>_axes_overlay.png` — X/Y/Z/RX/RY/RZ 六軸，predicted vs actual 疊在
  同一張子圖比較（姿態轉成 UR 旋轉向量再轉 degree，不直接比四元數分量——四元數 q 和 -q
  代表同一個旋轉，逐分量比較沒有物理意義）
- `<experiment_name>_axes_error.png` — 六軸個別的追蹤誤差（predicted − actual）隨時間變化，
  子圖標題附上該軸的 MAE/RMSE/MSE/R²
- `<experiment_name>_metrics.txt` — 上面兩張圖的數字版，另外附整體位置追蹤誤差（歐氏距離
  mean/std/max，DP/UMI 真實機器人部署常用指標）跟姿態追蹤誤差（四元數測地線角度）
- 圖上文字全部英文（中文字型在無頭環境顯示不出來），console 輸出維持中文

R² 在訊號幾乎不變的軸（例如平面圓的 Z 軸）數學上沒有意義（分母趨近 0），這種情況會標成
`N/A` 並附原因，不會印出誤導人的巨大數字或 `nan`。

## 依賴

- `dp_control_rtde.py`：`ur_rtde`（PyPI，不是 ROS package）
  `pip3 install --user ur_rtde`
- `analytic_ik.py`（`dp_control_ros2.py` 的高頻 IK 用這個）：`ur-analytic-ik`（PyPI，
  PickNik 的 IKFast 閉式解，不是 ROS package）
  `pip3 install --user ur-analytic-ik`
  目前寫死 `ur5e`，換機型要改 `analytic_ik.py` 裡的 `UR_TYPE`。

## `dp_control_ros2.py` 優化紀錄

這支程式是把 Diffusion Policy 部署到真實手臂的高頻串流控制路徑：`fake_dp_inference()`
模擬模型每 `T_a * action_step_dt` 秒推論一次、輸出未來 `T_p` 步，排進
`PoseTrajectoryInterpolator`/`JointTrajectoryInterpolator`，`CONTROL_HZ` 高頻迴圈只查表
送給 `forward_position_controller`。以下是實測驅動的優化過程，按時間順序，每項都附
問題證據、根因、修法、驗證結果——之後改東西前建議先看這節，避免重踩已經踩過的坑。

### 1. 座標系與旋轉向量正負號

**問題**：從教導器讀回來的位置 x/y 差一個負號；旋轉向量在多個角度都對不上。
**根因**：ROS `base_link` 跟 UR `Base` 座標系差 180° 繞 Z；UR 的旋轉向量實際代表
`q_tool0_base`（tool→base），不是直覺假設的 `q_base_tool0`。
**修法**：`pose_utils.ur_base_to_base_link()`/`base_link_to_ur_base()`，包含一個額外的
四元數共軛處理旋轉向量方向。全專案（`dp_control_rtde.py`、`analytic_ik.py`）共用同一份。

### 2. Interpolator 接縫加速度尖峰

**問題**：新 chunk 接上舊軌跡時出現明顯頓挫。
**根因**：`trim(t)` 原本只在 `t >= 緩衝區終點` 時才整個 collapse，否則會留著舊 chunk
沒走完的尾巴，新點接在尾巴後面而不是取代它，離線測試量到加速度尖峰 248 m/s²。
**修法**：`trim()` 一律 collapse 成單一錨點（目前時刻的插值姿態），舊軌跡的未來完全捨棄。
**驗證**：`test_trajectory_interpolator.py`，加速度尖峰降到 1.54 m/s²，兩個連續性檢查通過。

### 3. 延遲前饋補償

**問題**：predicted 軌跡沿時間軸平移 50ms 後跟實際執行路徑的誤差從 7.94mm 降到 1.62mm；
誤差波形是軌跡速度的同頻正弦（相位落後，不是形狀不對）。
**修法**：控制迴圈查詢 interpolator 時查 `now + LATENCY_FEEDFORWARD_SECONDS`（預設
50ms，`[0, 0.15s]` 防呆 clamp），而不是 `now`。`interpolate()` 本身已經內建「超出範圍
clamp 在最後一個 waypoint、不外插」，不需要額外處理。
**注意**：這個 50ms 是在 RViz/fake hardware 上量的，換到真實手臂延遲組成不同，正式部署
前要用 `analyze_trajectory.py` 的對齊掃描在目標硬體重新量。

### 4. Log 管線拆分（predicted / commanded / measured）

**問題**：早期版本把「送出去的 setpoint」跟「量測回來的狀態」混記在同一份
`executed.csv`，用高頻迴圈的取樣時刻硬記成一筆——如果量測來源本身比較慢，log 裡會出現
重複值的階梯（ZOH），看起來像機器人在頓，其實是記錄方式的假象。
**修法**：拆成三條各自帶原生時間戳、不互相插值的獨立序列（見上方「軌跡分析」）。

### 5. IK 從 MoveIt service 換成解析解

**問題**：控制迴圈 dt 的 p99/max 分別是 24.20ms / 42.44ms，100% 集中在 replan trigger
附近。根因是每個 chunk 最多呼叫 16 次 `compute_ik()`（MoveIt `/compute_ik` service
round-trip，每次十幾到幾十 ms），同步卡在控制迴圈裡。
**修法**：換成 `analytic_ik.py`（PickNik `ur-analytic-ik`，IKFast 閉式解，in-process、
微秒等級）。**沒有**手推解析 IK 公式——正負號錯一個就可能讓真實手臂跑到錯的姿態，風險
太高不划算，改用業界既有、PickNik 維護的套件。
**座標系驗證**（真實機器人，純讀取沒送任何動作指令）：`ur_analytic_ik` 操作在 UR 原生
`Base` 座標系，套用既有的 `ur_base_to_base_link()` 後跟 MoveIt `/compute_fk` 完全吻合
（position diff 0.0mm，orientation diff 0.0009°）；`inverse_kinematics_closest()` 正確
選出離 seed 最近的分支，8 組解全部能 round-trip 回目標姿態（誤差 0.000000mm）；跟既有
`compute_ik()`（MoveIt）比對 20 個隨機目標，最大關節角差異 0.0016°。另外有 500 組隨機
姿態的離線迴歸測試（`test_analytic_ik.py`）常駐在專案裡，之後改動可以隨時重跑。
**驗證結果**（真實機器人）：dt p99 24.20ms → **8.51ms**、dt max 42.44ms → **9.49ms**，
30 秒測試無任何一次掉幀；`predicted vs commanded` 殘差維持 0.02mm → 0.016mm（沒有變差）。
**副作用**：這支程式不再需要 MoveIt/`move_group` 跑起來（不再呼叫它的 service）。

### 6. 背景 executor + QoS + FK-based measured

**問題**：IK 換掉之後，`measured`（機器人實際量測到的狀態）不但沒變好，還從 2.4Hz 退步到
1.37Hz。根因追查：整支程式只有一個裸的 `rclpy.spin_once(arm, timeout_sec=0.0)`，語意是
「一次最多服務一個 ready 的 callback」；`/joint_states` 訂閱、TF 內部的 `/tf`/`/tf_static`
訂閱全部擠在同一個預設 callback group，互相競爭這唯一的名額，而且 `/joint_states` 用的是
預設 `RELIABLE` + depth 10 QoS，處理跟不上時會累積 backlog、越補越舊。之前殘存的 2.4Hz
其實是 MoveIt IK 的 `spin_until_future_complete()` 順便把 executor 跑起來的「意外副作用」，
換成解析解之後這個副作用也消失了。
**修法**（三件事一起做）：

  1. `URArmNode._joint_state_sub` 的 QoS 改成 `BEST_EFFORT` + `depth=1`（跟 `RELIABLE`
     publisher 相容，這是高頻感測資料該用的設定，全域生效、所有用到 `URArmNode` 的程式都受益）
  2. 新增 `URArmNode.get_current_tcp_pose_analytic()`：直接讀快取的 `/joint_states` 算
     FK（`analytic_ik.forward_kinematics()`），不查 TF，不受 TF listener 的 callback 排程
     影響
  3. `dp_control_ros2.py` 改用**背景 `SingleThreadedExecutor`**，在專屬 daemon thread 裡
     持續 `spin_once()`，控制迴圈本身完全不再呼叫 `spin_once()`。只需要
     `SingleThreadedExecutor`（不需要 `MultiThreadedExecutor` + `ReentrantCallbackGroup`）
     ——原本會需要是因為 MoveIt IK 的 `spin_until_future_complete()` 會跟背景 executor
     搶著 spin 同一個 node（經典 rclpy 死鎖組合），换成解析解後高頻路徑上已經沒有任何
     service/action 呼叫，這個風險不存在了。
**執行緒排程細節（重要，之後改動要保持）**：背景 executor **只在串流迴圈這段期間存在**。
啟動前（回 home、`wait_for_tcp_pose`、切換 controller）跟結束後（切回預設 controller、
回 home）用的都是 `URArmNode` 內建的阻塞方法，內部是裸的
`rclpy.spin_once()`/`spin_until_future_complete()`（走全域 executor）——如果背景
executor 這時候還在跑，同一個 node 會同時被兩個 executor spin，一樣會撞上死鎖組合。
所以嚴格只在使用者按下 Enter、要開始串流的那一刻才 `executor.add_node()` +
啟動背景 thread，`finally` 裡第一件事就是停掉背景 thread、`remove_node()`，才呼叫任何
`URArmNode` 的阻塞方法。
**額外一起做的**：控制迴圈改成睡到下一個絕對 deadline（而非固定 `sleep(period)`），
避免長時間跑下來因為累積誤差導致實際頻率系統性偏低；`/joint_states` 實際收到頻率現在
會跟迴圈 Hz 印在同一行 log，當作背景 executor 的健康度監控。
**驗證結果**：獨立讀取測試（背景 executor + 純讀取迴圈，複製 `dp_control_ros2.py` 的
確切結構，沒有送任何動作指令）5 秒收到 2512 筆 `/joint_states` = **500.0Hz**，跟 driver
端原生發布頻率完全吻合。**完整串流迴圈下（一邊控制手臂一邊記錄）的實測數字還沒有正式
跑過，下一輪測試要確認 `measured.csv` 的有效更新率是否真的達到三位數 Hz、且
`predicted vs commanded` 殘差沒有變差。**

## 待處理事項

- [ ] **`dp_control_ros2.py` 下一輪測試**：驗證「背景 executor + QoS + FK-based
      measured」在完整串流（不是獨立讀取測試）下的實際效果——`measured` 有效更新率、
      跟 replan trigger 的時間相關性、`predicted vs commanded` 殘差有沒有維持。
- [ ] **DP 模型還沒真的接上**：`dp_control_ros2.py`/`dp_control_rtde.py` 目前用
      `fake_dp_model.py` 模擬推論（固定圓）。接真模型時把 `fake_dp_inference()` 換成呼叫
      模型、回傳同樣格式即可，其餘串流/插值邏輯不用改；但要檢查真模型訓練資料收集端的
      pipeline 延遲，跟部署端這裡量到的延遲是否一致（DP 是閉環的，訓練/部署延遲不一致，
      policy 學到的補償會失準，這件事目前的資料收集端程式碼不在這個 repo 裡，需要另外確認）。
- [ ] **測試軌跡只測了 XY 圓**：Z 跟姿態全程幾乎不變，六軸中有四軸沒被真正驗證過
      （`analyze_trajectory.py` 會把這種軸的 R² 標成 N/A，但這不能取代真的讓那些軸動起來
      測試）。之後應該加一條會動用全部六軸的軌跡（Z 疊加不同頻率正弦、RX/RY/RZ 擺動，
      刻意跟 XY 不同步以測出軸間耦合，同時驗證 SLERP）。
- [ ] **`dp_control_rtde.py` 完全沒測過**（RTDE 沒有等效的假硬體可以測，需要真實手臂或
      URSim）。跟 ROS2 driver 的 external control 不能同時連線，測試前記得確認沒有其他
      程式正在佔用連線。
- [ ] 兩支測完之後要比較實測頻率/平順度，決定最終要走哪一條（見兩支程式開頭的架構說明）。
- [ ] **Robotiq_Gripper URCap 跟 rs485 URCap（tool communication forwarding）衝突**：教導器上要移除
      `Robotiq_Gripper` URCap、改裝 `rs485-1.0.urcap`（檔案在
      `~/ur_ws/src/Universal_Robots_ROS2_Driver/ur_robot_driver/resources/rs485-1.0.urcap`），
      `use_tool_communication` 才會正常。裝完用 `nc -zv <robot_ip> 54321` 先確認連得上再開 launch。
- [ ] 上面那步做完後，還沒重新驗證過 `ur_robotiq_bringup` 完整啟動 + 夾爪開合成功。

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

`dp_control_ros2.py` 額外注意：不需要 MoveIt/`move_group`（IK 走
`compute_ik_analytic()`，不呼叫 MoveIt service）；第一次測試務必先用
`use_fake_hardware:=true` 在 RViz 裡看過一輪、確認軌跡合理再接真實手臂。
