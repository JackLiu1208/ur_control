# ur_control

UR5e ROS 2 控制套件：關節/末端點控制、Robotiq 夾爪整合、Diffusion Policy 高頻串流部署工具。

## 快速開始

```bash
cd ~/ros2_ws && colcon build --packages-select ur_control --symlink-install
source ~/ros2_ws/install/setup.bash
```

啟動手臂（見文末）→ 確認教導器 External Control 在 Loop 裡執行中、面板切 Remote Control →
執行下面任一支程式。

## 核心模組：功能與方法

| 模組 | 功能 | 方法 |
| --- | --- | --- |
| `ur_arm_node.py` | 機器人控制／狀態讀取入口（`URArmNode`） | 關節控制走 `FollowJointTrajectory` action；Cartesian 規劃走 MoveIt2；`compute_ik_analytic()` 走解析 IK；`get_current_tcp_pose_analytic()` 走 FK |
| `pose_utils.py` | ROS `base_link` ↔ UR 教導器 `Base` 座標系互轉 | `ur_base_to_base_link()` / `base_link_to_ur_base()`，含旋轉向量正負號修正 |
| `analytic_ik.py` | 微秒等級 IK/FK，不呼叫 MoveIt service | PickNik `ur-analytic-ik`（IKFast 閉式解）+ 內建 round-trip 驗證，驗證失敗回傳 `None`；`load_calibration()` 選用性接上出廠校正檔，細節見模組開頭「出廠校正」小節 |
| `trajectory_interpolator.py` | 高頻控制迴圈的軌跡查表 | 時間索引 waypoint buffer，`schedule_waypoint()` 在目前時刻截斷舊軌跡、超速自動拉長時間 |
| `trajectory_logger.py` | 記錄推論/送出/量測三條時間序列 | 各自帶原生時間戳存 CSV，不在線上互相插值 |
| `examples/analyze_trajectory.py` | 事後比對三條序列、算誤差、畫圖 | 六軸（X/Y/Z/RX/RY/RZ）疊圖 + 誤差圖 + MAE/MSE/RMSE/R² |
| `dp_controller.py` | **可重用 API**：把任何模型的推論輸出變成平順的高頻機器人指令串流 | `DPController`（控制迴圈本體）+ `DPControlConfig`（參數），見下方「API」 |

## 程式清單

| 檔案 | 功能 | 指令 |
| --- | --- | --- |
| `run_waypoints.py` | 關節/末端點單點、多點連續軌跡範例 | `ros2 run ur_control run_waypoints` |
| `monitor_state.py` | 終端機即時監控狀態，純讀取不動手臂 | `ros2 run ur_control monitor_state` |
| `gripper_demo.py` | 開合夾爪一次 | `ros2 run ur_control gripper_demo` |
| `test_trajectory_interpolator.py` | 離線測試 interpolator 接縫連續性 | `ros2 run ur_control test_trajectory_interpolator` |
| `test_analytic_ik.py` | 離線迴歸測試 analytic_ik round-trip 正確性：隨機取樣 + 高度×旋轉結構化網格 | `ros2 run ur_control test_analytic_ik` |
| `dp_control_ros2.py` | `DPController` API 的示範程式，`TRAJECTORY_TYPE` 切換四種假軌跡 | `ros2 run ur_control dp_control_ros2` |
| `dp_control_rtde.py` | （已停用）原生 RTDE `servoJ` 路徑，ROS2 路徑已滿足需求，不再走這條 | `ros2 run ur_control dp_control_rtde` |
| `analyze_trajectory.py` | 分析 `dp_control_*.py` 錄下的資料 | `ros2 run ur_control analyze_trajectory <experiment_name>` |

`test_trajectory_interpolator.py`/`test_analytic_ik.py` 純數學、不需要 ROS/手臂，可以隨時重跑
當回歸測試。

`dp_control_ros2.py` 的四種 `TRAJECTORY_TYPE`：

- `"six_axis_circle"`：XY 圓 + Z 正弦 + RX/RY/RZ 擺動，四軸不同頻率/相位、刻意不
  同步——測軸間耦合跟 SLERP，六種裡對關節速度/加速度需求最高（持續轉彎，沒有
  直線段），比較接近壓力測試,不是真實 DP 推論軌跡的典型樣貌。
- `"figure_eight"`：XY 8 字形，曲率連續變化、姿態固定不變。
- `"step_waypoints"`：固定點之間離散跳動，測 `schedule_waypoint()` 超速自動拉長
  時間那段邏輯。
- `"line_back_and_forth"`：單軸三角波來回，等速度、端點瞬間反向，姿態固定不變。

只有 `six_axis_circle` 會動姿態；其餘三種姿態全程固定不變。

## API：把你的模型接上去

`dp_controller.py` 是可重用的部分：`DPController` 負責回家/切 controller/背景 executor/
延遲前饋/查表送指令/記錄這一整套（已在真實 UR5e 驗證過，見下一節），你只要給它一個
`inference_fn` 就好，不用碰控制迴圈本身。

`enable_markers=True`（預設）時會在 RViz 畫三條線（topic 見 `marker_topic`，預設
`/dp_trajectory_markers`），對應三個不同來源，可以直接視覺比對：

- **黃色**（`planned_marker_namespace`）：這次 replan 拿到的 16 步原始預測點。
- **綠色**（`executed_marker_namespace`）：`pose_interp` 在 Cartesian 空間插值
  出來的「理論上應該在哪」，跟真正送給機器人的關節指令是平行、獨立算出來的
  （不是量測值，細節見 `dp_controller.py` 對應段落）。
- **藍色**（`measured_marker_namespace`）：機器人真正的 TCP 位置
  （`get_current_tcp_pose_analytic()`，跟 `_measured.csv` 同一個來源）——
  這條才是「真的量到的軌跡」，跟綠色（命令/理論軌跡）分開看，才看得出命令
  軌跡有沒有跟量測結果對上。

```python
import rclpy
from ur_control.dp_controller import DPController, DPControlConfig

def my_policy(t_obs, current_joint_positions, current_tcp_pose):
    """t_obs: 這次 replan 的時刻（秒，相對 run() 開始執行）
       current_joint_positions: 目前 6 個關節角 (rad)
       current_tcp_pose: 目前 TCP 在 base_link 座標系的 (position_xyz, orientation_xyzw)
       回傳: (timestamps, positions, orientations)，三個等長 list，長度 = prediction_horizon，
             都是 base_link 座標系（不是教導器 Base！訓練資料如果是教導器座標系記的，
             這裡要自己先用 pose_utils.ur_base_to_base_link() 轉換）"""
    position, orientation = current_tcp_pose
    timestamps = [t_obs + i * 0.1 for i in range(1, 17)]
    positions = [...]      # 每步的目標位置 (x, y, z)
    orientations = [...]   # 每步的目標姿態四元數 (x, y, z, w)
    return timestamps, positions, orientations

rclpy.init()
config = DPControlConfig(
    home_joint_positions=[0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0],  # 安全關鍵，換工作空間要自己重新確認
    tcp_offset_xyz=TCP_OFFSET_ROBOTIQ_2F140,  # 見下方「TCP offset」
)
DPController(config).run(my_policy)
rclpy.shutdown()
```

`inference_fn` 只在每次 replan（每 `action_horizon * action_step_dt` 秒）被呼叫一次，
不是每個控制 tick——推論多慢都不會拖累高頻串流，串流的平順度是 `DPController` 的責任。

完整可執行版本、含註解說明每個設計選擇：`examples/dp_control_ros2.py`。所有可調參數見
`DPControlConfig`（`dp_controller.py`），預設值都是這次真實 UR5e 驗證用的數字，唯一沒有
預設值、必須自己填的是 `home_joint_positions`（安全關鍵，換工作空間要重新確認）。

`control_space` 控制迴圈要在哪個空間做 120Hz 插值：`"joint"`（預設，已驗證，每個
replan 解完整個 chunk 的 IK 再做關節空間插值）或 `"cartesian"`（每 tick 現場對
Cartesian 插值結果解一次 IK，是診斷用的消融測試選項，不是取代 joint 模式的正式
方案）。兩種模式都需要 `max_joint_speed`/`joint_accel_limit`（安全關鍵、無預設
值）——不管哪種模式，`run()` 最後實際發布的指令都會經過同一個平滑濾波器套用
這兩個上限，避免瞬間速度/加速度過大觸發 UR 內建（教導器 Installation 查不到、
也調不了）的保護性停機。來龍去脈跟目前進度見下方「問題與解決方法」第 10~18
項。

`DPController` 目前依賴的 UR 專屬部分只有 `analytic_ik.py`（IK/FK 解析解）；其餘（receding
horizon 排程、interpolator、延遲前饋、背景 executor、log 管線）都是通用邏輯，換機型只要
換掉 `analytic_ik.py` 那一層。

### TCP offset（裝/拆夾爪）

`inference_fn` 回傳的 `position_xyz` 是「TCP 點」的座標，不是法蘭面（`tool0`）本身。TCP 相對
法蘭面的偏移由 `DPControlConfig.tcp_offset_xyz` 控制（只有平移，量測單位公尺，`ur_arm_node.py`
有兩組現成預設）：

```python
from ur_control.ur_arm_node import TCP_OFFSET_BARE_FLANGE, TCP_OFFSET_ROBOTIQ_2F140

DPControlConfig(tcp_offset_xyz=TCP_OFFSET_ROBOTIQ_2F140)   # 裝了 25cm Robotiq 2F-140（目前預設）
DPControlConfig(tcp_offset_xyz=TCP_OFFSET_BARE_FLANGE)      # 拆掉夾爪，裸法蘭面
```

`analytic_ik.py` 的 IK/FK 完全不知道 TCP offset 這件事，只認法蘭面——offset 是在
`URArmNode._tcp_pose_to_tip_pose()`/`_tip_pose_to_tcp_pose()` 裡另外疊加的一個固定平移，換
工具只要換這個值，不需要動 IK。不填（`None`，預設）就沿用 `URArmNode` 自己的預設值（目前是
Robotiq 2F-140 的 25cm）。

這個值只影響 ROS 這邊怎麼算座標，**不會**自動同步到機器人控制器自己的安全系統——夾爪的
質量/質心要另外在教導器的 Installation 分頁設定，兩邊是獨立的兩件事。

## `dp_control_ros2.py`：問題與解決方法

實測驅動的優化紀錄，按時間順序。改這支程式前先看這張表，避免重踩已經踩過的坑。

| # | 問題 | 解決方法 | 結果 |
| --- | --- | --- | --- |
| 1 | 位置 x/y 差負號、旋轉向量對不上 | `pose_utils.ur_base_to_base_link()` 補 180° 繞 Z + 旋轉向量方向修正 | 全專案共用，`dp_control_rtde.py`/`analytic_ik.py` 都靠它 |
| 2 | 新 chunk 接舊軌跡時頓挫，加速度尖峰 248 m/s² | `trim(t)` 一律 collapse 成單一錨點 | 尖峰降到 1.54 m/s²（`test_trajectory_interpolator.py` 驗證） |
| 3 | predicted 軌跡有相位落後（跟波形速度同頻） | 查詢 interpolator 用 `now + LATENCY_FEEDFORWARD_SECONDS`（預設 50ms） | 誤差 7.94mm→1.62mm；此值換手臂要重新校準 |
| 4 | commanded/measured 混記一份 CSV，量測慢時出現 ZOH 階梯假象 | 拆成 predicted/commanded/measured 三條獨立時間序列，各自原生時間戳 | 分析時能分清楚誤差在控制端還是量測端 |
| 5 | 控制迴圈 dt p99/max = 24.20/42.44ms，集中在 replan 附近 | MoveIt `compute_ik()` service round-trip 換成 `analytic_ik.py`（in-process） | dt p99→8.51ms、max→9.49ms，30 秒無掉幀 |
| 6 | 換 IK 後 measured 更新率退步（2.4Hz→1.37Hz） | 見下方「背景 executor」 | `/joint_states` 訂閱改背景 executor + BEST_EFFORT QoS |
| 7 | **試過、撤回（三次）**：懷疑解析 IK 在高度/大幅旋轉時解錯，嘗試換 MoveIt Servo | 三版都撤回：阻塞單次呼叫接不上串流／P 控制器暴走／feedforward+trim 沒測就撤 | 懷疑後來查無實據（見第 8 項），`dp_controller.py` 維持解析 IK |
| 8 | 承上：光靠隨機取樣不夠讓人安心解析 IK 沒有高度/旋轉相關的隱藏 bug | `test_analytic_ik.py` 加**結構化網格測試**：J2/J3（高度/手臂形狀）× J4/J5/J6（姿態）× J1，11250 組跑 IK→FK round-trip | 11226/11250 通過，24 筆失敗全落在 J5=0°——UR 已知**腕部奇異點**，`analytic_ik.solve()` 正確回傳 `None`，非算錯 |
| 9 | `analyze_trajectory.py` 一直只比較 predicted vs actual，`_commanded.csv` 從沒被用到 | 加第二組獨立比較 **commanded vs measured**（純低階追蹤誤差），輸出獨立圖檔 | 六軸圓 demo：predicted vs actual mean 6.8mm，commanded vs measured mean 1.4mm——誤差主要來自規劃面，非執行端 |
| 10 | 消融測試：predicted vs actual 誤差裡有沒有一塊是關節空間 vs Cartesian 空間插值本身的幾何落差 | `DPControlConfig.control_space="cartesian"`：不在 replan 時解 IK，改成每個 120Hz tick 現場解。joint 模式不受影響，兩條路徑並存 | 位置誤差 mean 6.8mm→2.4mm，證實落差存在。**不是**取代 joint 模式的正式方案，是診斷工具 |
| 11 | 承上：cartesian 模式在特定姿態平滑地跟不上、切內側抄近路又衝出去——非 IK 解錯（IK 失敗全程 0 次）、非分支跳動（那是瞬間不連續，這個是平滑的） | 根因：奇異點附近所需關節角速度趨近無窮大，`max_pos_speed`/`max_rot_speed` 限的是 Cartesian 速度，管不到這個。修法：cartesian 模式加 `max_joint_speed`（安全關鍵，無預設值），`run()` 裡做**位移量夾住的 rate limiter**（每 tick 最多移動 `max_joint_speed * period`，夾的是位移不是先算速度再積分，剩餘距離小於這個上限就正好落在目標上，不會超調） | 模擬驗證收斂不超調。**還沒在手臂上實測**——注意 `關節限速觸發` 警告 log |
| 12 | 懷疑 cartesian 模式誤差是 `tcp_offset_xyz` 算錯 | 寫唯讀診斷比對 TF 版本 vs 解析 FK 版本 TCP 位置：差距忽大忽小（0.9~97mm），額外發現 TF 穩定落後 `/joint_states` 10~50ms | **排除**：是時間沒對齊，不是座標算錯；`dp_controller.py` 全程用不查 TF 的 `get_current_tcp_pose_analytic()`，不受影響 |
| 13 | 六軸圓加姿態擺動後，逐軸 RX/RY/RZ RMSE 高達 38~39deg，但整體 geodesic 誤差只有 0.8deg——對不上 | 根因：這個工作站姿態旋轉角接近 180°（旋轉向量表示法奇異邊界），`quaternion_to_rotation_vector()` 逐點獨立 force w>=0 導致相鄰時刻跳到相反代表。修法：兩條序列先展開四元數連續性（`_unwrap_quaternions()`）再比較 | RX/RY/RZ RMSE 38~39deg→0.6~0.7deg，跟整體 geodesic 誤差一致；四個既有資料集都驗證過 |
| 14 | 六軸圓測試軌跡本身可能太刁鑽（六軸同時連續變化，跟真實 DP 推論軌跡差很多），其他三種測試軌跡都追蹤準確 | 待驗證：調高 `max_joint_speed`，或改用較單純的軌跡類型交叉比對 | 進行中——目前傾向認為是測試軌跡刻意刁鑽 + 限制值偏保守的組合，不是控制邏輯本身的 bug |
| 15 | 示教器 Installation 分頁找不到「關節加速度上限」這個設定值，`max_joint_acceleration` 是編出來的數字，不是真實規格 | 移除 `max_joint_acceleration`：rate limiter 改成只夾位移量（見第 11 項），數學上不需要加速度上限也能保證不超調 | `DPControlConfig`/`run()`/README 三處同步移除，`control_space="cartesian"` 現在只需要 `max_joint_speed` |
| 16 | 實機測試（joint 模式）因為速度太快觸發保護性停機 | 根因：`joint_interp.schedule_waypoint()` 支援 `max_joint_speed` 參數夾關節角速度，但 `run()` 呼叫時**沒有傳這個參數**——joint 模式排點的時間只看 `pose_interp` 算出來的 Cartesian 速度限制（`max_pos_speed`/`max_rot_speed`），IK 在奇異點/大幅旋轉附近可能讓「Cartesian 移動很小」對應到「關節角要轉很大」，完全沒被夾到，一瞬間的高關節角速度指令觸發保護性停機 | 補上 `max_joint_speed=cfg.max_joint_speed`，兩種 control_space 現在都強制要求填這個值（`run()` 驗證從只檢查 cartesian 改成兩種模式都檢查）。**還沒在手臂上重新驗證** |
| 17 | 承上，cartesian 模式也在實機一啟動就出現關節瞬間跳變（觸發限速警告後卡住不動） | 根因：`run()` 一開始用 `wait_for_tcp_pose()`（查 TF）取得 `origin_pose` 當 `pose_interp` 的起點錨點——這個方法只要 TF buffer 裡**有任何一筆** transform 就立刻回傳，不保證是這次回完 home pose *之後* 的新值，可能撈到上一次啟動、甚至更早留下的舊姿態。同時間 `joint_interp` 的起點、`inference_fn` 拿到的 `current_tcp_pose` 都是用剛讀到的 `joint_positions` 算 analytic FK，是新的——三個「起點」對不上，第一個 replan 排出來的 Cartesian 軌跡就會從錯的舊姿態接到正確的新姿態，解 IK 後變成一段憑空的關節跳變 | 起點統一改用 `arm.get_current_tcp_pose_analytic()`（跟 `joint_positions` 同一份 `/joint_states`），不再查 TF。**還沒在手臂上重新驗證** |
| 18 | 承上兩項修好之後，實機（joint 模式）還是跳 Protective Stop，但 log 完全沒有我們軟體自己印出的警告——代表警報不是 log 裡任何東西觸發的 | 根因：UR 控制器本身有一套跟教導器 Installation 設定無關、**不能調、卻一直在監測**的低階保護機制，會檢查串流指令的速度變化夠不夠平滑；第 15 項拿掉 `max_joint_acceleration` 後，`joint` 模式的 `joint_interp` 在 chunk 交界處本來就可能有速度不連續（斜率瞬間改變），`cartesian` 模式的位移量夾子也只保證位移不超調、沒有限制速度變化的快慢——兩者都可能讓瞬間加速度衝到 UR 內建保護機制的門檻，教導器上找不到這個設定不代表它不存在 | 加回 `joint_accel_limit`（改名跟第 15 項的 `max_joint_acceleration` 區分：這次**明確定位成純軟體端的保守平滑常數，不宣稱是 UR 規格值**），改成 `run()` 裡統一的**平滑濾波器**：不管 `control_space` 是哪一種，最後真正發布的指令都會先經過同一組「煞車距離公式」速度/加速度限制（第 11 項那組數學，現在對兩種模式都生效，不再只在 cartesian 分支）。模擬驗證收斂、加速度確實卡在設定值不會衝過頭。**還沒在手臂上重新驗證**——先從保守值（2 rad/s²）開始測 |
| 19 | 第 16~18 項修好後的第一次真機測試：`analyze_trajectory.py` 分析 commanded vs measured（純用 `/joint_states` 算，不查 TF），確認控制迴圈本身有沒有如實執行 | 六軸圓 demo：位置 mean 1.88mm / max 5.58mm，姿態 mean 0.25° / max 7.27°；額外把 measured/commanded 的 XY 座標算圓心/半徑分佈：兩者都是半徑 50mm、std < 0.5mm 的乾淨圓 | **確認**：手臂真的有照指令走，不是只有軟體自己算得漂亮 |
| 20 | 承上，用戶從正上方拿尺目測夾爪實際畫出來的形狀，看起來是橢圓、範圍是設計值（10cm 直徑）的兩倍（0~20cm） | 檢查了旋轉數學（跟 scipy 逐位元一致）、TCP offset 轉換（只做一次，沒有重複疊加）——都正確。懷疑跟肉眼追蹤的參考點不是真正的 TCP 中心線（21.17cm 處的虛擬點）有關：夾爪還有 ±6° 姿態擺動，肉眼容易追蹤到夾爪外殼/指尖等實際看得到的部位，這些點相對 TCP 中心線的偏移會被姿態旋轉放大成看起來更誇張的晃動；另外 18~21cm 高度 + Z 軸 ±2.5cm 起伏也會造成明顯視差 | 待驗證：建議改用垂直架設的相機或量測夾爪尖端本身裝一個實體標記點，而不是肉眼追蹤夾爪外殼 |
| 21 | 承上，用戶提供這台手臂出廠校正檔 `my_robot_calibration.yaml`，懷疑 `analytic_ik.py` 用的標稱（未校正）UR5e 參數是根因 | 自己重建 DH 鏈驗證跟 `ur_analytic_ik.ur5e.forward_kinematics()` 在標稱參數下逐位元吻合，再代入校正參數量差距：home pose 附近 0.7~0.75mm，全關節隨機取樣 mean 1.05mm/max 1.87mm | **量級遠不足以解釋第 20 項的落差**（不到 2mm 沒辦法讓圓的直徑翻倍），但校正誤差本身是真實存在、可修的，見下一項 |
| 22 | 承上，決定把校正檔接進 `analytic_ik.py`：換成 MoveIt（IK 走 service）處理不了 cartesian 模式 250Hz 的頻率、之前也因為同樣理由撤回過三次（見第 7 項） | 不重新產生 IKFast（需要 OpenRAVE toolchain）：`forward_kinematics()` 直接用校正參數重建的 DH 鏈算（跟標稱模型同一套公式，已驗證逐位元吻合）；`solve()` 先用標稱 IKFast 解一個很接近的初始解，再用校正模型的數值 Jacobian 做 Newton 修正貼齊（初始解通常只差 < 2mm，1~2 次迭代收斂到微米級）。校正檔放進 `ur_control/config/`，`setup.py` 裝成 package data，`dp_control_ros2.py` 用 `ament_index_python` 找路徑、填進 `DPControlConfig.calibration_file` | round-trip 誤差降到微米級（<0.00001mm）。**代價**：Newton 修正比純 IKFast 慢（~900us vs ~20us/次），joint 模式只在 replan 時解（16 次/0.8秒，绰绰有餘）沒問題，cartesian 模式是每個 tick 都要解，`control_hz` 太高的話要注意這個開銷。`test_analytic_ik.py` 新增一輪校正模型 round-trip 測試，200 組全隨機姿態裡 8 組落在數值上驗證過的近奇異點（Jacobian 最小/最大奇異值比 < 0.002），正確被拒絕，不是修正邏輯的 bug |

### 問題 6 詳解：背景 executor + QoS + FK-based measured

三件事一起做：

1. `/joint_states` 訂閱 QoS 改 `BEST_EFFORT` + `depth=1`（`ur_arm_node.py`，全域生效）
2. 新增 `get_current_tcp_pose_analytic()`：直接讀快取的 `/joint_states` 算 FK，不查 TF
3. `dp_control_ros2.py` 改用背景 `SingleThreadedExecutor`（daemon thread 常駐 `spin_once()`），
   控制迴圈本身不再呼叫 `spin_once()`

**為什麼不用 `MultiThreadedExecutor`**：原本需要是因為 MoveIt IK 的
`spin_until_future_complete()` 會跟背景 executor 搶著 spin 同一個 node（經典 rclpy 死鎖）。
換成解析解後高頻路徑上已經沒有 service/action 呼叫，這個風險不存在了。

**執行緒排程規則（之後改動要保持）**：背景 executor 只在串流迴圈期間存在。啟動前（回
home、`wait_for_tcp_pose`、切換 controller）跟結束後都用 `URArmNode` 內建的阻塞方法（走全域
executor）——這時如果背景 executor 還在跑，同一個 node 會被兩個 executor 同時 spin，一樣會
死鎖。所以只在使用者按下 Enter 要開始串流那一刻才啟動背景 thread，`finally` 第一件事就是
停掉它。

**額外一起做的**：控制迴圈改成睡到下一個絕對 deadline（不是固定 `sleep(period)`，避免累積
誤差）；log 裡加印 `/joint_states` 實收頻率，當作背景 executor 的健康度監控。

**結果**：獨立讀取測試跟完整串流下都穩定量到 490~500Hz，跟 driver 端原生頻率吻合。

## 軌跡分析輸出

```bash
ros2 run ur_control analyze_trajectory <experiment_name>
```

獨立算兩組比較，因為它們回答不一樣的問題（見 `analyze_trajectory.py` 開頭註解）：

- **predicted vs actual**：模型「想要」的軌跡跟機器人「真正走到」的軌跡差多少——
  只比較「有推論、後來也真的被走到」的點（`T_a/T_p = 8/16 = 50%` 是 receding
  horizon 的定義，不是 bug），把 replan/interpolator/延遲前饋/controller 追蹤
  全部混在一起的總和誤差。
- **commanded vs measured**：控制迴圈每個 tick 真的送出去的 setpoint 跟機器人
  實際位置差多少——排除規劃面因素，純粹是低階追蹤（controller tracking）誤差。
  這組數字比 predicted vs actual 大是正常的（規劃階段的 interpolator 本來就會
  平滑掉一些高頻誤差）；如果這組數字本身偏大，問題在機器人執行端（增益、速度
  上限），不是模型或規劃邏輯。

輸出：

| 檔案 | 內容 |
| --- | --- |
| `<name>_predicted_vs_actual_overlay.png` | X/Y/Z/RX/RY/RZ 六軸，predicted vs actual 疊圖 |
| `<name>_predicted_vs_actual_error.png` | 六軸追蹤誤差 vs 時間，標題附 MAE/RMSE/MSE/R² |
| `<name>_commanded_vs_measured_overlay.png` | 同上，但 commanded vs measured |
| `<name>_commanded_vs_measured_error.png` | 同上，但 commanded vs measured |
| `<name>_metrics.txt` | 兩組數字版 + 整體位置誤差（歐氏距離）+ 姿態誤差（四元數測地線角度） |

姿態一律轉成 UR 旋轉向量比較，不比四元數分量（q 和 -q 同一個旋轉）。R² 在訊號幾乎不變的軸
（例如平面圓的 Z 軸）標成 `N/A` 並附原因，不印出誤導人的數字。圖上文字英文，console 中文。

## 依賴（pip，不是 ROS package）

| 套件 | 誰用 | 安裝 |
| --- | --- | --- |
| `ur_rtde` | `dp_control_rtde.py` | `pip3 install --user ur_rtde` |
| `ur-analytic-ik` | `analytic_ik.py`（`dp_control_ros2.py` 高頻 IK/FK） | `pip3 install --user ur-analytic-ik` |

`ur-analytic-ik` 目前寫死 `ur5e`，換機型改 `analytic_ik.py` 裡的 `UR_TYPE`。

### 出廠校正檔（`ur_control/config/my_robot_calibration.yaml`）

`analytic_ik.py` 預設用標稱（未校正）UR5e 參數；換一支手臂要用它自己的校正檔
（教導器 Installation -> Calibration 匯出），流程：把新的 yaml 覆蓋到
`ur_control/config/my_robot_calibration.yaml`，`colcon build --packages-select
ur_control` 重新裝一次 package data，`dp_control_ros2.py` 會自動透過
`ament_index_python` 找到裝好的路徑，不用改程式碼。細節/量級/效能代價見問題與
解決方法第 21~22 項。

## 待處理事項

- [ ] **cartesian 模式的關節限速調校**：見問題與解決方法第 14 項，`six_axis_circle`
      持續觸發關節限速，其他三種軌跡都準——待確認是測試軌跡本身太刁鑽、還是
      `max_joint_speed` 需要用真實規格值重調
- [ ] **`joint_accel_limit` 還沒在手臂上實測**：見問題與解決方法第 18 項，目前是
      保守起始值（2 rad/s²），需要在真機上確認還會不會跳 Protective Stop，穩定
      後再視情況調高
- [ ] **肉眼觀察夾爪軌跡像橢圓一事還沒有更嚴謹的量測驗證**：見問題與解決方法
      第 20 項，懷疑是視差/肉眼追蹤參考點誤差，還沒用相機或實體標記點覆核
- [ ] **`joint_accel_limit` 加上校正修正後還沒在手臂上重新驗證**：第 22 項的
      Newton 修正只在離線 round-trip 測試跑過，cartesian 模式每個 tick 多出來
      的 ~900us 開銷還沒在真機的高 `control_hz` 設定下實測過會不會拖慢串流

**下一步（等模型訓練好再做）**：

- [ ] **DP 真模型還沒接上**：`dp_control_ros2.py` 目前四個假軌跡 class（`_SixAxisCoupledCircleInference`
      等）都只是示範 `inference_fn` 介面長什麼樣子，換真模型時把 `_build_inference_fn()`
      換成建構真模型的 `inference_fn` 即可，`DPController`/`dp_controller.py` 其餘邏輯完全
      不用改；但要確認訓練資料收集端的 pipeline 延遲跟部署端（`latency_feedforward_seconds`）
      是否一致（DP 閉環，延遲不一致 policy 會失準——資料收集端程式碼不在這個 repo，需另外確認）
- [ ] **Robotiq URCap 衝突**：教導器要移除 `Robotiq_Gripper` URCap、改裝
      `rs485-1.0.urcap`（在 `~/ur_ws/src/Universal_Robots_ROS2_Driver/ur_robot_driver/resources/`），
      裝完用 `nc -zv <robot_ip> 54321` 確認連得上
- [ ] 上面那步做完後，還沒重新驗證 `ur_robotiq_bringup` 完整啟動 + 夾爪開合

## 啟動手臂 / 夾爪（在 `~/ur_ws`）

```bash
source ~/ur_ws/install/setup.bash
```

**純手臂**（兩個 terminal）：

```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.1.30
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e
```

**手臂 + Robotiq 2F-140 夾爪**（取代上面兩行）：

```bash
ros2 launch ur_robotiq_bringup ur_robotiq_control.launch.py ur_type:=ur5e robot_ip:=192.168.1.30 use_fake_hardware:=true
ros2 launch ur_robotiq_bringup ur_robotiq_moveit.launch.py ur_type:=ur5e
```

啟動前檢查：教導器 External Control 包 Loop 執行中；面板切 Remote Control；夾爪模式另需 rs485
URCap（見待處理事項）。

`dp_control_ros2.py` 不需要 MoveIt/`move_group`（IK 走 `compute_ik_analytic()`）。**第一次
測試務必先 `use_fake_hardware:=true` 在 RViz 驗證軌跡合理，再接真實手臂**。
