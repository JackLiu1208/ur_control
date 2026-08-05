#!/usr/bin/env python3
"""URArmNode: a reusable ROS2 node for controlling a UR arm.

Communication path matches the officially tested "external control" setup:
  - Joint-space targets  -> FollowJointTrajectory action on
    "<controller_name>/follow_joint_trajectory" (same pattern as
    Universal_Robots_ROS2_Driver's example_move.py / test_move.py).
  - Cartesian (end-effector pose) targets -> MoveIt2's MoveGroup action,
    which plans and then executes through the SAME scaled_joint_trajectory_controller,
    so on the wire it is still a FollowJointTrajectory goal to the robot.

This module only defines the Node (connection/communication layer). It does not
decide *what* to move to — that is left to whatever script imports this class,
whether it's a hand-written waypoint script or a model-inference loop.
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time as RclpyTime
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from builtin_interfaces.msg import Duration
from action_msgs.msg import GoalStatus
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory, GripperCommand

from geometry_msgs.msg import PoseStamped, Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    MoveItErrorCodes,
    RobotState,
    PositionIKRequest,
)
from tf2_ros import Buffer, TransformListener, TransformException

from ur_control.pose_utils import rotate_vector_by_quaternion, rotation_vector_to_quaternion
from ur_control import analytic_ik

# =============================================================================
# 可修改變數 (Configuration) — 依你的機器人 / driver 設定調整
# 這些是 URArmNode 建構子的預設值，皆可在建立 node 時以參數覆寫。
# =============================================================================

# Node 名稱 (可依用途自行命名，例如寫路徑點的程式跟模型輸出的程式可以取不同名字)
DEFAULT_NODE_NAME = "ur_arm_control_node"

# 六軸關節名稱與順序，須與 controller 設定 (ur_controllers.yaml) 中的 joints 順序一致
DEFAULT_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# 關節軌跡控制器名稱：與官方 external control 範例一致，走 FollowJointTrajectory action。
# 常見選項："scaled_joint_trajectory_controller"（預設，含速度縮放/安全機制）
#           "joint_trajectory_controller"（無速度縮放）
DEFAULT_CONTROLLER_NAME = "scaled_joint_trajectory_controller"

# 關節狀態 topic
DEFAULT_JOINT_STATES_TOPIC = "/joint_states"

# MoveIt2 設定（末端點 / Cartesian 控制用）
DEFAULT_MOVE_GROUP_NAME = "ur_manipulator"   # ur_moveit_config 預設 planning group
DEFAULT_BASE_LINK = "base_link"              # 規劃參考座標系 (chain base_link)
DEFAULT_TIP_LINK = "tool0"                   # MoveIt/tf 認得的實際末端 link（法蘭面），不要改成虛擬的 TCP 名稱
DEFAULT_MOVE_ACTION_NAME = "move_action"     # move_group 節點提供的 MoveGroup action 名稱

# TCP（工具中心點）相對 tip_link（法蘭面 tool0）的固定偏移，只有平移、在 tool0
# 自己的座標系下量測：預設沿法蘭面 Z 軸正向 25cm（例如裝了一支 25cm 長的工具）。
# 換了工具就改這裡；若工具還有旋轉偏移（不是單純沿 Z 伸出去），現在的實作不支援，
# 需要另外擴充。move_pose()/move_pose_linear() 的 position_xyz 與
# get_current_tcp_pose() 回傳的都是「這個 TCP 點」的座標，不是法蘭面本身。
DEFAULT_TCP_OFFSET_XYZ = (0.0, 0.0, 0.25)

# 預設運動時間 / 速度、加速度縮放
DEFAULT_JOINT_TIME_FROM_START = 3.0   # 秒；單點關節目標的預設到達時間
DEFAULT_GOAL_TIME_TOLERANCE = 0.5     # 秒；FollowJointTrajectory 到達時間容忍度
DEFAULT_VELOCITY_SCALING = 0.3        # MoveIt 規劃的最大速度縮放 (0~1)
DEFAULT_ACCELERATION_SCALING = 0.3    # MoveIt 規劃的最大加速度縮放 (0~1)
DEFAULT_PLANNING_TIME = 5.0           # MoveIt 規劃逾時秒數
DEFAULT_PLANNING_ATTEMPTS = 5         # MoveIt 規劃嘗試次數
DEFAULT_POSITION_TOLERANCE = 0.001    # 末端點位置容忍 (m)
DEFAULT_ORIENTATION_TOLERANCE = 0.01  # 末端點姿態容忍 (rad)

# 等待 action server / 關節狀態的逾時秒數
DEFAULT_SERVER_TIMEOUT = 10.0

# Robotiq 夾爪設定（需搭配 ur_robotiq_bringup 的 robotiq_gripper_controller，
# GripperActionController 走的是 finger_joint 自己的關節角度，單位 rad，不是真的
# 夾爪開口寬度公尺數，數值範圍要跟 xacro 的 gripper_closed_position 對上）
DEFAULT_GRIPPER_ACTION_NAME = "robotiq_gripper_controller/gripper_cmd"
DEFAULT_GRIPPER_OPEN_POSITION = 0.0     # finger_joint 全開
DEFAULT_GRIPPER_CLOSED_POSITION = 0.695  # finger_joint 全閉，需跟 urdf xacro 的 gripper_closed_position 一致
DEFAULT_GRIPPER_MAX_EFFORT = 100.0

# 直線 Cartesian 移動 (move_pose_linear，走 compute_cartesian_path，適合 jog 這種
# 需要快速響應、不想每次都跑完整 OMPL 規劃的場合)
DEFAULT_LINEAR_EEF_STEP = 0.005          # compute_cartesian_path 路徑點間距 (m)
DEFAULT_LINEAR_JUMP_THRESHOLD = 0.0      # 關節空間跳躍過濾閾值 (0 = 不啟用)
DEFAULT_LINEAR_MIN_FRACTION = 0.99       # 路徑至少要能規劃完成的比例，否則視為失敗
DEFAULT_LINEAR_RETIME_MAX_VELOCITY = 0.5  # velocity_scaling=1.0 時的最大關節角速度 (rad/s)，
                                           # 用來替 compute_cartesian_path 回傳的軌跡點重新配時
DEFAULT_MIN_TRAJECTORY_POINT_INTERVAL = 0.02  # 相鄰軌跡點最小時間間隔 (s)，避免時間戳重複


_MOVEIT_ERROR_CODE_NAMES = {
    value: name
    for name in dir(MoveItErrorCodes)
    if name.isupper()
    for value in [getattr(MoveItErrorCodes, name)]
    if isinstance(value, int)
}


def _moveit_error_code_to_str(error_code: int) -> str:
    return _MOVEIT_ERROR_CODE_NAMES.get(error_code, f"UNKNOWN({error_code})")


def _seconds_to_duration(seconds: float) -> Duration:
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    return Duration(sec=sec, nanosec=nanosec)


def _pose_to_constraints(pose_stamped: PoseStamped, tip_link: str,
                          position_tolerance: float, orientation_tolerance: float) -> Constraints:
    """Build MoveIt goal Constraints for a single target pose (mirrors what
    moveit_commander used to generate for set_pose_target)."""
    constraints = Constraints()

    position_constraint = PositionConstraint()
    position_constraint.header = pose_stamped.header
    position_constraint.link_name = tip_link
    position_constraint.weight = 1.0
    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [position_tolerance]
    bounding_volume = BoundingVolume()
    bounding_volume.primitives = [sphere]
    bounding_volume.primitive_poses = [pose_stamped.pose]
    position_constraint.constraint_region = bounding_volume
    constraints.position_constraints = [position_constraint]

    orientation_constraint = OrientationConstraint()
    orientation_constraint.header = pose_stamped.header
    orientation_constraint.link_name = tip_link
    orientation_constraint.orientation = pose_stamped.pose.orientation
    orientation_constraint.absolute_x_axis_tolerance = orientation_tolerance
    orientation_constraint.absolute_y_axis_tolerance = orientation_tolerance
    orientation_constraint.absolute_z_axis_tolerance = orientation_tolerance
    orientation_constraint.weight = 1.0
    constraints.orientation_constraints = [orientation_constraint]

    return constraints


def _retime_joint_trajectory(trajectory: JointTrajectory, max_velocity: float,
                              min_interval: float) -> JointTrajectory:
    """Assign time_from_start to a purely-geometric JointTrajectory (as returned
    by compute_cartesian_path, which does not time-parameterize its output) using
    a constant per-joint velocity limit."""
    retimed = JointTrajectory()
    retimed.joint_names = list(trajectory.joint_names)

    elapsed = 0.0
    previous_positions = None
    for point in trajectory.points:
        if previous_positions is not None:
            max_delta = max(abs(a - b) for a, b in zip(point.positions, previous_positions))
            elapsed += max(max_delta / max_velocity, min_interval)
        new_point = JointTrajectoryPoint()
        new_point.positions = list(point.positions)
        new_point.time_from_start = _seconds_to_duration(elapsed)
        retimed.points.append(new_point)
        previous_positions = point.positions

    return retimed


def _estimate_velocities(positions, times):
    """Central-difference velocity estimate at each sample, given per-sample
    positions and their time_from_start (both same length, times strictly
    increasing). First and last samples get zero velocity (safe "start and
    end at rest" boundary condition); everything in between gets a
    continuous, non-zero estimate.

    Without this, joint_trajectory_controller either assumes zero velocity at
    every waypoint you don't explicitly set (stop-start motion at each point)
    or, if you never set velocities at all, falls back to plain linear
    interpolation between points (an instantaneous velocity jump — a corner —
    at every waypoint). Feeding it a continuous velocity estimate instead
    gives smooth cubic-spline motion through the whole trajectory."""
    n = len(positions)
    dim = len(positions[0])
    velocities = [[0.0] * dim for _ in range(n)]
    for i in range(1, n - 1):
        dt = times[i + 1] - times[i - 1]
        if dt > 1e-9:
            velocities[i] = [(positions[i + 1][d] - positions[i - 1][d]) / dt for d in range(dim)]
    return velocities


def _retime_segment_to_duration(points, duration: float):
    """Distribute `duration` seconds across a purely-geometric list of
    JointTrajectoryPoint (points[0] is the segment's start state) proportionally
    to joint-space path length. Returns [(positions, time_from_start), ...] for
    points[1:] — the start point is dropped since callers stitch it onto the
    previous segment's end (or the robot's current state, for the first
    segment) rather than duplicating it."""
    deltas = [
        max(abs(a - b) for a, b in zip(points[i].positions, points[i - 1].positions))
        for i in range(1, len(points))
    ]
    total = sum(deltas)
    if total < 1e-9:
        return [(points[-1].positions, duration)]
    elapsed = 0.0
    result = []
    for i, delta in enumerate(deltas):
        elapsed += delta / total * duration
        result.append((points[i + 1].positions, elapsed))
    return result


class URArmNode(Node):
    """Reusable UR arm control node.

    Create one instance and call its move_* methods from any control script —
    hand-written waypoints and model-generated targets both go through the
    same node and the same communication path.
    """

    def __init__(self,
                 node_name: str = DEFAULT_NODE_NAME,
                 joint_names=None,
                 controller_name: str = DEFAULT_CONTROLLER_NAME,
                 joint_states_topic: str = DEFAULT_JOINT_STATES_TOPIC,
                 move_group_name: str = DEFAULT_MOVE_GROUP_NAME,
                 base_link: str = DEFAULT_BASE_LINK,
                 tip_link: str = DEFAULT_TIP_LINK,
                 move_action_name: str = DEFAULT_MOVE_ACTION_NAME,
                 default_joint_time_from_start: float = DEFAULT_JOINT_TIME_FROM_START,
                 velocity_scaling: float = DEFAULT_VELOCITY_SCALING,
                 acceleration_scaling: float = DEFAULT_ACCELERATION_SCALING,
                 planning_time: float = DEFAULT_PLANNING_TIME,
                 planning_attempts: int = DEFAULT_PLANNING_ATTEMPTS,
                 position_tolerance: float = DEFAULT_POSITION_TOLERANCE,
                 orientation_tolerance: float = DEFAULT_ORIENTATION_TOLERANCE,
                 server_timeout: float = DEFAULT_SERVER_TIMEOUT,
                 linear_eef_step: float = DEFAULT_LINEAR_EEF_STEP,
                 linear_jump_threshold: float = DEFAULT_LINEAR_JUMP_THRESHOLD,
                 linear_min_fraction: float = DEFAULT_LINEAR_MIN_FRACTION,
                 linear_retime_max_velocity: float = DEFAULT_LINEAR_RETIME_MAX_VELOCITY,
                 min_trajectory_point_interval: float = DEFAULT_MIN_TRAJECTORY_POINT_INTERVAL,
                 tcp_offset_xyz=DEFAULT_TCP_OFFSET_XYZ,
                 gripper_action_name: str = DEFAULT_GRIPPER_ACTION_NAME,
                 gripper_open_position: float = DEFAULT_GRIPPER_OPEN_POSITION,
                 gripper_closed_position: float = DEFAULT_GRIPPER_CLOSED_POSITION,
                 gripper_max_effort: float = DEFAULT_GRIPPER_MAX_EFFORT):
        super().__init__(node_name)

        self.joint_names = list(joint_names) if joint_names else list(DEFAULT_JOINT_NAMES)
        self.controller_name = controller_name
        self.move_group_name = move_group_name
        self.base_link = base_link
        self.tip_link = tip_link
        self.default_joint_time_from_start = default_joint_time_from_start
        self.velocity_scaling = velocity_scaling
        self.acceleration_scaling = acceleration_scaling
        self.planning_time = planning_time
        self.planning_attempts = planning_attempts
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.server_timeout = server_timeout
        self.linear_eef_step = linear_eef_step
        self.linear_jump_threshold = linear_jump_threshold
        self.linear_min_fraction = linear_min_fraction
        self.linear_retime_max_velocity = linear_retime_max_velocity
        self.min_trajectory_point_interval = min_trajectory_point_interval
        self.tcp_offset_xyz = tuple(tcp_offset_xyz)
        self.gripper_open_position = gripper_open_position
        self.gripper_closed_position = gripper_closed_position
        self.gripper_max_effort = gripper_max_effort

        self._latest_joint_state = None
        # 每收到一次 /joint_states 就 +1。純粹的健康度診斷用：拿兩個時間點的差
        # 除以經過時間，就是這個 subscription 實際被服務的頻率——直接證實/推翻
        # 「背景 executor 有沒有真的在跑」，不用猜測。
        self.joint_state_message_count = 0

        self._jtc_action_client = ActionClient(
            self, FollowJointTrajectory, f"{controller_name}/follow_joint_trajectory")
        self._move_group_action_client = ActionClient(self, MoveGroup, move_action_name)
        self._cartesian_path_client = self.create_client(
            GetCartesianPath, "compute_cartesian_path")
        self._ik_client = self.create_client(GetPositionIK, "compute_ik")
        self._gripper_action_client = ActionClient(self, GripperCommand, gripper_action_name)
        # /joint_states 是高頻感測資料（driver 端通常 500Hz），用 RELIABLE+depth 10
        # 這種預設 QoS 在訊息處理跟不上時會累積 backlog、越補越舊；BEST_EFFORT+depth 1
        # 才是這種「只要最新值、不要求每一筆都送達」的資料該用的設定，跟 RELIABLE
        # publisher 相容（DDS QoS 相容規則允許 BEST_EFFORT subscriber 接 RELIABLE
        # publisher）。
        joint_state_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._joint_state_sub = self.create_subscription(
            JointState, joint_states_topic, self._joint_state_callback, joint_state_qos)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.get_logger().info(
            f"URArmNode '{node_name}' 已建立 "
            f"(controller={controller_name}, move_group={move_group_name})")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def wait_for_servers(self, timeout_sec: float = None) -> bool:
        """Block until both the joint-trajectory action server and MoveIt's
        MoveGroup action server are available."""
        timeout_sec = self.server_timeout if timeout_sec is None else timeout_sec
        ok_jtc = self._jtc_action_client.wait_for_server(timeout_sec=timeout_sec)
        if not ok_jtc:
            self.get_logger().error(
                f"等不到 action server: {self.controller_name}/follow_joint_trajectory")
        ok_move_group = self._move_group_action_client.wait_for_server(timeout_sec=timeout_sec)
        if not ok_move_group:
            self.get_logger().error("等不到 MoveGroup action server (move_action)")
        return ok_jtc and ok_move_group

    def _joint_state_callback(self, msg: JointState):
        self._latest_joint_state = msg
        self.joint_state_message_count += 1

    def get_current_joint_positions(self, timeout_sec: float = 5.0):
        """Return current joint positions (list, ordered as self.joint_names),
        or None if no /joint_states message arrived within timeout_sec."""
        end_time = time.time() + timeout_sec
        while self._latest_joint_state is None and time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._latest_joint_state is None:
            return None
        name_to_position = dict(zip(self._latest_joint_state.name, self._latest_joint_state.position))
        return [name_to_position[name] for name in self.joint_names]

    def peek_joint_positions(self):
        """Non-blocking read of the last received /joint_states (ordered as
        self.joint_names), or None if nothing has arrived yet. Does not spin —
        intended for callers (e.g. a GUI) that already pump the executor
        themselves and just want the latest cached value."""
        if self._latest_joint_state is None:
            return None
        name_to_position = dict(zip(self._latest_joint_state.name, self._latest_joint_state.position))
        return [name_to_position[name] for name in self.joint_names]

    def peek_joint_velocities(self):
        """Non-blocking read of the last received /joint_states velocities
        (ordered as self.joint_names), or None if unavailable (no message yet,
        or the velocity field is empty/mismatched)."""
        state = self._latest_joint_state
        if state is None or len(state.velocity) != len(state.name):
            return None
        name_to_velocity = dict(zip(state.name, state.velocity))
        return [name_to_velocity[name] for name in self.joint_names]

    def _tip_pose_to_tcp_pose(self, position, orientation):
        """tip_link (tool0/flange) pose -> TCP pose, applying tcp_offset_xyz
        (measured in tool0's own frame) rotated into base_link."""
        offset = rotate_vector_by_quaternion(self.tcp_offset_xyz, orientation)
        return (position[0] + offset[0], position[1] + offset[1], position[2] + offset[2]), orientation

    def _tcp_pose_to_tip_pose(self, position, orientation):
        """Inverse of _tip_pose_to_tcp_pose(): TCP pose -> tip_link pose."""
        offset = rotate_vector_by_quaternion(self.tcp_offset_xyz, orientation)
        return (position[0] - offset[0], position[1] - offset[1], position[2] - offset[2]), orientation

    def get_current_tcp_pose(self):
        """Non-blocking lookup of the current TCP pose (tip_link + tcp_offset_xyz)
        in base_link frame via tf2, using whatever transforms are already in the
        local buffer. Returns ((x, y, z), (qx, qy, qz, qw)) or None if unavailable."""
        try:
            transform = self._tf_buffer.lookup_transform(
                self.base_link, self.tip_link, RclpyTime())
        except TransformException:
            return None
        t = transform.transform.translation
        r = transform.transform.rotation
        return self._tip_pose_to_tcp_pose((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))

    def get_current_tcp_pose_analytic(self):
        """跟 get_current_tcp_pose() 一樣的回傳格式，但不查 TF，直接用目前快取的
        /joint_states（self._latest_joint_state）算 FK（ur_control.analytic_ik，
        微秒等級）。

        高頻路徑請用這個，不要用 TF 版本：TF 的 base_link->tip_link transform
        只有在 TransformListener 內部的 /tf subscription callback 被 executor
        服務到的時候才會前進，如果控制迴圈是自己手動 rclpy.spin_once() 或背景
        executor 沒有確實在跑，TF 會遠遠落後於 /joint_states 實際的發布頻率
        （這是實測踩過的坑：曾經量到 TF 版本的量測路徑只有 1-2Hz 有效更新，
        即使 /joint_states 本身是 500Hz）。這個方法完全不依賴 TF/executor 排程，
        只要 self._latest_joint_state 是新的，這裡算出來的就是新的。

        Returns ((x, y, z), (qx, qy, qz, qw)) or None（還沒收到 /joint_states，
        或收到的訊息缺關節名稱）。"""
        if self._latest_joint_state is None:
            return None
        name_to_position = dict(zip(
            self._latest_joint_state.name, self._latest_joint_state.position))
        try:
            joint_positions = [name_to_position[name] for name in self.joint_names]
        except KeyError:
            return None
        tip_position, tip_orientation = analytic_ik.forward_kinematics(joint_positions)
        return self._tip_pose_to_tcp_pose(tip_position, tip_orientation)

    def wait_for_tcp_pose(self, timeout_sec: float = 5.0):
        """Blocking version of get_current_tcp_pose(): spins until the
        base_link -> tip_link transform becomes available or timeout_sec
        elapses. Useful right after startup, before robot_state_publisher
        has had a chance to publish /tf. Returns the same shape as
        get_current_tcp_pose(), or None on timeout (e.g. robot_state_publisher
        isn't running, or tip_link/base_link don't match the actual TF tree —
        double check for a tf_prefix mismatch in that case)."""
        end_time = time.time() + timeout_sec
        pose = self.get_current_tcp_pose()
        while pose is None and time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)
            pose = self.get_current_tcp_pose()
        return pose

    # ------------------------------------------------------------------
    # Joint-space control (FollowJointTrajectory)
    # ------------------------------------------------------------------
    def move_joint(self, positions, time_from_start: float = None,
                    joint_names=None, wait: bool = True,
                    goal_time_tolerance: float = DEFAULT_GOAL_TIME_TOLERANCE) -> bool:
        """Send a single-point joint-space target."""
        joint_names = joint_names or self.joint_names
        if len(positions) != len(joint_names):
            raise ValueError(
                f"positions 長度 ({len(positions)}) 與 joint_names 長度 ({len(joint_names)}) 不符")
        time_from_start = (self.default_joint_time_from_start
                            if time_from_start is None else time_from_start)

        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.velocities = [0.0] * len(positions)
        point.time_from_start = _seconds_to_duration(time_from_start)

        trajectory = JointTrajectory()
        trajectory.joint_names = list(joint_names)
        trajectory.points = [point]

        return self.move_joint_trajectory(trajectory, wait=wait,
                                           goal_time_tolerance=goal_time_tolerance)

    def move_joint_waypoints(self, waypoints, joint_names=None, wait: bool = True,
                              goal_time_tolerance: float = DEFAULT_GOAL_TIME_TOLERANCE) -> bool:
        """Send a multi-point joint-space trajectory in a single goal.

        waypoints: list of dicts, each with:
          - "positions": list[float] (required)
          - "time_from_start": float seconds from trajectory start (required)
          - "velocities": list[float] (optional; if omitted, a continuous
            estimate is filled in automatically — see _estimate_velocities —
            instead of defaulting to zero, so the robot doesn't stop at every
            intermediate waypoint)
        """
        joint_names = joint_names or self.joint_names
        positions = [list(wp["positions"]) for wp in waypoints]
        times = [wp["time_from_start"] for wp in waypoints]
        estimated_velocities = _estimate_velocities(positions, times)

        trajectory = JointTrajectory()
        trajectory.joint_names = list(joint_names)
        for wp, position, time_from_start, estimated in zip(
                waypoints, positions, times, estimated_velocities):
            point = JointTrajectoryPoint()
            point.positions = position
            point.velocities = list(wp["velocities"]) if "velocities" in wp else estimated
            point.time_from_start = _seconds_to_duration(time_from_start)
            trajectory.points.append(point)

        return self.move_joint_trajectory(trajectory, wait=wait,
                                           goal_time_tolerance=goal_time_tolerance)

    def move_joint_trajectory(self, trajectory: JointTrajectory, wait: bool = True,
                               goal_time_tolerance: float = DEFAULT_GOAL_TIME_TOLERANCE) -> bool:
        """Send a pre-built JointTrajectory via FollowJointTrajectory action."""
        if not self._jtc_action_client.wait_for_server(timeout_sec=self.server_timeout):
            self.get_logger().error(
                f"等不到 action server: {self.controller_name}/follow_joint_trajectory")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance = _seconds_to_duration(goal_time_tolerance)

        send_goal_future = self._jtc_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("關節軌跡目標被拒絕")
            return False

        if not wait:
            return True

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        success = result.status == GoalStatus.STATUS_SUCCEEDED
        if not success:
            self.get_logger().error(
                f"關節軌跡執行失敗: status={result.status}, "
                f"error_code={result.result.error_code}, "
                f"error_string={result.result.error_string}")
        return success

    # ------------------------------------------------------------------
    # Cartesian control (MoveIt2 MoveGroup: plan + execute)
    # ------------------------------------------------------------------
    def move_pose(self, position_xyz, orientation_xyzw, wait: bool = True,
                   velocity_scaling: float = None, acceleration_scaling: float = None,
                   planning_time: float = None, planning_attempts: int = None,
                   position_tolerance: float = None, orientation_tolerance: float = None) -> bool:
        """Move the TCP (tip_link + tcp_offset_xyz) to a target pose. IK/planning
        is done by MoveIt2 for tip_link (offset removed internally); execution
        is handed off to the same trajectory controller used for move_joint()."""
        if not self._move_group_action_client.wait_for_server(timeout_sec=self.server_timeout):
            self.get_logger().error("等不到 MoveGroup action server (move_action)")
            return False

        tip_position, tip_orientation = self._tcp_pose_to_tip_pose(position_xyz, orientation_xyzw)

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_link
        pose_stamped.pose.position.x = float(tip_position[0])
        pose_stamped.pose.position.y = float(tip_position[1])
        pose_stamped.pose.position.z = float(tip_position[2])
        pose_stamped.pose.orientation.x = float(tip_orientation[0])
        pose_stamped.pose.orientation.y = float(tip_orientation[1])
        pose_stamped.pose.orientation.z = float(tip_orientation[2])
        pose_stamped.pose.orientation.w = float(tip_orientation[3])

        constraints = _pose_to_constraints(
            pose_stamped, self.tip_link,
            self.position_tolerance if position_tolerance is None else position_tolerance,
            self.orientation_tolerance if orientation_tolerance is None else orientation_tolerance)

        request = MotionPlanRequest()
        request.group_name = self.move_group_name
        request.goal_constraints = [constraints]
        request.num_planning_attempts = (
            self.planning_attempts if planning_attempts is None else planning_attempts)
        request.allowed_planning_time = (
            self.planning_time if planning_time is None else planning_time)
        request.max_velocity_scaling_factor = (
            self.velocity_scaling if velocity_scaling is None else velocity_scaling)
        request.max_acceleration_scaling_factor = (
            self.acceleration_scaling if acceleration_scaling is None else acceleration_scaling)
        request.start_state.is_diff = True  # 以目前機器人狀態作為規劃起點

        planning_options = PlanningOptions()
        planning_options.plan_only = False  # 規劃後直接執行

        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options = planning_options

        send_goal_future = self._move_group_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("MoveGroup 目標被拒絕")
            return False

        if not wait:
            return True

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        error_code = result.result.error_code.val
        success = error_code == MoveItErrorCodes.SUCCESS
        if not success:
            self.get_logger().error(
                f"MoveGroup 規劃/執行失敗: error_code={error_code} "
                f"({_moveit_error_code_to_str(error_code)})")
        return success

    def move_pose_linear(self, position_xyz, orientation_xyzw, wait: bool = True,
                          velocity_scaling: float = None,
                          eef_step: float = None, jump_threshold: float = None,
                          goal_time_tolerance: float = DEFAULT_GOAL_TIME_TOLERANCE) -> bool:
        """Straight-line Cartesian move: compute the path with MoveIt's
        compute_cartesian_path service (fast, no OMPL sampling) and execute it
        through the same FollowJointTrajectory action client used by
        move_joint(). position_xyz/orientation_xyzw target the TCP (tip_link +
        tcp_offset_xyz); the offset is removed internally before asking MoveIt
        to solve for tip_link. Intended for responsive Cartesian jogging; use
        move_pose() for one-shot free-space planning that also honors
        acceleration scaling."""
        if self._latest_joint_state is None:
            self.get_logger().error("尚未收到 /joint_states，無法取得起始關節狀態")
            return False
        if not self._cartesian_path_client.wait_for_service(timeout_sec=self.server_timeout):
            self.get_logger().error("等不到 compute_cartesian_path service")
            return False

        tip_position, tip_orientation = self._tcp_pose_to_tip_pose(position_xyz, orientation_xyzw)

        target_pose = Pose()
        target_pose.position.x = float(tip_position[0])
        target_pose.position.y = float(tip_position[1])
        target_pose.position.z = float(tip_position[2])
        target_pose.orientation.x = float(tip_orientation[0])
        target_pose.orientation.y = float(tip_orientation[1])
        target_pose.orientation.z = float(tip_orientation[2])
        target_pose.orientation.w = float(tip_orientation[3])

        request = GetCartesianPath.Request()
        request.header.frame_id = self.base_link
        request.start_state = RobotState(joint_state=self._latest_joint_state, is_diff=False)
        request.group_name = self.move_group_name
        request.link_name = self.tip_link
        request.waypoints = [target_pose]
        request.max_step = self.linear_eef_step if eef_step is None else eef_step
        request.jump_threshold = (
            self.linear_jump_threshold if jump_threshold is None else jump_threshold)
        request.avoid_collisions = True

        response_future = self._cartesian_path_client.call_async(request)
        rclpy.spin_until_future_complete(self, response_future, timeout_sec=self.server_timeout)
        response = response_future.result()
        if response is None or response.error_code.val != MoveItErrorCodes.SUCCESS:
            error_code = response.error_code.val if response else None
            self.get_logger().error(
                f"compute_cartesian_path 呼叫失敗: error_code={error_code} "
                f"({_moveit_error_code_to_str(error_code) if response else 'no response'})")
            return False
        if response.fraction < self.linear_min_fraction:
            self.get_logger().error(
                f"Cartesian 路徑只規劃出 {response.fraction:.2f}（需要 >= "
                f"{self.linear_min_fraction:.2f}），可能超出可達範圍或碰到奇異點，取消執行")
            return False

        velocity_scaling = self.velocity_scaling if velocity_scaling is None else velocity_scaling
        max_velocity = max(self.linear_retime_max_velocity * velocity_scaling, 1e-3)
        trajectory = _retime_joint_trajectory(
            response.solution.joint_trajectory, max_velocity, self.min_trajectory_point_interval)

        return self.move_joint_trajectory(trajectory, wait=wait,
                                           goal_time_tolerance=goal_time_tolerance)

    def compute_ik(self, position_xyz, orientation_xyzw, seed_positions=None,
                    timeout_sec: float = 0.1):
        """Solve IK for a single TCP pose (tip_link + tcp_offset_xyz), via
        MoveIt's /compute_ik service — this does NOT move the robot, it just
        returns a joint solution (list, ordered as self.joint_names) or None
        if no solution / service unavailable.

        Meant for batch use (solve a whole model action-chunk once, up
        front, then stream the results through a local interpolator) —
        each call is still one ROS service round-trip (tens of ms typically),
        far too slow to call once per high-rate control tick.

        seed_positions: joint positions (ordered as self.joint_names) to seed
        the solver near, so consecutive chunk points resolve to the same arm
        configuration branch instead of jumping between IK solutions. Falls
        back to the robot's current joint state if not given."""
        if not self._ik_client.wait_for_service(timeout_sec=self.server_timeout):
            self.get_logger().error("等不到 compute_ik service")
            return None

        tip_position, tip_orientation = self._tcp_pose_to_tip_pose(position_xyz, orientation_xyzw)

        if seed_positions is not None:
            seed_state = JointState()
            seed_state.name = list(self.joint_names)
            seed_state.position = list(seed_positions)
        elif self._latest_joint_state is not None:
            seed_state = self._latest_joint_state
        else:
            self.get_logger().error("沒有 seed_positions，也還沒收到 /joint_states，無法求解 IK")
            return None

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_link
        pose_stamped.pose.position.x = float(tip_position[0])
        pose_stamped.pose.position.y = float(tip_position[1])
        pose_stamped.pose.position.z = float(tip_position[2])
        pose_stamped.pose.orientation.x = float(tip_orientation[0])
        pose_stamped.pose.orientation.y = float(tip_orientation[1])
        pose_stamped.pose.orientation.z = float(tip_orientation[2])
        pose_stamped.pose.orientation.w = float(tip_orientation[3])

        request = GetPositionIK.Request()
        request.ik_request = PositionIKRequest()
        request.ik_request.group_name = self.move_group_name
        request.ik_request.robot_state = RobotState(joint_state=seed_state, is_diff=False)
        request.ik_request.pose_stamped = pose_stamped
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout = _seconds_to_duration(timeout_sec)

        response_future = self._ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, response_future, timeout_sec=self.server_timeout)
        response = response_future.result()
        if response is None or response.error_code.val != MoveItErrorCodes.SUCCESS:
            error_code = response.error_code.val if response else None
            self.get_logger().error(
                f"compute_ik 失敗: error_code={error_code} "
                f"({_moveit_error_code_to_str(error_code) if response else 'no response'})")
            return None

        name_to_position = dict(zip(
            response.solution.joint_state.name, response.solution.joint_state.position))
        return [name_to_position[name] for name in self.joint_names]

    def compute_ik_analytic(self, position_xyz, orientation_xyzw, seed_positions):
        """跟 compute_ik() 一樣的介面（TCP offset 處理相同），但用
        ur_control.analytic_ik（PickNik ur-analytic-ik，IKFast 閉式解），完全
        in-process、微秒等級，不呼叫 MoveIt service。適合高頻控制迴圈路徑：
        compute_ik() 每次呼叫是一次 ROS service round-trip（十幾到幾十 ms），
        一個 chunk 呼叫十幾次會在控制迴圈裡造成明顯的 dt 尖峰，這個方法不會。

        跟 compute_ik() 的重要差異：**不做碰撞檢查**（需要 MoveIt，跟這個方法
        存在的目的矛盾），而且 seed_positions 是必填（UR 一個姿態最多 8 組解，
        沒有 seed 沒辦法決定要哪一組）。回傳 None 表示無解，或者解通過內部
        round-trip 驗證失敗（見 analytic_ik.solve()），呼叫端應該當作「這個點
        求不出來」處理，不要送出去。"""
        tip_position, tip_orientation = self._tcp_pose_to_tip_pose(position_xyz, orientation_xyzw)
        return analytic_ik.solve(tip_position, tip_orientation, seed_positions)

    def move_pose_waypoints(self, waypoints, wait: bool = True,
                             eef_step: float = None, jump_threshold: float = None,
                             goal_time_tolerance: float = DEFAULT_GOAL_TIME_TOLERANCE) -> bool:
        """Continuous multi-point Cartesian trajectory: plans a straight-line
        segment (compute_cartesian_path) between each consecutive pair of
        waypoints — starting from the robot's current pose — and stitches
        all segments into ONE JointTrajectory, so the whole path executes as
        a single FollowJointTrajectory goal without stopping at each
        waypoint (unlike calling move_pose_linear() once per point).

        waypoints: list of dicts, each with:
          - "position": (x, y, z) TCP position in base_link frame, meters
          - "orientation": quaternion (x, y, z, w) — OR —
            "rotation_vector": (rx, ry, rz), radians, same axis-angle
            convention as pose_utils.rotation_vector_to_quaternion(). Both
            are in base_link frame, NOT the UR teach-pendant "Base" frame —
            convert with pose_utils.ur_base_to_base_link() first if your
            numbers come from the pendant.
          - "time_from_start": seconds from the START of the whole
            trajectory (must be strictly increasing across the list)
        """
        if self._latest_joint_state is None:
            self.get_logger().error("尚未收到 /joint_states，無法取得起始關節狀態")
            return False
        if not self._cartesian_path_client.wait_for_service(timeout_sec=self.server_timeout):
            self.get_logger().error("等不到 compute_cartesian_path service")
            return False

        eef_step = self.linear_eef_step if eef_step is None else eef_step
        jump_threshold = self.linear_jump_threshold if jump_threshold is None else jump_threshold

        combined_points = []
        combined_times = []
        joint_names = None
        start_state = RobotState(joint_state=self._latest_joint_state, is_diff=False)
        previous_time_from_start = 0.0

        for i, waypoint in enumerate(waypoints):
            quat = waypoint.get("orientation")
            if quat is None:
                quat = rotation_vector_to_quaternion(*waypoint["rotation_vector"])
            tip_position, tip_orientation = self._tcp_pose_to_tip_pose(waypoint["position"], quat)

            target_pose = Pose()
            target_pose.position.x = float(tip_position[0])
            target_pose.position.y = float(tip_position[1])
            target_pose.position.z = float(tip_position[2])
            target_pose.orientation.x = float(tip_orientation[0])
            target_pose.orientation.y = float(tip_orientation[1])
            target_pose.orientation.z = float(tip_orientation[2])
            target_pose.orientation.w = float(tip_orientation[3])

            request = GetCartesianPath.Request()
            request.header.frame_id = self.base_link
            request.start_state = start_state
            request.group_name = self.move_group_name
            request.link_name = self.tip_link
            request.waypoints = [target_pose]
            request.max_step = eef_step
            request.jump_threshold = jump_threshold
            request.avoid_collisions = True

            response_future = self._cartesian_path_client.call_async(request)
            rclpy.spin_until_future_complete(self, response_future, timeout_sec=self.server_timeout)
            response = response_future.result()
            if response is None or response.error_code.val != MoveItErrorCodes.SUCCESS:
                error_code = response.error_code.val if response else None
                self.get_logger().error(
                    f"第 {i} 個路徑點 compute_cartesian_path 失敗: error_code={error_code} "
                    f"({_moveit_error_code_to_str(error_code) if response else 'no response'})")
                return False
            if response.fraction < self.linear_min_fraction:
                self.get_logger().error(
                    f"第 {i} 個路徑點只規劃出 {response.fraction:.2f}（需要 >= "
                    f"{self.linear_min_fraction:.2f}），可能超出可達範圍或碰到奇異點，取消執行")
                return False

            segment = response.solution.joint_trajectory
            if joint_names is None:
                joint_names = list(segment.joint_names)
            if len(segment.points) < 2:
                self.get_logger().error(f"第 {i} 個路徑點跟前一點幾乎重合，無法規劃有意義的路徑")
                return False

            segment_duration = waypoint["time_from_start"] - previous_time_from_start
            if segment_duration <= 0:
                self.get_logger().error(f"第 {i} 個路徑點的 time_from_start 必須比前一個路徑點大")
                return False

            for positions, t in _retime_segment_to_duration(segment.points, segment_duration):
                point = JointTrajectoryPoint()
                point.positions = list(positions)
                absolute_time = previous_time_from_start + t
                point.time_from_start = _seconds_to_duration(absolute_time)
                combined_points.append(point)
                combined_times.append(absolute_time)

            previous_time_from_start = waypoint["time_from_start"]

            # 下一段的規劃起點 = 這一段規劃出的終點關節角度
            end_joint_state = JointState()
            end_joint_state.name = list(segment.joint_names)
            end_joint_state.position = list(segment.points[-1].positions)
            start_state = RobotState(joint_state=end_joint_state, is_diff=False)

        # 對整條合併後的軌跡（跨所有路徑點）估計速度，讓控制器用平滑的三次曲線插值，
        # 而不是每個路徑點都當成停頓點、或路徑點之間出現速度瞬間跳變的線性插值。
        estimated_velocities = _estimate_velocities(
            [p.positions for p in combined_points], combined_times)
        for point, velocity in zip(combined_points, estimated_velocities):
            point.velocities = velocity

        trajectory = JointTrajectory()
        trajectory.joint_names = joint_names
        trajectory.points = combined_points

        return self.move_joint_trajectory(trajectory, wait=wait,
                                           goal_time_tolerance=goal_time_tolerance)

    # ------------------------------------------------------------------
    # Gripper control (Robotiq 2F-140, via GripperActionController)
    # ------------------------------------------------------------------
    def set_gripper_position(self, position: float, max_effort: float = None,
                              wait: bool = True) -> bool:
        """Send a target position to the gripper's finger_joint (rad; 0.0 =
        open, self.gripper_closed_position = closed). Requires
        ur_robotiq_bringup's robotiq_gripper_controller to be running."""
        if not self._gripper_action_client.wait_for_server(timeout_sec=self.server_timeout):
            self.get_logger().error("等不到 gripper action server (robotiq_gripper_controller)")
            return False

        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = self.gripper_max_effort if max_effort is None else max_effort

        send_goal_future = self._gripper_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("夾爪目標被拒絕")
            return False

        if not wait:
            return True

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        success = result.status == GoalStatus.STATUS_SUCCEEDED
        if not success:
            self.get_logger().error(f"夾爪動作失敗: status={result.status}")
        return success

    def open_gripper(self, wait: bool = True) -> bool:
        return self.set_gripper_position(self.gripper_open_position, wait=wait)

    def close_gripper(self, wait: bool = True) -> bool:
        return self.set_gripper_position(self.gripper_closed_position, wait=wait)
