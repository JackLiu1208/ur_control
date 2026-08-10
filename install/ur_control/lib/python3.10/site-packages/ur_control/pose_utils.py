"""Conversion helpers between quaternions (used internally by URArmNode /
MoveIt) and UR's rotation-vector pose representation (rx, ry, rz — the same
axis*angle convention shown on the teach pendant and used by URScript)."""

import math


def quaternion_to_rotation_vector(x: float, y: float, z: float, w: float):
    """quaternion (xyzw) -> UR rotation vector (rx, ry, rz), radians.

    A quaternion q and -q represent the same rotation; forcing w >= 0 here
    picks the minimal-angle (<= pi) representative, matching what the UR
    teach pendant / URScript display."""
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-9:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(norm, w)
    return (x / norm * angle, y / norm * angle, z / norm * angle)


def rotation_vector_to_quaternion(rx: float, ry: float, rz: float):
    """UR rotation vector (rx, ry, rz), radians -> quaternion (x, y, z, w)."""
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    half_sin = math.sin(angle / 2.0) / angle
    return (rx * half_sin, ry * half_sin, rz * half_sin, math.cos(angle / 2.0))


def rotate_vector_by_quaternion(v, q):
    """Rotate 3D vector v=(x,y,z) by quaternion q (xyzw). Used to express a
    tool-frame offset (e.g. TCP relative to the flange) in the parent frame."""
    vx, vy, vz = v
    x, y, z, w = q
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def quaternion_conjugate(q):
    """Conjugate (= inverse, for a unit quaternion) of q (xyzw): represents
    the reverse rotation of q."""
    x, y, z, w = q
    return (-x, -y, -z, w)


def quaternion_multiply(q1, q2):
    """Hamilton product q1 * q2 (both xyzw). Represents "apply q2, then q1"."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


# UR's controller/teach-pendant "Base" frame and the ROS "base_link" frame
# (REP-103 aligned: X+ forward, Y+ left, Z+ up) share the same origin but
# differ by a 180 degree rotation about Z — see ur_description's
# base_link-base_fixed_joint (ur_macro.xacro). This is the quaternion for
# that fixed rotation.
_QUAT_BASE_LINK_FROM_BASE = (0.0, 0.0, 1.0, 0.0)


def ur_base_to_base_link(position_xyz, rotation_vector):
    """Convert a TCP pose expressed in the UR controller's 'Base' frame
    (position in meters, orientation as a rotation vector rx/ry/rz in
    radians — exactly what the teach pendant / URScript show) into
    (position_xyz, quaternion_xyzw) in the ROS 'base_link' frame, ready to
    pass to URArmNode.move_pose() / move_pose_linear().

    Empirically confirmed against a real robot (2026-07): UR's rotation
    vector represents the tool->base orientation (q_tool0_base), the reverse
    of the "orientation of tool0 in base" convention used everywhere else in
    this module — hence the extra conjugate below."""
    x, y, z = position_xyz
    position_base_link = (-x, -y, z)
    quat_tool0_base = rotation_vector_to_quaternion(*rotation_vector)
    quat_base_tool0 = quaternion_conjugate(quat_tool0_base)
    quat_base_link_tool0 = quaternion_multiply(_QUAT_BASE_LINK_FROM_BASE, quat_base_tool0)
    return position_base_link, quat_base_link_tool0


def base_link_to_ur_base(position_xyz, rotation_vector):
    """Inverse of ur_base_to_base_link(): convert a pose given as (position in
    meters, rotation vector in radians) in the ROS 'base_link' frame into the
    same representation in the UR controller's 'Base' frame (as shown on the
    teach pendant)."""
    x, y, z = position_xyz
    position_base = (-x, -y, z)
    quat_base_link_tool0 = rotation_vector_to_quaternion(*rotation_vector)
    quat_base_tool0 = quaternion_multiply(
        quaternion_conjugate(_QUAT_BASE_LINK_FROM_BASE), quat_base_link_tool0)
    quat_tool0_base = quaternion_conjugate(quat_base_tool0)
    rotation_vector_base = quaternion_to_rotation_vector(*quat_tool0_base)
    return position_base, rotation_vector_base
