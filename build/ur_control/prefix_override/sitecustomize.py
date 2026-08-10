import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jackliu1017/ros2_ws/src/ur_control/install/ur_control'
