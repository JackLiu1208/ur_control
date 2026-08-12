cat > ~/env.sh << 'EOF'
#!/bin/bash
# Lab UR5e ROS2 環境設定
source /opt/ros/humble/setup.bash
source ~/ur_ws/install/setup.bash
source ~/ros2_ws/install/setup.bash
echo "[env.sh] ROS2 Humble + ur_ws + ros2_ws 已載入"
EOF
chmod +x ~/env.sh
