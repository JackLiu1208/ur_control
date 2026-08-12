from setuptools import find_packages, setup

package_name = 'ur_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['ur_control/config/my_robot_calibration.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jackliu1017',
    maintainer_email='804jack@gmail.com',
    description=(
        'UR arm joint/Cartesian control node sharing the official external-control '
        'communication path (FollowJointTrajectory / MoveIt2).'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run_waypoints = ur_control.examples.run_waypoints:main',
            'monitor_state = ur_control.examples.monitor_state:main',
            'gripper_demo = ur_control.examples.gripper_demo:main',
            'test_trajectory_interpolator = ur_control.examples.test_trajectory_interpolator:main',
            'test_analytic_ik = ur_control.examples.test_analytic_ik:main',
            'dp_control_ros2 = ur_control.examples.dp_control_ros2:main',
            'dp_control_rtde = ur_control.examples.dp_control_rtde:main',
            'analyze_trajectory = ur_control.examples.analyze_trajectory:main',
        ],
    },
)
