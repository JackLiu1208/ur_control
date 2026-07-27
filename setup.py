from setuptools import find_packages, setup

package_name = 'ur_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
        ],
    },
)
