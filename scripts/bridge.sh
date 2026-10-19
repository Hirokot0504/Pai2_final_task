#!/bin/bash
# Gazebo と ROS 2 のトピックを接続する
#   /cmd_vel : ROS 2 → Gazebo(速度指令)
#   /odom    : Gazebo → ROS 2(ロボットの位置)
#   /camera  : Gazebo → ROS 2(カメラ画像)
source /opt/ros/jazzy/setup.bash
ros2 run ros_gz_bridge parameter_bridge \
  "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist" \
  "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry" \
  "/camera@sensor_msgs/msg/Image[gz.msgs.Image"
