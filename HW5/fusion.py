#!/usr/bin/env python3

import rospy
import os
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped as Pose
from nav_msgs.msg import Odometry
from math import cos, sin, atan2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '/ros_ws/src/fusion/scripts')
from EKF import ExtendedKalmanFilter

class Fusion:
    def __init__(self):
        rospy.Subscriber('/gps', Pose, self.gpsCallback)
        rospy.Subscriber('/radar_odometry', Odometry, self.odometryCallback)
        rospy.Subscriber('/gt_odom', Odometry, self.gtCallback)
        rospy.on_shutdown(self.shutdown)
        self.posePub = rospy.Publisher('/pred', Odometry, queue_size=10)
        self.EKF = None

        self.gt_list  = [[], []]
        self.est_list = [[], []]
        self.initial  = False

        self.last_odom = None  # [x, y, yaw]

    def shutdown(self):
        print("shutting down fusion.py")
        self.plot_path()
        print("plot saved")

    # ------------------------------------------------------------------
    def predictPublish(self):
        if self.EKF is None:
            return

        predPose = Odometry()
        predPose.header.stamp    = rospy.Time.now()
        predPose.header.frame_id = 'origin'

        predPose.pose.pose.position.x = self.EKF.pose[0]
        predPose.pose.pose.position.y = self.EKF.pose[1]
        predPose.pose.pose.position.z = 0.0

        quaternion = quaternion_from_euler(0, 0, self.EKF.pose[2])
        predPose.pose.pose.orientation.x = quaternion[0]
        predPose.pose.pose.orientation.y = quaternion[1]
        predPose.pose.pose.orientation.z = quaternion[2]
        predPose.pose.pose.orientation.w = quaternion[3]

        # Map 3x3 EKF covariance [x, y, yaw] → 6x6 ROS covariance [x,y,z,r,p,yaw]
        # x=0, y=1, yaw=5 in ROS convention
        S   = self.EKF.S
        cov = [0.0] * 36

        # row/col mapping: EKF[0,1,2] → ROS[0,1,5]
        ros_idx = [0, 1, 5]
        for i in range(3):
            for j in range(3):
                cov[ros_idx[i] * 6 + ros_idx[j]] = S[i, j]

        predPose.pose.covariance = cov
        self.posePub.publish(predPose)

    # ------------------------------------------------------------------
    def odometryCallback(self, data):
        odom_x = data.pose.pose.position.x
        odom_y = data.pose.pose.position.y
        odom_q = [
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w
        ]
        _, _, odom_yaw = euler_from_quaternion(odom_q)
        odom_cov = np.array(data.pose.covariance).reshape(6, 6)

        if not self.initial:
            self.EKF       = ExtendedKalmanFilter(odom_x, odom_y, odom_yaw)
            self.initial   = True
            self.last_odom = [odom_x, odom_y, odom_yaw]
            self.predictPublish()
            return

        # motion in local frame
        if self.last_odom is not None:
            diff_x   = odom_x   - self.last_odom[0]
            diff_y   = odom_y   - self.last_odom[1]
            diff_yaw = odom_yaw - self.last_odom[2]
            diff_yaw = atan2(sin(diff_yaw), cos(diff_yaw))

            last_yaw = self.last_odom[2]
            u_x   =  diff_x * cos(last_yaw) + diff_y * sin(last_yaw)
            u_y   = -diff_x * sin(last_yaw) + diff_y * cos(last_yaw)
            u_yaw =  diff_yaw

            control = np.array([u_x, u_y, u_yaw])

            # extract [x, y, yaw]
            idx = [0, 1, 5]
            R_full = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    R_full[i, j] = odom_cov[idx[i], idx[j]]

            R_full += np.eye(3) * 1e-6
            self.EKF.R = R_full

            self.EKF.predict(u=control)

        self.last_odom = [odom_x, odom_y, odom_yaw]
        self.predictPublish()

    # ------------------------------------------------------------------
    def gpsCallback(self, data):
        gps_x = data.pose.pose.position.x
        gps_y = data.pose.pose.position.y
        gps_cov = np.array(data.pose.covariance).reshape(6, 6)

        measurement = np.array([gps_x, gps_y])

        if not self.initial:
            self.EKF     = ExtendedKalmanFilter(gps_x, gps_y, 0.0)
            self.initial = True
        else:
            #  GPS covariance: [x, y] 
            Q = gps_cov[:2, :2].copy()
            Q += np.eye(2) * 1e-6   
            self.EKF.Q = Q

            self.EKF.update(z=measurement)

        self.predictPublish()

    # ------------------------------------------------------------------
    def gtCallback(self, data):
        self.gt_list[0].append(data.pose.pose.position.x)
        self.gt_list[1].append(data.pose.pose.position.y)

        if self.EKF is not None:
            self.est_list[0].append(self.EKF.pose[0])
            self.est_list[1].append(self.EKF.pose[1])

    # ------------------------------------------------------------------
    def plot_path(self):
        plt.figure(figsize=(10, 8))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.plot(self.gt_list[0], self.gt_list[1],
                 alpha=0.25, linewidth=8, label='Groundtruth path')
        plt.plot(self.est_list[0], self.est_list[1],
                 alpha=0.5,  linewidth=3, label='Estimation path')
        plt.title("EKF Fusion Result")
        plt.legend()

        save_dir = "/ros_ws/src/fusion/result"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "result.png"))


if __name__ == '__main__':
    rospy.init_node('ekf_fusion', anonymous=True)
    fusion = Fusion()
    rospy.spin()
