#!/usr/bin/env

import sys
import rospy
import signal
import numpy as np
from math import pi
import scipy.io as sio
from numpy.linalg import *
import matplotlib.pyplot as plt
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

import baxter_interface as baxter
from baxter_pykdl import baxter_kinematics
from baxter_core_msgs.msg import EndpointState


class BMI(object):
    """
    Moves Baxter's arm along X-axis by commanding end-effector velocity, imitating monkey's reach-to-grasp task
    """

    def __init__(self):
        self._left_arm = baxter.limb.Limb("left")
        self._right_arm = baxter.limb.Limb("right")  # Working hand
        self.leftL = baxter_kinematics('left')
        self.rightL = baxter_kinematics('right')
        self._left_joint_names = self._left_arm.joint_names()
        self._right_joint_names = self._right_arm.joint_names()

        # Retrieve BMI data
        self.bmi_Vx_raw = self.open_dataset()
        self._range = self.bmi_Vx_raw.shape[0]
        self.bmi_Vx = self.bmi_Vx_raw / 60  # Scale down linear twist

        # Joint parameters from sensors
        self.joint_position = np.zeros(7)
        self.joint_velocity = np.zeros(7)
        self.joint_torque = np.zeros(7)
        # End-effector parameters from sensors
        self.baxterPose = np.zeros(self._range)
        self.baxterTwist = np.zeros(self._range)

        # Subscribe to ROS JointState topic
        rospy.Subscriber("/robot/joint_states", JointState, self.callback_joint)
        # Subscribe to Baxter's EndpointState topic
        rospy.Subscriber("/robot/limb/right/endpoint_state", EndpointState, self.callback_endpoint)

        # Set the frequency at which the joint commands are updated
        self._pub_rate = rospy.Publisher('/robot/joint_state_publish_rate', Float64, queue_size=10)
        self._rate = 100  # Controller frequency = 100Hz.
        self._pub_rate.publish(self._rate)
        # Enable robot
        self._rs = baxter.RobotEnable()
        self._init_state = self._rs.state().enabled
        print("\nEnabling robot... ")
        self._rs.enable()

    def open_dataset(self):
        data = np.array(
            sio.loadmat('./i140703-001_alg_MAD8_MAF2_NEO_RMSPow2_fs10k')['targetDataInterpolated'][0, 3000:4380])
        return data

    def on_switch(self):  # Home position
        print("\nGetting ready to move to target object...")
        P = [-0.14013857, 0.12821865, 0.19197406, 0.91986668, -1.08875195, -1.11735184, 0.41248847]
        self._right_arm.move_to_joint_positions(dict(zip(self._right_joint_names, P)))

    def on_switch_test(self):  # Home position
        print("\nGetting ready to move to target object...")
        P = [0.623447850102183, 0.051407264172355305, -0.2292517537625658, 1.6905862920195016, 1.3801621211430115,
             -1.3724074763182923, 0.20965579991749994]
        self._left_arm.move_to_joint_positions(dict(zip(self._left_joint_names, P)))

    def callback_joint(self, msg):  # Joint data from sensors
        for i, joint in enumerate(self._right_joint_names):
            if joint in msg.name:
                self.joint_position[i] = msg.position[msg.name.index(joint)]
                self.joint_velocity[i] = msg.velocity[msg.name.index(joint)]
                self.joint_torque[i] = msg.effort[msg.name.index(joint)]
            else:
                return

    def callback_endpoint(self, msg):  # End-effector data from sensors
        self.baxterPose = msg.pose
        self.baxterTwist = msg.twist

    def bmi_control(self):
        """
        Controls Baxter's right arm using BMI dataset that is translated into joint velocities in "raw" position control mode
        """

        self.on_switch()
        rate = rospy.Rate(self._rate)

        # Joint data
        q = np.zeros((7, self._range))
        dq = np.zeros((7, self._range))
        tq = np.zeros((7, self._range))
        # End-effector data
        pose_x = np.zeros((1, self._range))
        pose_y = np.zeros((1, self._range))
        pose_z = np.zeros((1, self._range))
        orientation_x = np.zeros((1, self._range))
        orientation_y = np.zeros((1, self._range))
        orientation_z = np.zeros((1, self._range))
        orientation_w = np.zeros((1, self._range))

        q_des = np.zeros((7, self._range))
        dq_des = np.zeros((7, self._range))
        q_des[:, 0] = np.array([-0.14013857, 0.12821865, 0.19197406, 0.91986668, -1.08875195, -1.11735184, 0.41248847])

        for i in range(self._range):
            self._pub_rate.publish(self._rate)

            # Collect joint data
            q[:, i] = self.joint_position
            dq[:, i] = self.joint_velocity
            tq[:, i] = self.joint_torque
            # Collect end-effector data
            pose_x[:, i] = self.baxterPose.position.x
            pose_y[:, i] = self.baxterPose.position.y
            pose_z[:, i] = self.baxterPose.position.z
            orientation_x[:, i] = self.baxterPose.orientation.x
            orientation_y[:, i] = self.baxterPose.orientation.y
            orientation_z[:, i] = self.baxterPose.orientation.z
            orientation_w[:, i] = self.baxterPose.orientation.w

            if i > 0:
                q_des[:, i] = q_des[:, i - 1] + 0.01 * dq_des[:, i - 1]
            J = np.array(self.rightL.jacobian())
            JT = np.linalg.pinv(J)
            X = np.array([self.bmi_Vx[i], 0, 0, 0, 0, 0])
            dq_des[:, i] = np.dot(JT, X)

            self._right_arm.set_joint_positions(dict(zip(self._right_joint_names, q_des[:, i])))
            rate.sleep()  # Delay = 10ms. before receiving new data

        self.on_switch()

        plt.figure()
        joint_num = len(self._left_joint_names)
        for joint in range(joint_num):
            plt.subplot(joint_num, 3, joint * 3 + 1)
            plt.plot(q_des[joint, :] * 180 / pi, color='green')  # Desired
            plt.plot(q[joint, :] * 180 / pi, color='black')  # Actual
            plt.title('Joint ' + str(joint + 1) + ' Position')

            plt.subplot(joint_num, 3, joint * 3 + 2)
            plt.plot(dq_des[joint, :] * 180 / pi, color='green')  # Desired
            plt.plot(dq[joint, :] * 180 / pi, color='black')  # Actual
            plt.title('Joint ' + str(joint + 1) + ' Velocity')

            plt.subplot(joint_num, 3, joint * 3 + 3)
            plt.plot(tq[joint, :], color='black')
            plt.title('Joint ' + str(joint + 1) + ' Torque')
        plt.tight_layout(pad=0.001)
        plt.show()

        plt.figure()
        plt.subplot(211)
        plt.plot(pose_x[0, :], color='black')
        plt.plot(pose_y[0, :], color='blue')
        plt.plot(pose_z[0, :], color='yellow')
        plt.title('Linear displacement')

        plt.subplot(212)
        plt.plot(orientation_x[0, :], color='black')
        plt.plot(orientation_y[0, :], color='blue')
        plt.plot(orientation_z[0, :], color='yellow')
        plt.plot(orientation_w[0, :], color='red')
        plt.title('Angular displacement')

        plt.tight_layout(pad=0.001)
        plt.show()

    def bmi_control_test(self):
        """
        Controls Baxter's left arm using BMI dataset that is translated into joint velocities in velocity control mode
        """

        self.on_switch_test()
        rate = rospy.Rate(self._rate)

        # (!) Command rate > command timeout
        self._left_arm.set_command_timeout(0.005)

        # Joint data
        q = np.zeros((7, self._range))
        dq = np.zeros((7, self._range))
        tq = np.zeros((7, self._range))
        dq_bmi = np.zeros((7, self._range))
        # End-effector data
        pose_x = np.zeros((1, self._range))
        pose_y = np.zeros((1, self._range))
        pose_z = np.zeros((1, self._range))
        orientation_x = np.zeros((1, self._range))
        orientation_y = np.zeros((1, self._range))
        orientation_z = np.zeros((1, self._range))
        orientation_w = np.zeros((1, self._range))

        print("\nBMI is controlling the arm...")

        for i in range(self._range):
            self._pub_rate.publish(self._rate)

            # Collect joint data
            q[:, i] = self.joint_position
            dq[:, i] = self.joint_velocity
            tq[:, i] = self.joint_torque
            # Collect end-effector data
            pose_x[:, i] = self.baxterPose.position.x
            pose_y[:, i] = self.baxterPose.position.y
            pose_z[:, i] = self.baxterPose.position.z
            orientation_x[:, i] = self.baxterPose.orientation.x
            orientation_y[:, i] = self.baxterPose.orientation.y
            orientation_z[:, i] = self.baxterPose.orientation.z
            orientation_w[:, i] = self.baxterPose.orientation.w

            J = np.array(self.leftL.jacobian())
            JT = np.linalg.pinv(J)
            X = np.array([self.bmi_Vx[i], 0, 0, 0, 0, 0])
            dq_bmi_cmd = np.dot(JT, X)
            dq_bmi[:, i] = dq_bmi_cmd  # Validation parameter

            self._left_arm.set_joint_velocities(
                dict(zip(self._left_joint_names, dq_bmi_cmd)))  # Send commands to the velocity controller

            rate.sleep()  # Delay = 10ms. before receiving new data

        self.on_switch()
        self._left_arm.exit_control_mode()

        plt.figure()
        joint_num = len(self._right_joint_names)
        for joint in range(joint_num):
            plt.subplot(joint_num, 3, joint * 3 + 1)
            plt.plot(q[joint, :], color='black')
            plt.title('Joint ' + str(joint + 1) + ' Position')

            plt.subplot(joint_num, 3, joint * 3 + 2)
            plt.plot(dq_bmi[joint, :], color='green')  # Desired
            plt.plot(dq[joint, :], color='black')  # Actual
            plt.title('Joint ' + str(joint + 1) + ' Velocity')

            plt.subplot(joint_num, 3, joint * 3 + 3)
            plt.plot(tq[joint, :], color='black')
            plt.title('Joint ' + str(joint + 1) + ' Torque')
        plt.tight_layout(pad=0.001)
        plt.show()

        plt.figure()
        plt.subplot(211)
        plt.plot(pose_x[0, :], color='black')
        plt.plot(pose_y[0, :], color='blue')
        plt.plot(pose_z[0, :], color='yellow')
        plt.title('Linear displacement')

        plt.subplot(212)
        plt.plot(orientation_x[0, :], color='black')
        plt.plot(orientation_y[0, :], color='blue')
        plt.plot(orientation_z[0, :], color='yellow')
        plt.plot(orientation_w[0, :], color='red')
        plt.title('Angular displacement')

        plt.tight_layout(pad=0.001)
        plt.show()


def main():
    signal.signal(signal.SIGINT, shutdown)
    print("Initializing node... ")
    rospy.init_node("BMI_receiver")

    bmi_baxter = BMI()
    bmi_baxter.bmi_control()
    # bmi_baxter.bmi_control_test()
    print("Done.")


def shutdown(sig):
    print("\nExiting...")
    sys.exit(0)


if __name__ == '__main__':
    main()

