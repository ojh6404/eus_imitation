#!/usr/bin/env python3

import rospy
import time
from kortex_driver.srv import SendGripperCommand, SendGripperCommandRequest
from kortex_driver.msg import GripperCommand, GripperMode, GripperRequest, GripperMode, Gripper, Finger
from sensor_msgs.msg import Joy, JointState

class KinovaGripperController(object):
    def __init__(self):

        self.robot_name = "arm_gen3"
        # self.gripper_command_service = rospy.ServiceProxy('/arm_gen3/base/send_gripper_command', SendGripperCommand)
        self.gripper_command_service = rospy.ServiceProxy('/arm_gen3/base/send_gripper_command', SendGripperCommand)
        self.gripper_state_subscriber = rospy.Subscriber('/arm_gen3/joint_states', JointState, self.gripper_state_callback)
        rospy.wait_for_service('/arm_gen3/base/send_gripper_command')

        self.gripper_state = None
        self.gripper_open = 2.3
        self.gripper_closed = 3.7
        self.prev_gripper_pos = 0
        self.update_rate = 20

        self.joy_sub = rospy.Subscriber('/spacenav/joy', Joy, self.joy_callback, queue_size=1)
        self.linear_cmd = [0, 0, 0]
        self.angular_cmd = [0, 0, 0]
        self.button_cmd = [0, 0]



    def gripper_command(self, value):
        # Initialize the request
        # Close the gripper
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION

        rospy.loginfo("Sending the gripper command...")

        # Call the service
        try:
            # self.send_gripper_command(req)
            self.gripper_command_service(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SendGripperCommand")
            return False
        else:
            # time.sleep(0.5)
            return True


    def joy_callback(self, msg):
        self.linear_cmd = msg.axes[0:3]
        self.angular_cmd = msg.axes[3:6]
        self.button_cmd = msg.buttons[0:2]

        print(msg.buttons[0])

        if self.button_cmd[0] == 1:
            print("Closing gripper")
            self.gripper_command(1.0)
        else:
            self.gripper_command(0.0)


    def gripper_state_callback(self, msg):
        self.gripper_state = msg


    # def control_loop(self):
    #     rate = rospy.Rate(self.update_rate)  # 10 Hz control loop

    #     while not rospy.is_shutdown():
    #         if self.gripper_state is not None:
    #             # Control gripper based on the received gripper state
    #             target_position = self.gripper_state.position[6]
    #             speed = self.gripper_state.velocity[6]
    #             effort = self.gripper_state.effort[6]

    #             max_velocity = 0.03
    #             gain = 1.0


    #             print(self.button_cmd)

    #             rospy.loginfo("Gripper State: position: {}, speed: {}, effort: {}".format(target_position, speed, effort))


    #             for i in range(100):

    #                 # proprotional control
    #                 position_error = i - self.prev_gripper_pos
    #                 velocity_command = gain * position_error

    #                 # Clamp the vel command
    #                 velocity_command = max(min(velocity_command, max_velocity), -max_velocity)

    #                 target_position = self.prev_gripper_pos + velocity_command

    #                 gripper_command = GripperCommand()
    #                 finger = Finger()
    #                 finger.finger_identifier = 0
    #                 finger.value = i/100
    #                 gripper_command.gripper.finger.append(finger)
    #                 gripper_command.mode = GripperMode.GRIPPER_POSITION
    #                 self.prev_gripper_pos = target_position

    #             try:
    #                 self.gripper_command_service(gripper_command)
    #             except rospy.ServiceException as e:
    #                 rospy.logerr("Service call failed: {}".format(e))

    #         rate.sleep()

if __name__ == '__main__':
    rospy.init_node("kinova_gripper_controller")
    rospy.loginfo("Start kinova gripper control...")
    rosbag_manager = KinovaGripperController()
    rospy.spin()
