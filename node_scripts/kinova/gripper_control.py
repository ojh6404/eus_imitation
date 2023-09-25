#!/usr/bin/env python3

import rospy
import time
from kortex_driver.srv import SendGripperCommand, SendGripperCommandRequest
from kortex_driver.msg import GripperCommand, GripperMode, GripperRequest, GripperMode, Gripper, Finger
from sensor_msgs.msg import Joy, JointState

class KinovaGripperController(object):
    def __init__(self):

        self.robot_name = "arm_gen3"
        self.gripper_command_service = rospy.ServiceProxy('/arm_gen3/base/send_gripper_command', SendGripperCommand)
        self.gripper_state_subscriber = rospy.Subscriber('/arm_gen3/joint_states', JointState, self.gripper_state_callback)
        rospy.wait_for_service('/arm_gen3/base/send_gripper_command')

        self.gripper_state = None
        self.joy_sub = rospy.Subscriber('/spacenav/joy', Joy, self.joy_callback, queue_size=1)
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
            self.gripper_command_service(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SendGripperCommand")
            return False
        else:
            return True


    def joy_callback(self, msg):
        self.button_cmd = msg.buttons[0:2]
        if self.button_cmd[0] == 1:
            self.gripper_command(1.0)
        else:
            self.gripper_command(0.0)


    def gripper_state_callback(self, msg):
        self.gripper_state = msg

if __name__ == '__main__':
    rospy.init_node("kinova_gripper_controller")
    rospy.loginfo("Start kinova gripper control...")
    rosbag_manager = KinovaGripperController()
    rospy.spin()
