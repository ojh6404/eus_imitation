#!/usr/bin/env python
# -*- coding: utf-8 -*-


# just publish python version to check if it is working
import sys
print(sys.version)

import rospy
from std_msgs.msg import String

class TestNode:
    def __init__(self,):
        self.pub = rospy.Publisher("test", String, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(1), self.timer_callback)


    def timer_callback(self, event):
        print("Timer callback")
        self.pub.publish(f"python verion {sys.version}")


if __name__ == "__main__":
    rospy.init_node("test_node")
    rospy.loginfo("TestNode start")
    policy_running_node = TestNode()
    rospy.spin()
