#!/usr/bin/env python

# test for rospy init_node

import rospy
import sys
# python version
print(sys.version)

print("test_node is running")
rospy.init_node('test_node')
print("test_node is initialized")
rospy.spin()
