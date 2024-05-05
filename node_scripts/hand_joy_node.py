#!/usr/bin/env python

import numpy as np

import rospy
from sensor_msgs.msg import Joy
from hand_object_detection_ros.msg import HandDetectionArray
from speech_recognition_msgs.msg import SpeechRecognitionCandidates
from std_msgs.msg import Float32, String

from scipy.signal import savgol_filter

class HandJoyNode(object):
    def __init__(self):
        super(HandJoyNode, self).__init__()
        self.button = 0
        self.pub_left_joy = rospy.Publisher("~left_hand/joy", Joy, queue_size=1)
        self.pub_right_joy = rospy.Publisher("~right_hand/joy", Joy, queue_size=1)
        self.hand_dist_threshold = rospy.get_param("~hand_dist_threshold", 0.09)
        self.window_size = rospy.get_param("~window_size", 5)
        self.poly_order = rospy.get_param("~poly_order", 3)
        self.dist_queue = []
        self.debug = rospy.get_param("~debug", False)
        if self.debug:
            self.pub_dist = rospy.Publisher(
                "~hand_distance", Float32, queue_size=1
            ) # for rviz visualization
            self.pub_status = rospy.Publisher(
                "~status", String, queue_size=1
            )

        self.sub_hand_detections = rospy.Subscriber(
            "~hand_detections",
            HandDetectionArray,
            self.hand_detection_callback,
            queue_size=1,
        )

        self.sub_speech_to_text = rospy.Subscriber(
            "~speech_to_text",
            SpeechRecognitionCandidates,
            self.speech_recognition_callback,
        )

    def speech_recognition_callback(self, msg):
        flag = msg.transcript[0]
        if flag == "OK":
            self.button = 1
            rospy.sleep(0.5)
        self.button = 0  # reset button after 0.5 sec

    def apply_savgol_filter(self, data):
        if len(data) < self.window_size:
            return data[-1]
        else:
            filtered_data = savgol_filter(
                np.array(data), self.window_size, self.poly_order
            )
            return filtered_data[-1]

    def hand_detection_callback(self, hand_detections):
        for hand_detection in hand_detections.detections:
            joy = Joy(header=hand_detections.header)
            joy.axes = [0, 0, 0, 0, 0, 0]  # does not need

            # get thumb and index finger coordinates to calculate distance
            thumb_coords = [
                hand_detection.skeleton.bones[3].end_point.x,
                hand_detection.skeleton.bones[3].end_point.y,
                hand_detection.skeleton.bones[3].end_point.z,
            ]
            index_coords = [
                hand_detection.skeleton.bones[7].end_point.x,
                hand_detection.skeleton.bones[7].end_point.y,
                hand_detection.skeleton.bones[7].end_point.z,
            ]
            dist = np.linalg.norm(np.array(thumb_coords) - np.array(index_coords))

            # apply savgol filter to smooth the distance
            self.dist_queue.append(dist)
            if len(self.dist_queue) > self.window_size:
                self.dist_queue.pop(0)
            dist = self.apply_savgol_filter(self.dist_queue)

            # if distance is less than threshold, set button to 1
            if dist < self.hand_dist_threshold:
                joy.buttons = [1]  # grasp
            else:
                joy.buttons = [0]  # release

            # if flag is detected, set button to 1
            joy.buttons.append(self.button)

            if hand_detection.hand == "left_hand":
                self.pub_left_joy.publish(joy)
            elif hand_detection.hand == "right_hand":
                self.pub_right_joy.publish(joy)
                if self.debug:
                    self.pub_dist.publish(Float32(data=dist))
                    if dist < self.hand_dist_threshold:
                        self.pub_status.publish(String(data="Grasping"))
                    else:
                        self.pub_status.publish(String(data="Releasing"))


if __name__ == "__main__":
    rospy.init_node("hand_joy_node")
    policy_running_node = HandJoyNode()
    rospy.spin()
