#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import rospy
import rospkg

import os
import yaml
import tf
from speech_recognition_msgs.msg import SpeechRecognitionCandidates
from sound_play.libsoundplay import SoundClient


class CalibrationNode(object):
    def __init__(self):
        self.tf_listener = tf.TransformListener()
        self.robot_gripper_to_calib_cube = None
        self.hand_to_calib_cube = None
        self.sound_client = SoundClient()
        self.volume = 0.5
        self.state = "wait_robot_gripper"

        self.robot_gripper_frame = rospy.get_param(
            "~robot_gripper_frame", "r_gripper_tool_frame"
        )
        self.hand_frame = rospy.get_param("~hand_frame", "right_hand")
        self.calib_cube_frame = rospy.get_param("~calib_cube_frame", "calib_cube_frame")

        self.sub_speech_to_text = rospy.Subscriber(
            "~speech_to_text",
            SpeechRecognitionCandidates,
            self.speech_recognition_callback,
        )
        self.sound_client.say(
            "Please set calibration cube to the robot gripper", volume=self.volume
        )

    def speech_recognition_callback(self, msg):
        if msg.transcript[0] == "OK":
            if self.state == "wait_robot_gripper":
                self.robot_gripper_to_calib_cube = self.get_transform(
                    self.robot_gripper_frame, self.calib_cube_frame
                )
                if self.robot_gripper_to_calib_cube is None:
                    error_msg = "Failed to get transform from {} to {}".format(
                        self.robot_gripper_frame, self.calib_cube_frame
                    )
                    rospy.logerr(error_msg)
                    self.sound_client.say(error_msg, volume=self.volume)
                    self.sound_client.say(
                        "Please set calibration cube to the robot gripper",
                        volume=self.volume,
                    )
                self.sound_client.say(
                    "Please set calibration cube to the hand",
                    volume=self.volume,
                )
                rospy.loginfo(
                    "{} to {} : {}".format(
                        self.robot_gripper_frame,
                        self.calib_cube_frame,
                        self.robot_gripper_to_calib_cube,
                    )
                )
                rospy.sleep(3.0)
                self.state = "wait_hand"
            elif self.state == "wait_hand":
                self.hand_to_calib_cube = self.get_transform(
                    self.hand_frame, self.calib_cube_frame
                )
                if self.hand_to_calib_cube is None:
                    error_msg = "Failed to get transform from {} to {}".format(
                        self.hand_frame, self.calib_cube_frame
                    )
                    rospy.logerr(error_msg)
                    self.sound_client.say(error_msg, volume=self.volume)
                    self.sound_client.say(
                        "Please set calibration cube to the hand", volume=self.volume
                    )
                rospy.loginfo(
                    "{} to {} : {}".format(
                        self.hand_frame, self.calib_cube_frame, self.hand_to_calib_cube
                    )
                )
                self.hand_to_robot_gripper = (
                    self.hand_to_calib_cube - self.robot_gripper_to_calib_cube
                )  # TODO only translation, not rotation
                rospy.loginfo(
                    "offset from {} to {} : {}".format(
                        self.hand_frame,
                        self.robot_gripper_frame,
                        self.hand_to_robot_gripper,
                    )
                )
                rospy.sleep(3.0)
                package_path = rospkg.RosPack().get_path("eus_imitation")
                config_yaml = os.path.join(
                    package_path, "config", "hand_calibration.yaml"
                )
                with open(config_yaml, "w") as f:
                    yaml.dump(
                        {
                            self.hand_frame
                            + "_offset": self.hand_to_robot_gripper.tolist()
                        },
                        f,
                    )
                self.sound_client.say("Calibration done", volume=self.volume)
                rospy.signal_shutdown("Calibration done")

    def get_transform(self, frame1, frame2):
        try:
            # only use trans
            trans, _ = self.tf_listener.lookupTransform(frame1, frame2, rospy.Time(0))
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            rospy.logerr(
                "Failed to lookup transform from {} to {}".format(frame1, frame2)
            )
            return None
        return np.array(trans)


if __name__ == "__main__":
    rospy.init_node("calibration_node")
    node = CalibrationNode()
    rospy.spin()
