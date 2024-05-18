#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
from tf.transformations import (
    quaternion_from_euler,
    translation_matrix,
    quaternion_matrix,
    concatenate_matrices,
    translation_from_matrix,
    quaternion_from_matrix,
)


class HandFrameCalibrationNode(object):
    def __init__(self):
        super(HandFrameCalibrationNode, self).__init__()
        self.rate = rospy.get_param("~rate", 10)
        self.hand_frame = rospy.get_param("~hand_frame", "right_hand")
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")
        self.target_frame = rospy.get_param(
            "~target_frame", "target_right_hand"
        )
        self.offset = rospy.get_param("~offset", "right_hand_offset")
        self.offset_trans = rospy.get_param(self.offset, [0.0, 0.0, 0.0])
        self.offset_rot = rospy.get_param("~offset_rot", [0.0, 0.0, 0.0])

        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.rate), self.broadcast_target_frame
        )

    def broadcast_target_frame(self, event):
        try:
            # Listen to the transformation from frame1 to its parent frame
            (trans, rot) = self.tf_listener.lookupTransform(
                self.base_frame, self.hand_frame, rospy.Time(0)
            )

            # Define the offset as a translation vector and a rotation quaternion
            # offset_translation = (0.1, 0.0, 0.0)  # 10 cm along the x-axis
            offset_rotation = quaternion_from_euler(
                *self.offset_rot
            )  # Assuming no additional rotation

            # Create transformation matrices from the translations and rotations
            hand_frame_mat = concatenate_matrices(
                translation_matrix(trans), quaternion_matrix(rot)
            )
            offset_mat = concatenate_matrices(
                translation_matrix(self.offset_trans),
                quaternion_matrix(offset_rotation),
            )

            # Calculate the new transformation matrix for frame2
            target_frame_mat = concatenate_matrices(hand_frame_mat, offset_mat)

            # Extract the translation and rotation from the transformation matrix
            target_trans = translation_from_matrix(target_frame_mat)
            target_rot = quaternion_from_matrix(target_frame_mat)

            # Broadcast the new transformation
            self.tf_broadcaster.sendTransform(
                target_trans,
                target_rot,
                rospy.Time.now(),
                self.target_frame,
                self.base_frame,
            )

        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            rospy.loginfo("TF lookup exception caught")


if __name__ == "__main__":
    rospy.init_node("offset_calibration")
    node = HandFrameCalibrationNode()
    rospy.loginfo("HandFrameCalibrationNode initialized")
    rospy.spin()
