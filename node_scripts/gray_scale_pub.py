#!/usr/bin/env python3

import sys
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

"""
Node to transform an input Image topic into
an output grayscale Image topic.
Author: Sammy Pfeiffer <Sammy.Pfeiffer at student.uts.edu.au>
"""


class GrayScaler(object):
    def __init__(self, name_topic_in, name_topic_out):
        self.cv_bridge = CvBridge()
        rospy.loginfo(
            "Converting Images from topic "
            + name_topic_in
            + " to grayscale, output topic: "
            + name_topic_out
        )
        self.pub = rospy.Publisher(name_topic_out, CompressedImage, queue_size=10)
        self.sub = rospy.Subscriber(
            name_topic_in, CompressedImage, self.image_cb, queue_size=10
        )

    def image_cb(self, img_msg):
        # Transform to cv2/numpy image
        img_in_cv2 = self.cv_bridge.compressed_imgmsg_to_cv2(
            img_msg, desired_encoding="passthrough"
        )
        # Transform to grayscale,
        # available encodings: http://docs.ros.org/jade/api/sensor_msgs/html/image__encodings_8h_source.html
        # if "rgb" in img_msg.encoding:
        #     gray_img = cv2.cvtColor(img_in_cv2, cv2.COLOR_RGB2GRAY)
        # elif "bgr" in img_msg.encoding:
        # gray_img = cv2.cvtColor(img_in_cv2, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.cvtColor(img_in_cv2, cv2.COLOR_BGR2GRAY)
        # Transform back to Image message
        gray_img_msg = self.cv_bridge.cv2_to_compressed_imgmsg(gray_img)
        self.pub.publish(gray_img_msg)


if __name__ == "__main__":
    rospy.init_node("image_to_grayscale", anonymous=True)
    input_topic_img = "/kinect_head/rgb/image_rect_color/compressed"
    output_topic_img = "/kinect_head/rgb/image_rect_color/compressed/gray"

    gs = GrayScaler(input_topic_img, output_topic_img)
    rospy.spin()
