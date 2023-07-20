#!/usr/bin/env python
import os
import rosbag
import cv2
from cv_bridge import CvBridge

def get_full_path_of_rosbag_file(directory):
    rosbag_file_list = []
    for file in os.listdir(directory):
        if file.endswith(".bag"):
            rosbag_file_list.append(os.path.join(directory, file))
    return rosbag_file_list

def get_file_name_from_full_path(full_path):
    return os.path.basename(full_path)

def rosbag2gray(input):
    output_bag = os.path.join("./output",get_file_name_from_full_path(input))
    with rosbag.Bag(output_bag, 'w') as outbag:
        for topic, msg, t in rosbag.Bag(input).read_messages():
            if topic == "/kinect_head/rgb/image_rect_color/compressed":
                img_rgb = CvBridge().compressed_imgmsg_to_cv2(msg)
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                gray_msg = CvBridge().cv2_to_compressed_imgmsg(img_gray)
                gray_msg.header = msg.header
                gray_msg.format = "mono8; jpeg compressed "
                gray_topic = "/kinect_head/rgb/image_rect_mono/compressed"
                outbag.write(gray_topic, gray_msg, gray_msg.header.stamp if gray_msg._has_header else t)
            else:
                outbag.write(topic, msg, msg.header.stamp if msg._has_header else t)



if __name__=="__main__":

    rosbag_file_path_list = get_full_path_of_rosbag_file("/home/oh/.mohou/test_rosbag")
    for file_path in rosbag_file_path_list:
        rosbag2gray(file_path)
