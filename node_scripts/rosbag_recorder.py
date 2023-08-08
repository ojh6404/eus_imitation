#!/usr/bin/env python3

import subprocess
import threading
import time
import signal
import os

from omegaconf import OmegaConf
import rospy
import rospkg
from std_srvs.srv import Trigger, TriggerResponse
from sound_play.libsoundplay import SoundClient


import imitator.utils.ros_utils as RosUtils


class RosbagRecorderNode(object):
    def __init__(self):
        package_path = rospkg.RosPack().get_path("eus_imitation")
        try:
            self.config = OmegaConf.load("{}/config/config.yaml".format(package_path))
        except:
            raise FileNotFoundError("config.yaml not found")

        self.record_dir = os.path.join(
            package_path, "data", time.strftime("%Y%m%d"), "rosbags"
        )

        try:
            if not os.path.exists(self.record_dir):
                os.makedirs(self.record_dir)
        except OSError:
            print("Error: Failed to create the directory.")

        # self.record_topics = self.config.rosbag.record_topics

        self.obs_cfg = self.config.obs

        self.record_topics = []

        for key in self.obs_cfg.keys():
            self.record_topics.append(self.obs_cfg[key].topic_name)

        self.action_topic = self.config.actions.topic_name
        self.record_topics.append(self.action_topic)
        rospy.loginfo("Recording topics : {}".format(self.record_topics))

        self.is_record = False
        self.switch_record_state_service = rospy.Service(
            "/eus_imitation/rosbag_record_trigger", Trigger, self.switch_record_state_cb
        )
        self.remove_rosbag_service = rospy.Service(
            "/eus_imitation/rosbag_remove_trigger", Trigger, self.remove_rosbag_cb
        )
        self.sound_client = SoundClient()

    def switch_record_state_cb(self, req):
        if self.is_record:
            self.stop_record()
        else:
            self.start_record()
        message = "start" if self.is_record else "stop"
        return TriggerResponse(success=True, message=message)

    def remove_rosbag_cb(self, req):
        self.check_rosbag_files()
        if self.file_cnt == 0:
            rospy.loginfo("No rosbag file")
            self.sound_client.say("No rosbag file")
        else:
            remove_filepath = os.path.join(self.record_dir, self.rosbag_files[-1])
            if os.path.isfile(remove_filepath):
                os.remove(remove_filepath)
                self.rosbag_files.pop()
                rospy.loginfo("Remove rosbag : {}".format(self.file_cnt - 1))
                self.sound_client.say("Delete rosbag {}".format(self.file_cnt - 1))
        return TriggerResponse(success=True, message="remove rosbag")

    def create_cmd_rosbag(self, rosbag_filepath):
        cmd_rosbag = ["rosbag", "record"]
        record_topics = list(set(self.record_topics + ["/tf"]))
        # record_topics = self.record_topics
        cmd_rosbag.extend(record_topics)
        cmd_rosbag.extend(["--output-name", rosbag_filepath])
        return cmd_rosbag

    def check_rosbag_files(self):
        self.rosbag_files = RosUtils.get_rosbag_files(self.record_dir)
        self.file_cnt = len(self.rosbag_files)

    def start_record(self):
        assert not self.is_record

        self.check_rosbag_files()
        rosbag_filename = "rosbag-{}.bag".format(self.file_cnt)
        rosbag_filepath = os.path.join(self.record_dir, rosbag_filename)
        cmd = self.create_cmd_rosbag(rosbag_filepath)
        rospy.loginfo("subprocess cmd: {}".format(cmd))
        p = subprocess.Popen(cmd)
        rospy.loginfo(p)
        share = {"is_running": True}

        def closure_stop():
            share["is_running"] = False

        self.closure_stop = closure_stop

        class RosbagRecorder(threading.Thread):
            def run(self):
                while True:
                    time.sleep(0.5)
                    if not share["is_running"]:
                        rospy.loginfo("kill rosbag process")
                        os.kill(p.pid, signal.SIGTERM)
                        break

        self.sound_client.say("Start saving rosbag")
        thread = RosbagRecorder()
        thread.start()
        self.is_record = not self.is_record
        rospy.loginfo("Start record rosbag : {}".format(self.file_cnt))

    def stop_record(self):
        assert self.is_record  # checking record status
        assert self.is_running  # checking rosbag status
        assert self.closure_stop is not None
        self.check_rosbag_files()
        self.closure_stop()
        self.closure_stop = None
        self.sound_client.say(
            "Finish saving rosbag. Total number is {}".format(self.file_cnt + 1)
        )
        rospy.loginfo("Stop record rosbag : {}".format(self.file_cnt))
        self.is_record = not self.is_record

    def is_running(self):
        return self.closure_stop is not None


if __name__ == "__main__":
    rospy.init_node("rosbag_recorder")
    rospy.loginfo("Start rosbag recorder...")
    rosbag_manager = RosbagRecorderNode()
    rospy.spin()
