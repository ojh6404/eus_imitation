from copy import deepcopy
from functools import partial
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import rospy
from cv_bridge import CvBridge
from eus_imitation_msgs.msg import FloatVector
from sensor_msgs.msg import CompressedImage, Image, JointState


class ROSRobotEnv(gym.Env):
    """
    A Gym environment that interfaces with ROS topics to get real-time observations.
    """

    metadata = {
        "render_modes": ["topic"],
    }

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.obs_keys = list(self.cfg.obs.keys())
        self.obs_dims = {
            key: self.cfg.obs[key].dim for key in self.obs_keys
        }  # obs_key : dim
        self.obs = {
            obs_key: None for obs_key in self.obs_keys
        }  # initialize obs with None

        self.bridge = CvBridge()
        self.sub_obs = [
            rospy.Subscriber(
                self.cfg.obs[key].topic_name,
                eval(self.cfg.obs[key].msg_type),
                partial(self._obs_callback, key),
                queue_size=1,
                buff_size=2**24,
            )
            for key in self.obs_keys
        ]

        topic_names = [self.cfg.obs[key].topic_name for key in self.obs_keys]
        rospy.loginfo("Subscribing to topics: {}".format(topic_names))



        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.actions.dim,),
            dtype=np.float32,
        )
        spaces_dict = {
            key: (
                gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
                if "image" in key
                else gym.spaces.Box(-np.inf, np.inf, shape=(shape,), dtype=np.float32)
            )
            for key, shape in self.obs_dims.items()
        }
        self.observation_space = gym.spaces.Dict(spaces_dict)

        self.timer = rospy.Timer(rospy.Duration(1 / self.cfg.ros.rate), self._timer_callback)

    def _obs_callback(self, obs_key, msg):
        if self.cfg.obs[obs_key].msg_type == "Image":
            self.obs[obs_key] = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        elif self.cfg.obs[obs_key].msg_type == "CompressedImage":
            self.obs[obs_key] = self.bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
        elif self.cfg.obs[obs_key].msg_type == "JointState":
            self.obs[obs_key] = np.array(
                [
                    msg.position[msg.name.index(joint)]
                    for joint in self.cfg.obs[obs_key].joints
                ]
            ).astype(np.float32)
        elif self.cfg.obs[obs_key].msg_type == "FloatVector":
            self.obs[obs_key] = np.array(msg.data).astype(np.float32)
        else:
            raise NotImplementedError(f"msg_type {self.cfg.obs[obs_key].msg_type} not supported")

    def _pub_image_topic(self):
        pass # TODO

    def reset(self, seed=None, options=None):
        """
        Dummy reset method for real-world environments.
        It is expected that the real-world environment will be reset externally.
        """
        super().reset(seed=seed) # it is dummy for real world

        # wait_for_message for each obs_key
        for obs_key in self.obs_keys:
            while self.obs[obs_key] is None:
                obs_msg = rospy.wait_for_message(
                    self.cfg.obs[obs_key].topic_name,
                    eval(self.cfg.obs[obs_key].msg_type),
                    timeout=5.0,
                )
                self._obs_callback(obs_key, obs_msg)
        observation = self.obs


        info = None

        if self.render_mode == "topic":
            self._pub_image_topic()


        return observation, info

    def _timer_callback(self, event):
        """
        Timer callback for real-world environments.
        """
        self.obs_buf = deepcopy(self.obs)

    def step(self, action):
        """
        Dummy step method for real-world environments.
        Action will be executed externally.
        """
        observation = self.obs_buf
        reward = 0.0
        terminated = False
        truncated = False
        info = None

        if self.render_mode == "topic":
            self._pub_image_topic()

        return observation, reward, terminated, truncated, info
