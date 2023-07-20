#!/usr/bin/env python3
# import itertools
import sys
import time
import rospy
import rosbag
import message_filters


# Monkey-patch Time.now so we don't need to call rospy.init_node (so we don't need a roscore)
# Also solves problems with /use_sim_time (simulated time)
class MonkeyPatchTime(rospy.Time):
    def __init__(self, secs=0, nsecs=0):
        super(rospy.Time, self).__init__(secs, nsecs)

    @staticmethod
    def now():
        # initialize with wallclock
        float_secs = time.time()
        secs = int(float_secs)
        nsecs = int((float_secs - secs) * 1000000000)
        return MonkeyPatchTime(secs, nsecs)


rospy.Time = MonkeyPatchTime

# Get an input rosbag
args = rospy.myargv(sys.argv)
if len(args) < 2:
    raise RuntimeError("Missing input rosbag argument")
input_bag_name = args[1]
output_bag_name = input_bag_name + "_synchronized.bag"
print("Synchronizing {} into {}".format(input_bag_name, output_bag_name))

# Create a rosbag with only synchronized messages
with rosbag.Bag(output_bag_name, "w") as outbag:
    # Setup an ApproximateTimeSynchronizer

    # -------------------------------------------------------------------
    # EDITABLE PART: Substitute these with any kind of message or number of topics
    from sensor_msgs.msg import CompressedImage
    from sensor_msgs.msg import JointState

    joint_states_sub = message_filters.Subscriber("/joint_states", JointState)
    compressed_img_sub = message_filters.Subscriber(
        "/kinect_head/rgb/image_rect_color/compressed", CompressedImage
    )

    # ApproximateTimeSynchronizer wants a list of subscribers
    subscriber_list = [joint_states_sub, compressed_img_sub]
    # Customize ApproximateTimeSynchronizer parameters
    ats_queue_size = 1000  # Max messages in any queue
    ats_slop = 0.1  # Max delay to allow between messages
    rosbag_reader_skip_index = (
        False  # Makes opening the bag faster, but if the bag is unindexed it will fail
    )
    # -------------------------------------------------------------------

    # We want a dictionary view of the list for efficiency when dispatching messages
    subscriber_dict = {}
    for subscriber in subscriber_list:
        subscriber_dict[subscriber.topic] = subscriber
    # We want a list with the topic names in the same order to correctly dispatch messages
    #
    print("test")
    print(subscriber_dict)
    topic_names = [subscriber.topic for subscriber in subscriber_list]
    ts = message_filters.ApproximateTimeSynchronizer(
        subscriber_list,
        queue_size=ats_queue_size,
        slop=ats_slop,
        allow_headerless=False,
    )

    def callback(*msgs):
        # The callback processing the pairs of numbers that arrived at approximately the same time
        for topic_name, msg in zip(topic_names, msgs):
            # Warning: we lose the ROS original time of publication
            # This is because 1) We need to choose from which from any of the input messages
            # 2) We need to create a mechanism to get those times to this callback
            # The workaround is to use the message timestamp itself.
            outbag.write(topic_name, msg, t=msg.header.stamp)

    ts.registerCallback(callback)

    print("Opening bag... (Only reading topics {})".format(topic_names))
    bag_reader = rosbag.Bag(input_bag_name, skip_index=True)
    print("Synchronizing...")
    ini_t = time.time()
    for message_idx, (topic, msg, t) in enumerate(
        bag_reader.read_messages(topics=topic_names)
    ):
        # Send the message to the correct message_filters.Subscriber
        subscriber = subscriber_dict.get(topic)
        if subscriber:
            # Show some output to show we are alive
            if message_idx % 1000 == 0:
                print(
                    "Message #{}, Topic: {}, message stamp: {}".format(
                        message_idx, topic, msg.header.stamp
                    )
                )
            subscriber.signalMessage(msg)
    fin_t = time.time()
    total_t = fin_t - ini_t
print("Done. (Parsed {} messages in {}s)".format(message_idx, total_t))
print(
    "Output bag contains {} synchronized messages ( {} ).".format(
        outbag.get_message_count(), output_bag_name
    )
)
