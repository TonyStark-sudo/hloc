#!/home/mark50/miniconda3/envs/hloc/bin/python

import os
import sys

ros_package_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_package_path not in sys.path:
    sys.path.append(ros_package_path)

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


import cv2
import torch
import pycolmap
from hloc import extract_features, match_features
from hloc.extract_features_one_frame import FeatureExtractor

class HlocNode:
    def __init__(self):
        rospy.init_node('hloc_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/usb_camera/color/image_raw', Image, self.image_callback)
        self.image_pub = rospy.Publisher('/usb_camera/gray/new_pub', Image, queue_size=10)

        # config
        self.feature_conf = extract_features.confs['superpoint_aachen']
        self.retrieval_conf = extract_features.confs['netvlad']

        # load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        self.feature_extractor = FeatureExtractor(self.feature_conf, device)
        self.retrieval_extractor = FeatureExtractor(self.retrieval_conf, device)

        rospy.loginfo("HLoc node initialized, waiting for images...")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        if cv_image is None:
            rospy.logerr("Failed to convert ROS Image message to OpenCV image")
            return
        rospy.loginfo("Received image, processing with HLoc...")

        # Extract features
        feats = self.feature_extractor(cv_image)
        # Extract global descriptors
        global_desc = self.retrieval_extractor(cv_image)
        
        num_kpts = feats['keypoints'].shape[0] if 'keypoints' in feats else 0
        desc_dim = global_desc['global_descriptor'].shape[0] if 'global_descriptor' in global_desc else 0
        
        rospy.loginfo(f"Extracted {num_kpts} SuperPoint keypoints.")
        rospy.loginfo(f"Extracted NetVLAD descriptor of size {desc_dim}.")
        

if __name__ == '__main__':
    node = HlocNode()
    rospy.spin()

