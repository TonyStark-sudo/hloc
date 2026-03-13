#!/home/mark50/miniconda3/envs/hloc/bin/python

import os
import sys
import collections.abc as collections

ros_package_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_package_path not in sys.path:
    sys.path.append(ros_package_path)

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


import cv2
import torch
import pycolmap
import h5py
import numpy as np
from pathlib import Path
from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.extract_features_one_frame import FeatureExtractor
from hloc.match_features_one_frame import FeatureMatcher
from hloc.pairs_from_retrieval_one_frame import pairs_from_retrieval_one_frame
from hloc.utils.io import list_h5_names
from hloc.utils.parsers import parse_image_lists

def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
        if len(names) == 0:
            raise ValueError(f"Could not find any image with the prefix `{prefix}`.")
    elif names is not None:
        if isinstance(names, (str, Path)):
            names = parse_image_lists(names)
        elif isinstance(names, collections.Iterable):
            names = list(names)
        else:
            raise ValueError(
                f"Unknown type of image list: {names}."
                "Provide either a list or a path to a list file."
            )
    else:
        names = names_all
    return names


def get_descriptors(names, path, name2idx=None, key="global_descriptor"):
    if name2idx is None:
        with h5py.File(str(path), "r", libver="latest") as fd:
            desc = [fd[n][key].__array__() for n in names]
    else: 
        desc = []
        # if path is not a list, wrap it in a list so indexing works
        if isinstance(path, (str, Path)):
            path = [path]
            
        for n in names:
            with h5py.File(str(path[name2idx[n]]), "r", libver="latest") as fd:
                desc.append(fd[n][key].__array__())
    return torch.from_numpy(np.stack(desc, 0)).float()

class HlocNode:
    def __init__(self):
        rospy.init_node('hloc_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/usb_camera/color/image_raw', Image, self.image_callback)
        self.image_pub = rospy.Publisher('/usb_camera/color/image_with_keypoints', Image, queue_size=10)

        # config
        self.feature_conf = extract_features.confs['superpoint_aachen']
        self.matching_conf = match_features.confs['superglue']
        self.retrieval_conf = extract_features.confs['netvlad']

        # load hloc-model superpoint
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        self.feature_extractor = FeatureExtractor(self.feature_conf, device)
        self.retrieval_extractor = FeatureExtractor(self.retrieval_conf, device)

        # load hloc-model superglue
        self.feature_matcher = FeatureMatcher(self.matching_conf, device)

        # load 3D-pointcloud-model
        model_path = './outputs/ours/sfm_superpoint+superglue'
        self.pointcloud_model = pycolmap.Reconstruction(model_path)
        if self.pointcloud_model.exists_point3D:
            print("Loaded 3D point cloud model successfully.")

        # Load 3D model features
        feature_path = Path('./outputs/ours')
        self.feature_path = feature_path
        self.global_descriptors_path = feature_path / 'global-feats-netvlad.h5'
        self.local_features_path = feature_path / 'feats-superpoint-n4096-r1024.h5'
        
        if isinstance(self.global_descriptors_path, (Path, str)):
            global_descriptors_path = [self.global_descriptors_path]
        
        name2db = {n: i for i, p in enumerate(global_descriptors_path) for n in list_h5_names(p)}
        db_names_h5 = list(name2db.keys())
        db_list = []
        db_names = parse_names("db", db_list, db_names_h5)
        if len(db_names) == 0:
            raise ValueError("Could not find any database images.")

        self.db_descs = get_descriptors(db_names, self.global_descriptors_path, name2db)
        self.db_names = db_names
        self.device = device
        
        print(f"Loaded {len(self.db_descs)} global descriptors from the database.")
        
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
        
        # Retrieval
        matched_kpts_indices = []
        if 'global_descriptor' in global_desc:
            query_desc = global_desc['global_descriptor']
            retrieval_pairs = pairs_from_retrieval_one_frame(
                query_desc, 
                self.db_names, 
                self.db_descs, 
                num_matched=20, 
                device=self.device
            )
            
            print(f"Found {len(retrieval_pairs)} retrieval pairs.")
            if len(retrieval_pairs) > 0:
                # all matches
                score_sum = 0
                for pair in retrieval_pairs:
                    score_sum += pair[1]
                    print(f"Match: {pair}")
                
                avg_score = score_sum / len(retrieval_pairs)
                print(f"Average retrieval score: {avg_score:.4f}")

                if avg_score > 0.25:
                    best_match = retrieval_pairs[0]
                    db_name = best_match[0]
                    print(f"Similar scene detected! Starting local matching with best match: {db_name}")
                    
                    with h5py.File(str(self.local_features_path), 'r') as fd:
                        if db_name in fd:
                            grp = fd[db_name]
                            # Add image size (W, H) for matching normalization
                            feats['image_size'] = np.array([cv_image.shape[1], cv_image.shape[0]])
                            
                            db_feats = {
                                'keypoints': grp['keypoints'].__array__(),
                                'scores': grp['scores'].__array__(),
                                'descriptors': grp['descriptors'].__array__(),
                                'image_size': grp['image_size'].__array__() if 'image_size' in grp else None
                            }
                            
                            if 'scales' in grp:
                                db_feats['scales'] = grp['scales'].__array__()
                            
                            matches, scores = self.feature_matcher(feats, db_feats)
                            valid_matches_count = np.sum(matches > -1)
                            rospy.loginfo(f"SuperGlue found {valid_matches_count} matches with {db_name}.")
                            
                            # Store indices for visualization
                            matched_kpts_indices = np.where(matches > -1)[0]
                        else:
                            rospy.logwarn(f"Features for {db_name} not found in local features file.")

                    
        # Visualize keypoints
        if 'keypoints' in feats:
            kpts = feats['keypoints']
            for kp in kpts:
                cv2.circle(cv_image, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)
            
            # Draw matched keypoints in Red and larger
            for idx in matched_kpts_indices:
                kp = kpts[idx]
                cv2.circle(cv_image, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)

            try:
                out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                self.image_pub.publish(out_msg)
            except Exception as e:
                rospy.logerr(f"CvBridge Publish Error: {e}")
        


if __name__ == '__main__':
    node = HlocNode()
    rospy.spin()

