#!/home/mark50/miniconda3/envs/hloc/bin/python

import os
import sys
import collections.abc as collections
import time

ros_package_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_package_path not in sys.path:
    sys.path.append(ros_package_path)

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
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
from hloc.localize_sfm_one_frame import Localizer

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
        
        # Configuration
        self.enable_visualization = False # Check via param or hardcode
        
        self.image_sub = rospy.Subscriber('/usb_camera/color/image_raw', Image, self.image_callback)
        if self.enable_visualization:
            self.image_pub = rospy.Publisher('/usb_camera/color/image_with_keypoints', Image, queue_size=10)
            self.match_pub = rospy.Publisher('/hloc/image_with_matches', Image, queue_size=10)
            
        self.pose_pub = rospy.Publisher('/hloc/pose', PoseStamped, queue_size=10)
        self.position_pub = rospy.Publisher('/hloc/position', PointStamped, queue_size=10)

        # CSV output path for TUM-style pose records.
        self.pose_csv_path = Path('~/output/hloc_pose.csv').expanduser()
        self.pose_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # config
        self.feature_conf = extract_features.confs['sift']
        self.matching_conf = match_features.confs['NN-ratio']
        self.retrieval_conf = extract_features.confs['netvlad']

        # load hloc-model superpoint
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        self.feature_extractor = FeatureExtractor(self.feature_conf, device)
        self.retrieval_extractor = FeatureExtractor(self.retrieval_conf, device)

        # load hloc-model superglue
        self.feature_matcher = FeatureMatcher(self.matching_conf, device)

        # load 3D-pointcloud-model
        # model_path = './outputs/2026-02-28/sfm_superpoint+superglue_no_globalBA'
        model_path = './outputs/2026-02-28/sfm_sift_light'
        self.pointcloud_model = pycolmap.Reconstruction(model_path)
        if self.pointcloud_model.exists_point3D:
            print("Loaded 3D point cloud model successfully.")
            
        # PnP config
        self.pnp_config = {
            "estimation": {"ransac": {"max_error": 12}},
            "refinement": {}
        }
        self.localizer = Localizer(self.pointcloud_model, self.pnp_config)

        # Load 3D model features
        feature_path = Path('./outputs/2026-02-28')
        self.feature_path = feature_path
        self.global_descriptors_path = feature_path / 'global-feats-netvlad.h5'
        self.local_features_path = feature_path / 'feats-sift.h5'
        
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
        
        self.frame_count = 0

        self.total_local_extract_time = 0.0
        self.total_global_extract_time = 0.0
        self.total_global_match_time = 0.0
        self.total_local_match_time = 0.0
        self.total_pnp_time = 0.0
        self.total_frame_time = 0.0
        self.inner_points = 0

        rospy.loginfo("HLoc node initialized, waiting for images...")

    def image_callback(self, msg):
        import time
        frame_start = time.perf_counter()
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
        t0 = time.perf_counter()
        feats = self.feature_extractor(cv_image)
        local_extract_time = time.perf_counter() - t0

        rospy.loginfo(f"Feature extraction took {local_extract_time * 1000:.4f} ms.")
        # Extract global descriptors
        t0 = time.perf_counter()
        global_desc = self.retrieval_extractor(cv_image)
        global_extract_time = time.perf_counter() - t0

        rospy.loginfo(f"Global descriptor extraction took {global_extract_time * 1000:.4f} ms.")

        num_kpts = feats['keypoints'].shape[0] if 'keypoints' in feats else 0
        desc_dim = global_desc['global_descriptor'].shape[0] if 'global_descriptor' in global_desc else 0
        
        rospy.loginfo(f"Extracted {num_kpts} SuperPoint keypoints.")
        rospy.loginfo(f"Extracted NetVLAD descriptor of size {desc_dim}.")
        
        # Retrieval
        matched_kpts_indices = []
        if 'global_descriptor' in global_desc:
            query_desc = global_desc['global_descriptor']
            t0 = time.perf_counter()
            retrieval_pairs = pairs_from_retrieval_one_frame(
                query_desc, 
                self.db_names, 
                self.db_descs, 
                num_matched=20, 
                device=self.device
            )
            global_match_time = time.perf_counter() - t0
            rospy.loginfo(f"Global Retrieval took {global_match_time * 1000:.4f} ms.")

            print(f"Found {len(retrieval_pairs)} retrieval pairs.")
            if len(retrieval_pairs) > 0:
                # all matches
                score_sum = 0
                for pair in retrieval_pairs:
                    score_sum += pair[1]
                    print(f"Match: {pair}")
                
                avg_score = score_sum / len(retrieval_pairs)
                print(f"Average retrieval score: {avg_score:.4f}")

                if avg_score > 0.15:
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

                            t0 = time.perf_counter()
                            matches, scores = self.feature_matcher(feats, db_feats)
                            local_match_time = time.perf_counter() - t0
                            rospy.loginfo(f"Local matching took {local_match_time * 1000:.4f} ms.")
                            valid_matches_count = np.sum(matches > -1)
                            rospy.loginfo(f"SuperGlue found {valid_matches_count} matches with {db_name}.")
                            
                            # Store indices for visualization
                            matched_kpts_indices = np.where(matches > -1)[0]
                            
                            # Visualization of matches
                            if self.enable_visualization:
                                # Get DB image for visualization if possible. 
                                # Since we don't have the DB image file easily accessible (only features), 
                                # we can only visualize on query image or need to load DB image from disk.
                                # Assuming we have access to DB images at dataset path.
                                # Let's try to assume dataset structure: dataset/ours/db/image_name
                                try:
                                    db_image_path = Path('datasets') / '2026-02-28' / db_name
                                    if db_image_path.exists():
                                        db_img_cv = cv2.imread(str(db_image_path))
                                        if db_img_cv is not None:
                                            # Draw matches
                                            # Filter valid matches
                                            valid = matches > -1
                                            mkpts0 = feats['keypoints'][valid]
                                            mkpts1 = db_feats['keypoints'][matches[valid]]
                                            
                                            # Use hloc or opencv to draw matches
                                            # Let's use simple OpenCV drawMatches-like logic
                                            h0, w0 = cv_image.shape[:2]
                                            h1, w1 = db_img_cv.shape[:2]
                                            
                                            # Create a composite image with padding
                                            padding = 10
                                            viz_h = max(h0, h1) + padding * 2
                                            viz_w = w0 + w1 + padding * 3
                                            viz_img = np.full((viz_h, viz_w, 3), 255, dtype=np.uint8) # White background
                                            
                                            # Place images: Query Left, DB Right
                                            viz_img[padding:padding+h0, padding:padding+w0] = cv_image
                                            viz_img[padding:padding+h1, padding*2+w0:padding*2+w0+w1] = db_img_cv
                                            
                                            # Draw labels
                                            font = cv2.FONT_HERSHEY_SIMPLEX
                                            cv2.putText(viz_img, "Query Image", (padding, padding - 5), font, 0.5, (0, 0, 0), 1)
                                            cv2.putText(viz_img, "Database Image", (padding*2 + w0, padding - 5), font, 0.5, (0, 0, 0), 1)
                                            cv2.putText(viz_img, f"Matches: {valid.sum()}", (10, viz_h - 10), font, 0.5, (0, 0, 0), 1)
                                            
                                            # Draw borders
                                            cv2.rectangle(viz_img, (padding, padding), (padding+w0, padding+h0), (0, 0, 0), 2)
                                            cv2.rectangle(viz_img, (padding*2+w0, padding), (padding*2+w0+w1, padding+h1), (0, 0, 0), 2)

                                            for pt0, pt1 in zip(mkpts0, mkpts1):
                                                pt1_shifted = (int(pt1[0] + w0 + padding*2), int(pt1[1] + padding))
                                                pt0_int = (int(pt0[0] + padding), int(pt0[1] + padding))
                                                color = (0, 255, 0)
                                                cv2.line(viz_img, pt0_int, pt1_shifted, color, 1)
                                                cv2.circle(viz_img, pt0_int, 2, (0, 0, 255), -1)
                                                cv2.circle(viz_img, pt1_shifted, 2, (0, 0, 255), -1)
                                                
                                            match_msg = self.bridge.cv2_to_imgmsg(viz_img, encoding="bgr8")
                                            self.match_pub.publish(match_msg)
                                except Exception as e:
                                    rospy.logwarn(f"Match visualization failed: {e}")

                            # localization
                            if matches is not None and valid_matches_count > 100:
                                # localization
                                # Prepare data for localization: (query_kpt_idx, db_3d_id)
                                query_image_size = (cv_image.shape[1], cv_image.shape[0])
                                
                                t0 = time.perf_counter()
                                ret, error_msg = self.localizer.localize(feats['keypoints'], matches, db_name, query_image_size)

                                localize_time = time.perf_counter() - t0
                                rospy.loginfo(f"Localization took {localize_time * 1000:.4f} ms.")

                                if ret is not None and 'cam_from_world' in ret:
                                    # cam_from_world is a Rigid3d object (pycolmap)
                                    cam_from_world = ret['cam_from_world']
                                    rospy.loginfo(f"Localization SUCCESS!")

                                    frame_time = time.perf_counter() - frame_start
                                    self.frame_count += 1

                                    self.total_local_extract_time += local_extract_time
                                    self.total_global_extract_time += global_extract_time
                                    self.total_global_match_time += global_match_time
                                    self.total_local_match_time += local_match_time
                                    self.total_pnp_time += localize_time
                                    self.total_frame_time += frame_time
                                    self.inner_points += valid_matches_count

                                    rospy.loginfo("================ HLoc Timing ================")
                                    rospy.loginfo(
                                        f"Frame {self.frame_count}"
                                    )
                                    rospy.loginfo(
                                        f"Local Feature Extract : {local_extract_time*1000:.2f} ms "
                                        f"(Avg {self.total_local_extract_time/self.frame_count*1000:.2f} ms)"
                                    )
                                    rospy.loginfo(
                                        f"Global Feature Extract: {global_extract_time*1000:.2f} ms "
                                        f"(Avg {self.total_global_extract_time/self.frame_count*1000:.2f} ms)"
                                    )
                                    rospy.loginfo(
                                        f"Global Retrieval      : {global_match_time*1000:.2f} ms "
                                        f"(Avg {self.total_global_match_time/self.frame_count*1000:.2f} ms)"
                                    )
                                    rospy.loginfo(
                                        f"Local Matching        : {local_match_time*1000:.2f} ms "
                                        f"(Avg {self.total_local_match_time/self.frame_count*1000:.2f} ms)"
                                    )
                                    rospy.loginfo(
                                        f"PnP Localization      : {localize_time*1000:.2f} ms "
                                        f"(Avg {self.total_pnp_time/self.frame_count*1000:.2f} ms)"
                                    )
                                    rospy.loginfo(
                                        f"Total Frame           : {frame_time*1000:.2f} ms "
                                        f"(Avg {self.total_frame_time/self.frame_count*1000:.2f} ms)"
                                    )
                                    rospy.loginfo(
                                        f"Inner Points          : {valid_matches_count} "
                                        f"(Avg {self.inner_points/self.frame_count:.2f})"
                                    )
                                    rospy.loginfo("============================================")

                                    # Convert to cam_to_world (camera pose in world frame)
                                    cam_to_world = cam_from_world.inverse()
                                    
                                    tvec = cam_to_world.translation
                                    q = cam_to_world.rotation.quat
                                    
                                    rospy.loginfo(f"Pose Translation: {tvec}")
                                    rospy.loginfo(f"Pose Rotation (quat w,x,y,z): {q}")
                                    
                                    # Publish PoseStamped
                                    pose_msg = PoseStamped()
                                    pose_msg.header.stamp = msg.header.stamp # Use image timestamp
                                    pose_msg.header.frame_id = "map" # Assuming map frame
                                    pose_msg.pose.position.x = tvec[0]
                                    pose_msg.pose.position.y = tvec[1]
                                    pose_msg.pose.position.z = tvec[2]
                                    # pycolmap quaternion is (w, x, y, z)
                                    pose_msg.pose.orientation.w = q[0]
                                    pose_msg.pose.orientation.x = q[1]
                                    pose_msg.pose.orientation.y = q[2]
                                    pose_msg.pose.orientation.z = q[3]
                                    self.pose_pub.publish(pose_msg)

                                    # Publish PointStamped
                                    position_msg = PointStamped()
                                    position_msg.header.stamp = msg.header.stamp # Use image timestamp
                                    position_msg.header.frame_id = "map" # Assuming map frame
                                    position_msg.point.x = tvec[0]
                                    position_msg.point.y = tvec[1]
                                    position_msg.point.z = tvec[2]
                                    self.position_pub.publish(position_msg)

                                    # Save in requested TUM-like CSV format: ts,x,y,0,0,0,0,0
                                    ts = msg.header.stamp.to_sec()
                                    with open(self.pose_csv_path, 'a', encoding='utf-8') as f:
                                        f.write(f"{ts:.9f} {tvec[0]} {tvec[1]} 0 0 0 0 0\n")

                                else:
                                    if error_msg:
                                        rospy.logwarn(f"Localization failed: {error_msg}")
                                    else:
                                        rospy.logwarn("Localization PnP failed (returned None or no pose).")
                            else:
                                rospy.loginfo("Not enough matches for localization.")
                        else:
                            rospy.logwarn(f"Features for {db_name} not found in local features file.")

                    
        # Visualize keypoints
        if self.enable_visualization and 'keypoints' in feats:
            kpts = feats['keypoints']
            # Only draw on copy to keep original clean if needed, 
            # here cv_image is local and bridge converts it, so safe to draw.
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

