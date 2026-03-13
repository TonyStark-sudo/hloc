import pycolmap
import numpy as np
from . import logger

class Localizer:
    def __init__(self, reconstruction, config=None):
        self.reconstruction = reconstruction
        self.config = config or {}

    def localize(self, kpq, matches, db_name, query_image_size):
        """
        kpq: keypoints in query image (N, 2)
        matches: matches from query to db (N,) -1 for no match
        db_name: name of the database image
        query_image_size: (width, height) of query image
        """
        
        # Get 3D points from DB image
        try:
            image = self.reconstruction.find_image_with_name(db_name)
            if not image:
                 return None, f"DB image {db_name} not found in reconstruction"
            db_image = self.reconstruction.images[image.image_id]
        except Exception as e:
            return None, f"Error finding DB image {db_name}: {e}"

        # Get 3D point IDs corresponding to DB keypoints
        db_kp_idx_to_3d_id = {}
        for i, p2d in enumerate(db_image.points2D):
            if p2d.has_point3D():
                db_kp_idx_to_3d_id[i] = p2d.point3D_id
                
        # Apply 0.5 offset to align with COLMAP coordinates
        kpq = kpq.astype(float) + 0.5 
        
        # Create a dummy camera for query based on image size
        q_width, q_height = query_image_size
        
        # Standard simple radial/pinhole approximation
        focal_length = max(q_width, q_height) * 1.2
        cx = q_width / 2.0
        cy = q_height / 2.0
        
        query_camera = pycolmap.Camera(
            model='SIMPLE_RADIAL',
            width=int(q_width),
            height=int(q_height),
            params=[focal_length, cx, cy, 0.0],
        )
        
        # Collect 2D-3D correspondences
        points2D = []
        points3D = []
        valid_matches_idxs = np.where(matches > -1)[0]
        
        if len(valid_matches_idxs) == 0:
             return None, "No valid matches provided"

        for q_idx in valid_matches_idxs:
            db_idx = matches[q_idx]
            if db_idx in db_kp_idx_to_3d_id:
                p3d_id = db_kp_idx_to_3d_id[db_idx]
                points2D.append(kpq[q_idx])
                # Retrieve 3D coordinate from reconstruction
                xyz = self.reconstruction.points3D[p3d_id].xyz
                points3D.append(xyz)
        
        if not points2D:
            return None, "No 2D-3D correspondences found (DB keypoints missing 3D points)"

        # Explicitly convert list of arrays to array of arrays
        points2D = np.array(points2D)
        points3D = np.array(points3D)
        
        try:
            # estimation options from config or default
            est_opts = self.config.get("estimation", {"ransac": {"max_error": 12}})
            ref_opts = self.config.get("refinement", {})
            
            ret = pycolmap.absolute_pose_estimation(
                points2D,
                points3D,
                query_camera,
                estimation_options=est_opts,
                refinement_options=ref_opts
            )
            return ret, None
        except Exception as e:
            return None, f"PnP execution failed: {e}"
