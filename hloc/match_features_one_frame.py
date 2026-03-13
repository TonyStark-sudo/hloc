import torch
import numpy as np
from . import matchers
from .utils.base_model import dynamic_load

class FeatureMatcher:
    def __init__(self, conf, device='cpu'):
        self.conf = conf
        self.device = device
        self.Model = dynamic_load(matchers, conf['model']['name'])
        self.model = self.Model(conf['model']).eval().to(device)

    @torch.no_grad()
    def __call__(self, feats0, feats1):
        """
        feats0: query features (dict)
        feats1: db features (dict)
        Each dict should contain: 'keypoints', 'descriptors', 'scores', and optionally 'image_size'
        """
        data = {}
        for i, feats in enumerate([feats0, feats1]):
            # Add keypoints: [N, 2] -> [1, N, 2]
            key = f'keypoints{i}'
            if isinstance(feats['keypoints'], np.ndarray):
                kpts = torch.from_numpy(feats['keypoints']).float()
            else:
                kpts = feats['keypoints']
            # Ensure it is 2 dims before unsqueeze
            if kpts.dim() == 3: kpts = kpts.squeeze(0)
            data[key] = kpts.unsqueeze(0).to(self.device)
            
            # Add descriptors: [D, N] -> [1, D, N] (SuperPoint outputs [D, N])
            key = f'descriptors{i}'
            # Check descriptor shape
            if isinstance(feats['descriptors'], np.ndarray):
                desc = torch.from_numpy(feats['descriptors']).float()
            else:
                desc = feats['descriptors']
            # make sure descriptors are (D, N)
            if desc.shape[0] != self.conf['model'].get('descriptor_dim', 256):
                 # if not matching descriptor dim, assuming it is (N, D) and needs transpose
                 # SuperPoint default is 256
                 if desc.shape[1] == self.conf['model'].get('descriptor_dim', 256):
                     desc = desc.T
            
            # Ensure it is 2 dims before unsqueeze
            if desc.dim() == 3: desc = desc.squeeze(0)
            data[key] = desc.unsqueeze(0).to(self.device)
            
            # Add scores: [N] -> [1, N]
            key = f'scores{i}'
            if isinstance(feats['scores'], np.ndarray):
                scores = torch.from_numpy(feats['scores']).float()
            else:
                scores = feats['scores']
            # Ensure it is 1 dim before unsqueeze
            if scores.dim() == 2: scores = scores.squeeze(0)
            data[key] = scores.unsqueeze(0).to(self.device)
            
            # Add image size for normalization if available: (W, H)
            # SuperGlue needs image size to normalize keypoints
            if 'image_size' in feats:
                if isinstance(feats['image_size'], np.ndarray):
                    size = tuple(feats['image_size'].tolist())
                elif isinstance(feats['image_size'], torch.Tensor):
                    size = tuple(feats['image_size'].cpu().numpy().tolist())
                else:
                    size = tuple(feats['image_size'])
                
                # Check if size is correct format (W, H) or (H, W) extract_features usually stores (W, H)
                # We create a dummy tensor with shape (1, 1, H, W)
                data[f'image{i}'] = torch.empty((1, 1, int(size[1]), int(size[0])), device=self.device)

        pred = self.model(data)
        matches = pred['matches0'][0].cpu().numpy()
        scores = pred['matching_scores0'][0].cpu().numpy()
        
        return matches, scores

