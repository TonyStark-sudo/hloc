import torch
import numpy as np
import cv2
import PIL.Image
from . import extractors
from .utils.base_model import dynamic_load

def resize_image(image, size, interp):
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(PIL.Image, interp[len("pil_") :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized

class FeatureExtractor:
    def __init__(self, conf, device='cpu'):
        self.conf = conf
        self.device = device
        
        Model = dynamic_load(extractors, conf['model']['name'])
        self.model = Model(conf['model']).eval().to(device)
        
    @torch.no_grad()
    def __call__(self, image):
        # image: numpy array (BGR from opencv)
        
        # Preprocessing
        prep = self.conf.get('preprocessing', {})
        resize_max = prep.get('resize_max')
        resize_force = prep.get('resize_force')
        interpolation = prep.get('interpolation', 'cv2_area')
        grayscale = prep.get('grayscale', False)
        
        # Convert to grayscale if requested
        if grayscale:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # If model expects RGB, convert BGR to RGB
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        
        if resize_max and (resize_force or max(size) > resize_max):
            scale = resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, interpolation)
        
        if grayscale:
            image_tensor = image[None]
        else:
            image_tensor = image.transpose((2, 0, 1))
            
        image_tensor = image_tensor / 255.0
        
        data = {
            "image": torch.from_numpy(image_tensor).unsqueeze(0).to(self.device),
        }
        
        pred = self.model(data)
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        
        if "keypoints" in pred:
            current_size = np.array(image.shape[:2][::-1])
            original_size = np.array(size)
            scales = (original_size / current_size).astype(np.float32)
            pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
            if "scales" in pred:
                pred["scales"] *= scales.mean()
                
        return pred
