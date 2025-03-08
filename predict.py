# predict.py
from typing import Any
from cog import BasePredictor, Input, Path
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Get directory of predict.py
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) # Add to the beginning of sys.path
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "external") # Assuming 'external' is a subdirectory
PATCHFUSION_DIR = os.path.join(PROJECT_ROOT, "patchfusion") # Assuming 'patchfusion' is a subdirectory
if EXTERNAL_DIR not in sys.path and os.path.exists(EXTERNAL_DIR):
    sys.path.insert(0, EXTERNAL_DIR)
if PATCHFUSION_DIR not in sys.path and os.path.exists(PATCHFUSION_DIR):
    sys.path.insert(0, PATCHFUSION_DIR)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add repo root to path for imports


import os.path as osp
import numpy as np
import torch
from PIL import Image
import cv2
from torchvision import transforms
from mmengine.config import Config # Import Config for loading configs
from estimator.models.builder import build_model # Import build_model
from mmengine import print_log # For logging (optional, can use standard print)
import time # For timing (optional)
# --- ADDED IMPORT for ResizeDA ---
from depth_anything.transform import Resize as ResizeDA 

class Predictor(BasePredictor):
    def setup(self):
        """Load the PatchFusion model and preprocessing transforms."""
        print("Starting PatchFusion model setup (local checkpoint)...")
        start_time = time.time()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 1. Load Config ---
        config_path = "configs/patchfusion_depthanything/depthanything_vitl_patchfusion_u4k.py"
        print(f"Loading config from: {config_path}")
        self.cfg = Config.fromfile(config_path)

        # --- 2. Local Checkpoint Path ---
        local_ckp_path = "./ckps/patchfusion.pth"
        print(f"Loading checkpoint from: {local_ckp_path}")
        if not os.path.exists(local_ckp_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {local_ckp_path}")
        self.cfg.ckp_path = local_ckp_path

        # --- 3. Build Model ---
        print("Building PatchFusion model architecture...")
        model = build_model(self.cfg.model)

        print(f'Checkpoint Path: {self.cfg.ckp_path}. Loading from a local file')
        print_log(f'Checkpoint Path: {self.cfg.ckp_path}. Loading from a local file', logger='current')

        # --- 4. Load Model Weights ---
        print("Loading model weights from local checkpoint...")
        checkpoint = torch.load(self.cfg.ckp_path)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        if hasattr(model, 'load_dict'):
            load_info = model.load_dict(state_dict)
            print_log(load_info, logger='current')
        else:
            load_info = model.load_state_dict(state_dict, strict=True)
            print_log(load_info, logger='current')

        self.model = model.to(self.device).eval()

        # --- 5. Preprocessing Transforms (for normalization) ---
        print("Setting up preprocessing transforms (normalization)...")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("Preprocessing transforms (normalization) setup complete.")

        # --- 6. Dataset-like ResizeDA Transform for image_lr ---  <--- NEW CODE BLOCK START
        print("Setting up dataset-like ResizeDA transform for image_lr...")
        network_process_size = (392, 518)
        net_h, net_w = network_process_size[0], network_process_size[1]
        self.resize = ResizeDA( # Use ResizeDA for 'depth-anything' mode
            net_w, net_h, 
            keep_aspect_ratio=False, 
            ensure_multiple_of=14,
            resize_method="minimal"
        )
        print("Dataset-like ResizeDA transform setup complete.") # <--- NEW CODE BLOCK END

        print(f"Setup finished in {time.time() - start_time:.2f} seconds.")


    def predict(self,
            image: Path = Input(description="Input image for depth estimation"),
            ) -> Path:
        """Run depth estimation with PatchFusion and return path to depth map JPG."""
        print(f"Processing image: {image}")
        start_time = time.time()

        image_pil = None
        depth_map_image_pil = None
        output_depth_map_path = None

        try:
            image_pil = Image.open(str(image)).convert('RGB')
            original_width, original_height = image_pil.size
            patch_split_num = [2, 2]
            image_hr = self.transform(image_pil).unsqueeze(0).to(self.device)
            image_lr = self.resize(image_hr)
            tile_cfg = dict()
            tile_cfg['image_raw_shape'] = [original_height, original_width]
            tile_cfg['patch_split_num'] = patch_split_num

            # --- 8. Run PatchFusion Prediction ---
            print("Running PatchFusion prediction...")
            print(f"Shape of image_lr: {image_lr.shape}") # Debug shape
            print(f"Shape of image_hr: {image_hr.shape}") # Debug shape
            with torch.no_grad():
                depth_output, _ = self.model(
                    'infer',
                    image_lr=image_lr,
                    image_hr=image_hr,
                    cai_mode='r32',
                    process_num=2,
                    tile_cfg=tile_cfg,
                )

            # --- 9. Postprocess Depth Map ---
            print("Postprocessing depth map...")
            depth_map = depth_output['depth_pred'].cpu().squeeze().numpy()

            depth_map_pil = Image.fromarray(depth_map).resize((original_width, original_height), Image.BILINEAR)
            depth_map_resized = np.array(depth_map_pil)

            depth_min = depth_map_resized.min()
            depth_max = depth_map_resized.max()
            normalized_depth_map = (255 * (depth_map_resized - depth_min) / (depth_max - depth_min)).astype(np.uint8)

            # --- 10. Save Depth Map as Image ---
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_depth_map_path = os.path.join(output_dir, "depth_map.webp")
            depth_map_image_pil = Image.fromarray(normalized_depth_map, mode='L')
            depth_map_image_pil.save(output_depth_map_path, format="WebP", lossless=True, quality=100)

            print(f"Prediction finished in {time.time() - start_time:.2f} seconds.")
            return Path(output_depth_map_path)

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

        finally:
            print("Cleaning up memory...")
            if image_pil is not None:
                image_pil.close()
            if depth_map_image_pil is not None:
                depth_map_image_pil.close()
            torch.cuda.empty_cache()
            print("Memory cleanup complete.")