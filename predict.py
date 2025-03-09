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
            patch_split_num = [4, 4]

            print("Calculating downscaled dimensions (multiples of patch split)...")
            downscale_factor_width = 2 * patch_split_num[1]
            downscale_factor_height = 2 * patch_split_num[0]
            new_width = (original_width // downscale_factor_width) * downscale_factor_width
            new_height = (original_height // downscale_factor_height) * downscale_factor_height
            downscaled_size = (new_width, new_height)
            print(f"Original size: ({original_width}, {original_height}), Downscaled size: {downscaled_size}, patch_split_num: {patch_split_num}")
            resized_image_pil = image_pil.resize(downscaled_size, Image.BICUBIC)

            image_hr = self.transform(resized_image_pil).unsqueeze(0).to(self.device)
            image_lr = self.resize(image_hr)
            tile_cfg = dict()
            tile_cfg['image_raw_shape'] = [new_height, new_width]
            tile_cfg['patch_split_num'] = patch_split_num

            print("Running PatchFusion prediction...")
            print(f"Shape of image_lr: {image_lr.shape}") # Debug shape
            print(f"Shape of image_hr: {image_hr.shape}") # Debug shape
            with torch.no_grad():
                result, _ = self.model(
                    'infer', image_lr, image_hr,
                    cai_mode='r32',
                    process_num=1,
                    tile_cfg=tile_cfg,
                )

            depth_array_float32 = result.clone().squeeze().detach().cpu().numpy().astype(np.float32)
            depth_array_log_scaled = np.log1p(depth_array_float32)
            min_log_depth = np.min(depth_array_log_scaled)
            max_log_depth = np.max(depth_array_log_scaled)
            if max_log_depth > min_log_depth:
                depth_array_log_rescaled_0_1 = (depth_array_log_scaled - min_log_depth) / (max_log_depth - min_log_depth)
                depth_array_rescaled_0_255 = (depth_array_log_rescaled_0_1 * 255).astype(np.uint8)
            else:
                depth_array_rescaled_0_255 = np.zeros_like(depth_array_log_scaled, dtype=np.uint8)
            rescaled_depth_image_pil = Image.fromarray(255 - depth_array_rescaled_0_255, mode='L')

            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_depth_map_path = os.path.join(output_dir, "depth_map.webp")
            rescaled_depth_image_pil.save(output_depth_map_path, format="WebP", lossless=True, quality=100) # Save rescaled depth map

            print(f"Prediction finished in {time.time() - start_time:.2f} seconds.")
            return Path(output_depth_map_path) # Return path to saved WEBP depth map

        finally:
            print("Cleaning up memory...")
            torch.cuda.empty_cache()
            print("Memory cleanup complete.")