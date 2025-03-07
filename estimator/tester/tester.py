import os
import cv2
import wandb
import numpy as np
import torch
import mmengine
from mmengine.optim import build_optim_wrapper
import torch.optim as optim
import matplotlib.pyplot as plt
from mmengine.dist import get_dist_info, collect_results_cpu, collect_results_gpu
from mmengine import print_log
from estimator.utils import colorize, colorize_infer_pfv1, colorize_rescale
import torch.nn.functional as F
from tqdm import tqdm
from mmengine.utils import mkdir_or_exist
import copy
from skimage import io
import kornia
from PIL import Image

class Tester:
    """
    Tester class
    """
    def __init__(
        self, 
        config,
        runner_info,
        dataloader,
        model):
       
        self.config = config
        self.runner_info = runner_info
        self.dataloader = dataloader
        self.model = model
        self.collect_input_args = config.collect_input_args
    
    def collect_input(self, batch_data):
        collect_batch_data = dict()
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                if k in self.collect_input_args:
                    collect_batch_data[k] = v.cuda()
        return collect_batch_data
    
    @torch.no_grad()
    def run(self, cai_mode='p16', process_num=4, image_raw_shape=[2160, 3840], patch_split_num=[4, 4]):
        
        results = []
        dataset = self.dataloader.dataset
        loader_indices = self.dataloader.batch_sampler
        
        rank, world_size = get_dist_info()
        if self.runner_info.rank == 0:
            prog_bar = mmengine.utils.ProgressBar(len(dataset))

        for idx, (batch_indices, batch_data) in enumerate(zip(loader_indices, self.dataloader)):
            
            batch_data_collect = self.collect_input(batch_data)
            
            tile_cfg = dict()
            tile_cfg['image_raw_shape'] = image_raw_shape
            tile_cfg['patch_split_num'] = patch_split_num # use a customized value instead of the default [4, 4] for 4K images
            result, log_dict = self.model(mode='infer', cai_mode=cai_mode, process_num=process_num, tile_cfg=tile_cfg, **batch_data_collect) # might use test/val to split cases
            
            if self.runner_info.save:

                color_pred = colorize(result, cmap='gray_r')[:, :, [2, 1, 0]]
                cv2.imwrite(os.path.join(self.runner_info.work_dir, '{}.png'.format(batch_data['img_file_basename'][0])), color_pred)

                # Save log-scaled and inverted 0-255 grayscale depth map as PNG (closer = whiter, non-linear scale)
                depth_array_float32 = result.clone().squeeze().detach().cpu().numpy().astype(np.float32)
                min_depth_val = np.min(depth_array_float32)
                max_depth_val = np.max(depth_array_float32)

                # 1. Logarithmic Scaling (using np.log1p to handle values near zero)
                depth_array_log_scaled = np.log1p(depth_array_float32) # log(1 + depth)

                # 2. Rescale Log-Scaled Values to 0-1 (using min-max of log-scaled values)
                min_log_depth = np.min(depth_array_log_scaled)
                max_log_depth = np.max(depth_array_log_scaled)

                if max_log_depth > min_log_depth:
                    depth_array_log_rescaled_0_1 = (depth_array_log_scaled - min_log_depth) / (max_log_depth - min_log_depth)
                    depth_array_rescaled_0_255 = (depth_array_log_rescaled_0_1 * 255).astype(np.uint8)
                else:
                    depth_array_rescaled_0_255 = np.zeros_like(depth_array_log_scaled, dtype=np.uint8)

                # 3. Invert grayscale (closer = whiter)
                depth_array_inverted_0_255 = 255 - depth_array_rescaled_0_255

                rescaled_depth_image_pil = Image.fromarray(depth_array_inverted_0_255, mode='L') # 'L' mode for grayscale
                rescaled_depth_image_pil.save(os.path.join(self.runner_info.work_dir, '{}_depth_0_255_log_closer_white.png'.format(batch_data['img_file_basename'][0])))

            if batch_data_collect.get('depth_gt', None) is not None:
                metrics = dataset.get_metrics(
                    batch_data_collect['depth_gt'], 
                    result, 
                    seg_image=batch_data_collect.get('seg_image', None),
                    disp_gt_edges=batch_data.get('boundary', None), 
                    image_hr=batch_data.get('image_hr', None))
                results.extend([metrics])
            
            if self.runner_info.rank == 0:
                batch_size = len(result) * world_size
                for _ in range(batch_size):
                    prog_bar.update()
        
        if batch_data_collect.get('depth_gt', None) is not None:   
            results = collect_results_gpu(results, len(dataset))
            if self.runner_info.rank == 0:
                ret_dict = dataset.evaluate(results)
    
    