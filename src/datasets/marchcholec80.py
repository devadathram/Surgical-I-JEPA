import os
import json
import torch
import cv2
import numpy as np
from logging import getLogger
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import distance_transform_edt

_GLOBAL_SEED = 0
logger = getLogger()

class MarchCholec(Dataset):
    def __init__(self, json_path, segmented_frames_path, raw_frame_path, transform=None, patch_size=16):
        self.frame_root = segmented_frames_path
        self.raw_frame_root = raw_frame_path
        self.transform = transform
        self.patch_size = patch_size
        self.grid_size = 224 // patch_size

        with open(json_path, "r") as file:
            self.full_data = json.load(file)

        self.samples = []
        self.data_dict = {}  

        logger.info("Filtering dataset for frames with segmentations...")
        for frame_id, meta in self.full_data.items():
            vid_id, frame_num = frame_id.split('_')
            # Path to the SAM segmentation frame
            mask_path = os.path.join(self.frame_root, f"video{vid_id}", f"frame_{int(frame_num):05d}.jpg")
            
            if os.path.exists(mask_path):
                self.samples.append(frame_id)
                self.data_dict[frame_id] = meta
        
        logger.info(f"Initialization complete. Found {len(self.samples)} valid pairs out of {len(self.full_data)} total frames.")  


    def distance_transform(self, img_rgb):
        # Tools: Green, Blue, Red (Clipper/Hook/Grasper)
        mask_green = cv2.inRange(img_rgb, (50, 150, 50), (110, 255, 110))
        mask_blue  = cv2.inRange(img_rgb, (50, 50, 150), (110, 110, 255))
        mask_red   = cv2.inRange(img_rgb, (250, 90, 90), (255, 110, 110))
        master_mask = cv2.bitwise_or(mask_green, cv2.bitwise_or(mask_blue, mask_red)) 

        kernel = np.ones((5, 5), np.uint8)
        master_mask = cv2.dilate(master_mask, kernel, iterations=1)
        
        # Exponential 'Glow' for spatial interaction
        external_dist = distance_transform_edt(master_mask == 0)
        sigma = 60
        dist_map = np.exp(-(external_dist**2) / (2 * sigma**2))
        dist_map[master_mask > 0] = 0
        return dist_map.astype(np.float32), master_mask

    def pixel_to_patch_idx(self, bboxes, org_w=854, org_h=480):
        target_patches = set()
        grid_size = 14
        
        # Calculate the "Original Pixel" size of one grid cell
        patch_w = org_w / grid_size # 61.0
        patch_h = org_h / grid_size # 34.28

        for box in bboxes:
            x1, y1, x2, y2 = box

            # 1. Map pixels directly to grid indices
            col_start = int(x1 // patch_w)
            col_end   = int(np.ceil(x2 / patch_w))
            
            row_start = int(y1 // patch_h)
            row_end   = int(np.ceil(y2 / patch_h))

            # 2. Safety clip to the 14x14 grid
            col_start, col_end = max(0, col_start), min(grid_size, col_end)
            row_start, row_end = max(0, row_start), min(grid_size, row_end)

            # 3. Add to set (Only the indices!)
            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    idx = r * grid_size + c
                    target_patches.add(idx)

        return torch.tensor(list(target_patches), dtype=torch.long)
    
    
    def mask_to_patch_idx(self, master_mask):
        if master_mask.shape != (224, 224):
            master_mask = cv2.resize(master_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(master_mask).unsqueeze(0).unsqueeze(0).float()
        grid_mask = torch.nn.functional.max_pool2d(mask_tensor, kernel_size=self.patch_size, stride=self.patch_size)
        
        target_patches = torch.where(grid_mask.flatten() > 0)[0]
        
        # SAFETY SHIELD: Prevent IndexError in Collator
        max_valid_idx = self.grid_size ** 2 # 196
        target_patches = target_patches[target_patches < max_valid_idx]
        
        return target_patches

    def __getitem__(self, index):
        try:
            frame_id = self.samples[index]
            meta = self.data_dict[frame_id]
            vid_id, frame_num = frame_id.split('_')

            raw_img_path = os.path.join(self.raw_frame_root, f"video{vid_id}", f"frame_{int(frame_num):05d}.jpg")
            raw_img = cv2.imread(raw_img_path)
            raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            
            img_path = os.path.join(self.frame_root, f"video{vid_id}", f"frame_{int(frame_num):05d}.jpg")
            mask_bgr = cv2.imread(img_path)
                
            
            mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
            
            dst_map, master_mask = self.distance_transform(mask_rgb)
            raw_rgb = cv2.resize(raw_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            dst_map = cv2.resize(dst_map, (224, 224), interpolation=cv2.INTER_LINEAR)

            raw_rgb_norm = raw_rgb.astype(np.float32) / 255.0            
            combined = np.concatenate([raw_rgb_norm, dst_map[:, :, np.newaxis]], axis=2) 
            org_h, org_w = mask_rgb.shape[:2]

            # 1. Get Patches from JSON BBoxes (If they exist)
            json_bboxes = meta.get('bboxes', [])
            if len(json_bboxes) > 0:
                target_patches_from_json = self.pixel_to_patch_idx(json_bboxes, org_w=org_w, org_h=org_h)
            else:
                target_patches_from_json = torch.tensor([], dtype=torch.long)
            
            # 2. Get Patches from the Mask 
            target_patches_from_mask = self.mask_to_patch_idx(master_mask)
            
            # 3. Combine them (Union of both)
            target_patches = torch.unique(torch.cat([target_patches_from_json, target_patches_from_mask]))
            if self.transform:
                combined = self.transform(combined)
            else:
                combined = torch.from_numpy(combined).permute(2, 0, 1)


            PHASE_MAP = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    'CleaningCoagulation': 5,
    "GallbladderRetraction": 6
}    

            return {
                "image": combined, 
                "target_patches": target_patches, 
                "phase": torch.tensor(PHASE_MAP[meta['phase']], dtype=torch.long)
            }

        except Exception as e:
            logger.warning(f"Error at index {index}: {e}")
            return self.__getitem__(np.random.randint(0, len(self.samples)))

    def __len__(self):
        return len(self.samples)

def make_imagenet1k(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,      # Path to segmented frames
    json_path=None,      # NEW: Path to your Stage 1 JSON
    raw_frame_path=None, # Path to raw frames
    drop_last=True,
):
    """
    Factory function for I-JEPA training loop.
    """
    dataset = MarchCholec(
        json_path=json_path,
        segmented_frames_path=root_path,
        raw_frame_path=raw_frame_path,
        transform=transform
    )

    # DistributedSampler handles splitting the data across multiple GPUs
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=_GLOBAL_SEED
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator, # This will be your SurgicalBBoxCollator
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=True
    )

    return dataset, loader, sampler