from multiprocessing import Value
from logging import getLogger
import torch
import numpy as np

logger = getLogger()

class SurgicalMaskCollator(object):
    def __init__(
        self,
        ratio=(0.15, 0.2), # Fallback ratio for tool-less frames
        input_size=(224, 224),
        patch_size=16,
    ):
        super(SurgicalMaskCollator, self).__init__()
        self.patch_size = patch_size
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.num_patches = self.height * self.width
        self.ratio = ratio
        self._itr_counter = Value('i', -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch):
        # 1. Collate standard data
        collated_images = torch.stack([s['image'] for s in batch])
        collated_phases = torch.tensor([s['phase'] for s in batch], dtype=torch.long)
        
        collated_batch = {
            'image': collated_images, 
            'phase': collated_phases
        }
        
        B = len(batch)
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        
        collated_masks_pred, collated_masks_enc = [], []
        
        for i in range(B):
            tool_indices = batch[i]['target_patches']
            
            # --- SURGICAL I-JEPA LOGIC ---
            if tool_indices.numel() > 0:
                # Ensure indices are within 0-195 and unique
                tool_indices = tool_indices[tool_indices < self.num_patches].unique()
                #print(tool_indices)
                
                # TARGET: The Tool
                # We want the Predictor to reconstruct the tool.
                target_idx = tool_indices 
                
                # CONTEXT: Anatomy only (Strictly exclude tool)
                all_indices = torch.arange(self.num_patches)
                mask = torch.ones(self.num_patches, dtype=torch.bool)
                mask[tool_indices] = False
                
                # To make it harder, we only keep a subset of the anatomy as context
                # (Standard I-JEPA uses about 70-80% of the image as context)
                available_context = all_indices[mask]
                num_keep = int(len(available_context) * 0.75) 
                perm = torch.randperm(len(available_context), generator=g)
                context_idx = available_context[perm[:num_keep]]
                
                collated_masks_pred.append([target_idx])
                collated_masks_enc.append([context_idx])
            
            else:
                # FALLBACK: Random block masking for tool-less frames
                m = torch.randperm(self.num_patches, generator=g)
                num_keep = int(self.num_patches * 0.7)
                collated_masks_enc.append([m[:num_keep]])
                collated_masks_pred.append([m[num_keep:]])
                
        return collated_batch, collated_masks_enc, collated_masks_pred