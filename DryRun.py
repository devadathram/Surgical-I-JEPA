import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from src.datasets.marchcholec80 import MarchCholec
from src.masks.surgical_collator import SurgicalMaskCollator

dataset = MarchCholec(
    json_path="/home/devadath/cholec80/MarchPipeline/Metadata/cholec_manifest.json", 
    segmented_frames_path="/home/devadath/cholec80/OptSurgAI/sample video/processed_output",
    raw_frame_path="/home/devadath/cholec80/OptSurgAI/AEApproach/Subset_folder",
    transform=None
)
mask_collator = SurgicalMaskCollator(
    ratio=(0.15, 0.2), 
    patch_size=16
)

phase_weights = torch.tensor([5.1762, 0.4602, 1.8756, 0.3900, 2.6181, 1.6523, 1.8198]).cuda()
criterion_phase = torch.nn.CrossEntropyLoss(weight=phase_weights)


def verify_training_pipeline(dataset, collator, batch_size=2):
    print("🚀 Starting I-JEPA Surgical Pipeline Verification...\n")
    
    # 1. Test Dataset __getitem__
    sample = dataset[0]
    img_tensor = sample['image']
    target_patches = sample['target_patches']
    
    print(f"--- DATASET LEVEL CHECK ---")
    print(f"Input Tensor Shape: {img_tensor.shape} (Expected: [4, H, W])")
    print(f"Channel 1-3 (RGB) Mean: {img_tensor[:3].mean():.4f}")
    print(f"Channel 4 (Glow) Max: {img_tensor[3].max():.4f}")
    print(f"Tool Patches Found in JSON/Mask: {len(target_patches)}")
    print(f"Phase Label: {sample['phase']}\n")

    # 2. Test DataLoader + Collator
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collator
    )
    
    # Get one batch
    batch_data, masks_enc, masks_pred = next(iter(loader))
    
    imgs = batch_data['image']
    phases = batch_data['phase']

    print(f"--- BATCH LEVEL CHECK (DataLoader) ---")
    print(f"Batch Image Shape: {imgs.shape} (Expected: [{batch_size}, 4, 224, 224])")
    print(f"Phase Batch: {phases}")

    print(f"--- CLASS WEIGHTING CHECK ---")
    print(f"Phase Weights: {phase_weights}")
    # Check a single forward pass of the loss
    dummy_logits = torch.randn(2, 7).cuda()
    dummy_labels = torch.tensor([0, 3]).cuda() # Testing a rare (0) and common (3) class
    weighted_loss = criterion_phase(dummy_logits, dummy_labels)
    print(f"Weighted Phase Loss Sample: {weighted_loss.item():.4f}")
    
    # 3. Masking Logic Check
    print(f"\n--- I-JEPA MASKING CHECK ---")
    print(f"Number of Context Masks (masks_enc): {len(masks_enc)}")
    print(f"Number of Prediction Masks (masks_pred): {len(masks_pred)}")
    
    # Check if target_patches from dataset are actually inside masks_pred
    # (Assuming your collator uses the 'target_patches' key)
    example_pred_indices = masks_pred[0]
    print(f"Sample Prediction Mask Indices (First 10): {example_pred_indices[:10]}")
    
    print("\n✅ Verification Complete. If shapes and means look correct, you are ready to train.")

# EXECUTION
# Assuming your MarchCholec and Collator are already defined
verify_training_pipeline(dataset, mask_collator)