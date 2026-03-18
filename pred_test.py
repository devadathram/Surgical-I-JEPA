from src.models.vision_transformer import VisionTransformer
from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.models.vision_transformer import VisionTransformerPredictor as Predictor
from src.helper import load_checkpoint
from src.transforms import make_transforms
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.datasets.cholec80 import CholecValDataset

# --- Configuration (Match your YAML) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patch_size = 16
embed_dim = 768  # Or 768 for ViT-Base
img_size = (224, 224)  

# 1. Initialize Encoder (The "Context" Model)
encoder = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    embed_dim=embed_dim,
    depth=12,
    num_heads=12,
    num_classes=7 # Cholec80 has 7 phases
).to(device)

# 2. Initialize Predictor (The "Guessing" Model)
predictor = Predictor(
    embed_dim=embed_dim,
    predictor_embed_dim=384,
    depth=6,
    num_heads=12,
    num_patches=196,
).to(device)

# 3. Setup Mask Collator
mask_collator = MBMaskCollator(
    input_size=img_size,
    patch_size=patch_size
    )

# 4. Setup Transforms
transform = make_transforms(crop_size=img_size)

checkpoint_path = "/home/devadath/cholec80/OptSurgAI/AEApproach/ijepa/logs/vanilla_test/cholec_phase_01-ep50.pth.tar"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load Encoder and Predictor states
    encoder.load_state_dict(checkpoint['encoder'])
    predictor.load_state_dict(checkpoint['predictor'])
    
    print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
else:
    print("No checkpoint found! Ensure the path is correct.")

val_pickle_path = "/home/devadath/cholec80/OptSurgAI/AEApproach/ijepa/ch80_labels/labels/val/1fps.pickle"

def visualize_prediction(encoder, predictor, val_pickle, frames_root, transform, collator, device):
    # 1. Prepare Data
    dataset = CholecValDataset(val_pickle, frames_root, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Get a single sample
    img, label = next(iter(loader))
    img = img.to(device)

    # 2. Generate Masks
    # We pass a dummy label list to the collator
    _, masks_enc, masks_pred = collator(([img], [label]))
    masks_enc = [m.to(device) for m in masks_enc]
    masks_pred = [m.to(device) for m in masks_pred]

    # 3. Model Forward Pass
    encoder.eval()
    predictor.eval()
    with torch.no_grad():
        # Get latent representations for context patches
        z, logits = encoder(img, masks_enc)
        # Predict the latent representations for the target patches
        _ = predictor(z, masks_enc, masks_pred)
        
        pred_phase = torch.argmax(logits, dim=-1).item()

    # 4. Plotting logic
    plot_ijepa_result(img[0], masks_enc[0][0], masks_pred[0][0], label.item(), pred_phase)

def plot_ijepa_result(img_tensor, enc_idx, pred_idx, true_label, pred_label):
    # Un-normalize image
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    
    patch_size = 16
    grid_size = 224 // patch_size
    
    # Create masks
    context_mask = np.zeros((grid_size, grid_size))
    target_mask = np.zeros((grid_size, grid_size))
    
    for i in enc_idx: context_mask[i // grid_size, i % grid_size] = 1
    for i in pred_idx: target_mask[i // grid_size, i % grid_size] = 1

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: What the model sees (Context)
    ax[0].imshow(img)
    # Highlight the invisible areas in dark gray
    ax[0].imshow(np.kron(1-context_mask, np.ones((patch_size, patch_size))), cmap='gray', alpha=0.8)
    ax[0].set_title(f"Context (Input to Encoder)\nTrue Phase: {true_label}")
    
    # Right: What the model is guessing (Target)
    ax[1].imshow(img)
    # Highlight the target areas in Blue/Cyan
    ax[1].imshow(np.kron(target_mask, np.ones((patch_size, patch_size))), cmap='cool', alpha=0.5)
    ax[1].set_title(f"Target (Prediction Regions)\nPredicted Phase: {pred_label}")
    
    plt.tight_layout()
    plt.show()

# EXECUTE
visualize_prediction(encoder, predictor, val_pickle_path, "/home/devadath/cholec80/frames/", transform, mask_collator, device)