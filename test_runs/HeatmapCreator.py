import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

# --- Configuration ---
IMG_SIZE = 224
PATCH_SIZE = 16
N_PATCHES = IMG_SIZE // PATCH_SIZE # 14x14 grid
N_ITERATIONS = 500 # More iterations = smoother heatmap
MASK_RATIO = 0.6   # Standard I-JEPA masking ratio

# Load one of your masks (replace with your path)
mask_path = "/home/devadath/cholec80/OptSurgAI/sample video/processed_output/video07/frame_11639.jpg" 
mask_img = cv2.imread(mask_path, 0) # Load as grayscale
mask_resized = cv2.resize(mask_img, (IMG_SIZE, IMG_SIZE))

# --- Heatmap Logic ---
heatmap = np.zeros((N_PATCHES, N_PATCHES))

for _ in range(N_ITERATIONS):
    # Simulate I-JEPA's torch.randperm(num_patches)
    perm = np.random.permutation(N_PATCHES**2)
    num_mask = int(N_PATCHES**2 * MASK_RATIO)
    masked_indices = perm[:num_mask]
    
    # Update heatmap grid
    grid = np.zeros(N_PATCHES**2)
    grid[masked_indices] = 1
    heatmap += grid.reshape((N_PATCHES, N_PATCHES))

# Normalize heatmap
heatmap /= N_ITERATIONS

# --- Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: The Surgical Mask with Tool Highlighted
axes[0].imshow(mask_resized, cmap='gray')
axes[0].set_title("Ground Truth Mask (Tool is the white part)")
axes[0].axis('off')

# Plot 2: The Masking Heatmap
im = axes[1].imshow(heatmap, cmap='hot', interpolation='nearest')
axes[1].set_title(f"I-JEPA Masking Distribution ({N_ITERATIONS} iterations)")
fig.colorbar(im, ax=axes[1], label='Probability of being masked')
axes[1].axis('off')

plt.tight_layout()
plt.savefig("ijepa_heatmap_masking_viz.png")
plt.show()