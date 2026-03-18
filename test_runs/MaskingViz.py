import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# --- CONFIGURE THIS ---
# Path to ONE of your segmentation mask PNG files (class-colored)
MASK_PATH = "/home/devadath/cholec80/OptSurgAI/sample video/processed_output/video15/frame_31175.jpg"
# I-JEPA Standard patch size
PATCH_SIZE = 16 
# Target Image Size (matching your I-JEPA config)
IMG_SIZE = 224
# How many blocks to mask randomly (tune this to make it dramatic)
NUM_MASKS = 10 

# Check if file exists
if not os.path.exists(MASK_PATH):
    print(f"❌ Error: Could not find mask at {MASK_PATH}. Please update the path.")
    exit()

def get_rand_masks(img, patch_size, num_masks, target_shape):
    H, W = target_shape
    mask_map = torch.ones((1, H // patch_size, W // patch_size))
    
    for _ in range(num_masks):
        # Pick a random starting patch (excluding edges for simplicity)
        h0 = np.random.randint(2, (H // patch_size) - 6)
        w0 = np.random.randint(2, (W // patch_size) - 6)
        
        # Define a mask block (e.g., a 4x4 patch block)
        mask_map[:, h0:h0+4, w0:w0+4] = 0.0
        
    # Upscale to full image resolution
    mask_img = F.interpolate(mask_map.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
    return mask_img

img = cv2.imread(MASK_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

# Convert to Tensor [C, H, W]
transform = transforms.ToTensor()
img_tensor = transform(img_resized)

# 2. Generate and Apply Masks
import torch.nn.functional as F
# Generate the I-JEPA context masks
context_mask = get_rand_masks(img_tensor, PATCH_SIZE, NUM_MASKS, (IMG_SIZE, IMG_SIZE))

# Apply the masks (black out regions)
masked_img_tensor = img_tensor * context_mask

# 3. Convert back to displayable format [H, W, C]
masked_img = masked_img_tensor.permute(1, 2, 0).numpy()

# --- PLOT & SAVE ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img_resized)
axes[0].set_title("Original Segmentation Mask")
axes[0].axis('off')

axes[1].imshow(masked_img)
axes[1].set_title("Masked with I-JEPA Rand-Masking")
axes[1].axis('off')

plt.tight_layout()
# Save as a PNG for your presentation (high quality)
plt.savefig("ijepa_rand_masking_viz2.jpg")
plt.show()

print("✅ Visualization saved as 'ijepa_rand_masking_viz.png'")