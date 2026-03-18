import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from collections import OrderedDict
from sklearn.decomposition import PCA
from src.helper import init_model
from src.datasets.imagenet1k import make_imagenet1k
from src.transforms import make_transforms

# --- CONFIGURATION ---
DEVICE = torch.device('cuda:0')
CHECKPOINT_PATH = '/home/devadath/cholec80/OptSurgAI/AEApproach/ijepa/logs/vanilla_test/cholec_01-latest.pth.tar'
DATA_PATH = '/home/devadath/cholec80/frames/'
VIDEO_TO_VISUALIZE = 'video17'  
PATCH_SIZE = 16  

# ---------------------------------------------------------
# 1. HELPER: Load Weights safely (stripping "module.")
# ---------------------------------------------------------
def load_weights(model, state_dict, key_prefix=''):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Remove 'module.' if present
        name = k[7:] if k.startswith('module.') else k
        # Remove specific prefix if needed (e.g. 'encoder.')
        if key_prefix and name.startswith(key_prefix):
            name = name[len(key_prefix):]
        new_state_dict[name] = v
    
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded {model.__class__.__name__}: {msg}")
    return model

# ---------------------------------------------------------
# 2. HELPER: Simple Mask Generator for Visualization
# ---------------------------------------------------------
def create_random_mask(B, N, pred_ratio=0.6):
    """
    Creates a random boolean mask for visualization.
    True = Keep (Context), False = Mask (Target)
    """
    len_keep = int(N * (1 - pred_ratio))
    noise = torch.rand(B, N, device=DEVICE)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_mask = ids_shuffle[:, len_keep:]

    # Create boolean mask (1 = Masked/Target, 0 = Visible/Context)
    mask = torch.ones([B, N], device=DEVICE)
    mask[:, :len_keep] = 0
    # Unshuffle to get back to original grid order
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return ids_keep, ids_mask, mask.bool()

# ---------------------------------------------------------
# 3. HELPER: PCA Visualization (The "Color Trick")
# ---------------------------------------------------------
def apply_pca(features):
    """
    Projects (H*W, Dim) -> (H*W, 3) for RGB visualization.
    """
    # Standardize
    features = (features - features.mean(0)) / (features.std(0) + 1e-6)
    
    # Fit PCA
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features.cpu().numpy())
    
    # Normalize to 0-1 for plotting
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    return pca_features

# ---------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------

# A. Initialize Models
print(f"Initializing models on {DEVICE}...")
encoder, predictor = init_model(device=DEVICE, model_name='vit_base')
target_encoder, _ = init_model(device=DEVICE, model_name='vit_base') # We need a separate target encoder instance

# B. Load Checkpoint
print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

# Load Encoder
load_weights(encoder, checkpoint['encoder'])
# Load Predictor (weights usually stored under 'predictor')
load_weights(predictor, checkpoint['predictor'])
# Load Target Encoder (weights usually stored under 'target_encoder')
load_weights(target_encoder, checkpoint['target_encoder'])

encoder.eval().to(DEVICE)
predictor.eval().to(DEVICE)
target_encoder.eval().to(DEVICE)

# C. Prepare Data
transform = make_transforms(crop_size=224, gaussian_blur=False)
# Pointing to root frames folder
dataset, loader, _ = make_imagenet1k(
    transform=transform, 
    batch_size=1,  # We only need 1 image
    training=False, 
    root_path=DATA_PATH,
    image_folder=None
)

# D. Run Inference
print("Running inference...")
with torch.no_grad():
    for img, label in loader:
        img = img.to(DEVICE)
        
        # 1. Create Masks
        # N = number of patches (e.g., 14x14 = 196)
        B, C, H, W = img.shape
        h, w = H // PATCH_SIZE, W // PATCH_SIZE
        N = h * w
        
        # Create indices for Context (Visible) and Target (Masked)
        ids_keep, ids_mask, bool_mask = create_random_mask(B, N, pred_ratio=0.75)
        
        # 2. Get Target Features (Ground Truth)
        # Target encoder sees the FULL image
        h_target = target_encoder(img) # (B, N, D)
        # We only care about the features at the MASKED locations for comparison
        h_target_masked = torch.gather(h_target, dim=1, index=ids_mask.unsqueeze(-1).repeat(1, 1, h_target.shape[-1]))

        # 3. Get Context Features (Encoder)
        # Encoder only sees VISIBLE patches
        # Note: Implementation details vary. Some I-JEPA encoders take the full img + mask, 
        # others take only the subset of patches. 
        # Assuming standard I-JEPA `encoder(x, mask_indices)` signature:
        h_context = encoder(img) # Get full feature map first
        # Select only kept patches
        h_context_kept = torch.gather(h_context, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, h_context.shape[-1]))
        
        # 4. Predict
        # Predictor takes Context + Target Positions -> Predicted Features
        # Note: Your predictor signature might require positional embeddings or specific masks.
        # This is a generic call:
        target_masks = [ids_mask] 

        try:
    # h_context_kept: features from the visible patches
    # target_masks: the indices of the patches to be predicted
            h_pred = predictor(h_context_kept, target_masks)
    
    # I-JEPA predictors often return a list of predictions (one per mask block)
            if isinstance(h_pred, (list, tuple)):
                h_pred = h_pred[0]
        
        except TypeError as e:
    # If it still fails, check if your specific version 
    # expects the tensor directly without the list wrapper
            h_pred = predictor(h_context_kept, ids_mask)
        
        break # Just process one image

# ---------------------------------------------------------
# E. Visualization (The Fun Part)
# ---------------------------------------------------------
print("Generating plot...")

# 1. Prepare Original Image
img_vis = img[0].permute(1, 2, 0).cpu().numpy()
# Un-normalize (Approximate ImageNet stats)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_vis = std * img_vis + mean
img_vis = np.clip(img_vis, 0, 1)

# 2. Create the "Masked View"
mask_vis = bool_mask[0].reshape(h, w).cpu().numpy()
masked_img = img_vis.copy()
masked_img[mask_vis == 1] = 0 # Set masked areas to black

# 3. PCA on Features
# We concatenate Target and Prediction to ensure they share the same PCA space (same colors = same features)
n_masked = h_target_masked.shape[1]
combined_feats = torch.cat([h_target_masked[0], h_pred[0]], dim=0)

# Apply PCA
pca_colors = apply_pca(combined_feats)

# Split back
target_colors = pca_colors[:n_masked]
pred_colors = pca_colors[n_masked:]

# 4. Reconstruct Feature Maps
# Since we only have patches for the MASKED regions, we create a blank grid and fill them in
target_map = np.zeros((N, 3))
pred_map = np.zeros((N, 3))

# Fill the masked locations with the PCA colors
mask_indices = ids_mask[0].cpu().numpy()
target_map[mask_indices] = target_colors
pred_map[mask_indices] = pred_colors

# Reshape to 2D
target_img = target_map.reshape(h, w, 3)
pred_img = pred_map.reshape(h, w, 3)

# ---------------------------------------------------------
# F. Plotting
# ---------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

ax[0].imshow(img_vis)
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(masked_img)
ax[1].set_title("Masked Input (Encoder Sees)")
ax[1].axis('off')

ax[2].imshow(target_img)
ax[2].set_title("Target Features (Ground Truth)")
ax[2].axis('off')

ax[3].imshow(pred_img)
ax[3].set_title("Predicted Features (I-JEPA)")
ax[3].axis('off')

plt.tight_layout()
save_path = 'ijepa_prediction_viz.png'
plt.savefig(save_path)
print(f"Saved visualization to {save_path}")