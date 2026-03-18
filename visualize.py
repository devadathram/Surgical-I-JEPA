import torch
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import OrderedDict
from src.helper import init_model
from src.datasets.imagenet1k import make_imagenet1k
from src.transforms import make_transforms

# 1. Setup & Load Model
device = torch.device('cuda:0')
checkpoint_path = '/home/devadath/cholec80/OptSurgAI/AEApproach/ijepa/logs/vanilla_test/cholec_01-latest.pth.tar'

# Initialize model
encoder, predictor = init_model(device=device, model_name='vit_base')

# --- FIX: Direct loading to handle the "module." prefix ---
print(f"Loading weights from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# In I-JEPA checkpoints, the encoder weights are under the 'encoder' key
state_dict = checkpoint['encoder']
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    # Strip 'module.' if it exists (added by DistributedDataParallel)
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v

msg = encoder.load_state_dict(new_state_dict, strict=False)
print(f"Loaded encoder with message: {msg}")
encoder.eval()
encoder.to(device)

# 2. Prepare Data
transform = make_transforms(crop_size=224, gaussian_blur=False)

# --- FIX: Point to the parent 'frames' folder so ImageFolder finds 'video17' as a class ---
root_path = '/home/devadath/cholec80/frames/'
# If your make_imagenet1k uses the training_videos filter, make sure video17 is in it!
_, loader, _ = make_imagenet1k(
    transform=transform, 
    batch_size=32, 
    training=False, 
    root_path=root_path,
    image_folder=None 
)

features = []
labels = []

print("Extracting features...")
with torch.no_grad():
    for i, (imgs, target_labels) in enumerate(loader):
        # We only want to visualize video17. Check if the current batch is from video17
        # Note: In ImageFolder, target_labels are indices. video17's index depends on folder order.
        
        if len(features) >= 2000: 
            break 
            
        imgs = imgs.to(device)
        # Get the CLS token (index 0)
        emb = encoder(imgs) 
        cls_token = emb[:, 0, :]

        features.append(cls_token.cpu().numpy())
        labels.append(target_labels.numpy())

if len(features) == 0:
    raise ValueError("No frames found! Check if 'video17' is in your training_videos list inside make_imagenet1k.")

features = np.concatenate(features)
labels = np.concatenate(labels)

# 3. Run t-SNE
print(f"Running t-SNE on {len(features)} samples...")
tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
embed_2d = tsne.fit_transform(features)

# 4. Plotting
phase_names = [
    "0: Preparation", "1: Calot Triangle", "2: Clipping", 
    "3: Dissection", "4: Packaging", "5: Cleaning", "6: Retraction"
]

plt.figure(figsize=(12, 8))
scatter = plt.scatter(embed_2d[:, 0], embed_2d[:, 1], 
                      c=labels, cmap='tab10', alpha=0.6, 
                      vmin=0, vmax=6)

cbar = plt.colorbar(scatter, ticks=range(7))
cbar.ax.set_yticklabels(phase_names) 
plt.title('t-SNE Visualization of I-JEPA Latent Space')
plt.savefig('tsne_results_v2.png')
print("Plot saved as tsne_results_v2.png")