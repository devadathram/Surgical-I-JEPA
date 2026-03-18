import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn.functional as F
from src.helper import init_model
from src.masks.surgical_collator import SurgicalMaskCollator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PHASE_NAMES = [
    "Preparation", "Calot Triangle Dissection", "Clipping & Cutting", 
    "Gallbladder Dissection", "Gallbladder Packaging", "Cleaning", "Retraction"
]

def visualize_complete_surgical_ai(model, predictor, image_path, mask_path, patch_size=16):
    model.eval()
    predictor.eval()
    device = next(model.parameters()).device

    # 1. Load Data
    raw_img = cv2.imread(image_path)
    raw_img = cv2.resize(raw_img, (224, 224))
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    
    raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    raw_mask = cv2.resize(raw_mask, (14, 14), interpolation=cv2.INTER_NEAREST)
    
    # Prep 4-channel input
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    dummy_channel = torch.zeros((1, 1, 224, 224)).to(device)
    img_t = torch.cat([img_t, dummy_channel], dim=1)

    # 2. Run Inference
    with torch.no_grad():
        # Get Phase Predictions and Latents
        # Based on your forward: returns x, phase_logits
        latents, phase_logits = model(img_t)
        phase_probs = F.softmax(phase_logits, dim=-1).cpu().numpy()[0]
        predicted_phase = PHASE_NAMES[np.argmax(phase_probs)]

        # Get Attention (Last Block)
        x_attn = model.patch_embed(img_t)
        x_attn = x_attn + model.interpolate_pos_encoding(x_attn, model.pos_embed)
        for blk in model.blocks[:-1]:
            x_attn = blk(x_attn)
        attn_out = model.blocks[-1](x_attn, return_attention=True)
        importance_map = torch.mean(torch.mean(attn_out[0], dim=0), dim=0).reshape(14, 14).cpu().numpy()
        importance_map = cv2.resize(importance_map, (224, 224))

        # Dynamic Masking Logic
        num_patches = 196
        num_masks = int(0.25 * num_patches) 

        # 2. Generate random indices
        all_indices = np.arange(num_patches)
        np.random.shuffle(all_indices)
        target_indices = all_indices[:num_masks]
        context_indices = all_indices[num_masks:]

        # 3. Sort them for consistent tensor processing
        target_indices.sort()
        context_indices.sort()
        t_idx = torch.tensor(target_indices).unsqueeze(0).to(device)
        c_idx = torch.tensor(context_indices).unsqueeze(0).to(device)

        context_feats, _ = model(img_t, masks=c_idx)
        pred_targets = predictor(context_feats, c_idx, t_idx)
        true_targets = latents[:, target_indices, :]
        similarity = F.cosine_similarity(pred_targets, true_targets, dim=-1)

    # 3. Final Dashboard Plot
    fig = plt.figure(figsize=(25, 10))
    gs = fig.add_gridspec(2, 4)

    # Row 1: The Visuals
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.set_title("Input Frame", fontsize=16)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_rgb)
    ax2.imshow(importance_map, cmap='magma', alpha=0.5)
    ax2.set_title("Attention (Tool Focus)", fontsize=16)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    context_view = img_rgb.copy()
    for idx in target_indices:
        r, c = (idx // 14) * patch_size, (idx % 14) * patch_size
        context_view[r:r+patch_size, c:c+patch_size] = 0
    ax3.imshow(context_view)
    ax3.set_title("Dynamic Masked Input", fontsize=16)
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    sim_grid = np.zeros((14, 14))
    for i, idx in enumerate(target_indices):
        sim_grid[idx // 14, idx % 14] = similarity[0, i].cpu().item()
    im = ax4.imshow(sim_grid, cmap='viridis')
    ax4.set_title(f"Prediction Match: {similarity.mean():.4f}", fontsize=16)
    fig.colorbar(im, ax=ax4)
    ax4.axis('off')

    # Row 2: The Phase Prediction (Spanning multiple columns)
    ax5 = fig.add_subplot(gs[1, :])
    colors = ['gray'] * 7
    colors[np.argmax(phase_probs)] = '#2ecc71' # Green for the winner
    ax5.barh(PHASE_NAMES, phase_probs, color=colors)
    ax5.set_xlim(0, 1.0)
    ax5.set_title(f"Predicted Phase: {predicted_phase} ({np.max(phase_probs)*100:.1f}%)", fontsize=18, fontweight='bold')
    ax5.set_xlabel("Confidence Score", fontsize=14)
    ax5.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("Complete_Surgical_IJEPA_Dashboard_vanilla.png", dpi=300)


collator = SurgicalMaskCollator()

encoder, predictor = init_model(
    device=device,
    patch_size=16,
    crop_size=224, # Adjust to your training size
    model_name="vit_base" # Adjust to yours
)
model_path = "/home/devadath/cholec80/OptSurgAI/AEApproach/ijepa/logs/vanilla_test/cholec_phase_01-ep50.pth.tar"
checkpoint = torch.load(model_path, map_location=device)
encoder.load_state_dict(checkpoint['encoder'])
predictor.load_state_dict(checkpoint['predictor'])

img_path = "/home/devadath/cholec80/OptSurgAI/AEApproach/Subset_folder/video01/frame_00700.jpg"
mask_path = "/home/devadath/cholec80/OptSurgAI/sample video/processed_output/video01/frame_00700.jpg"

visualize_complete_surgical_ai(encoder, predictor, img_path, mask_path=mask_path)
print(f"Success! Check context_target_visualization.png to see if the model found the tools in {img_path}")