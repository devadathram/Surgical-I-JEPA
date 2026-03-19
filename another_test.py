from src.helper import init_model
from src.masks.surgical_collator import SurgicalMaskCollator
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PHASE_NAMES = [
    "Preparation", "Calot Triangle Dissection", "Clipping & Cutting", 
    "Gallbladder Dissection", "Gallbladder Packaging", "Cleaning Coagulation", "Retraction"
]

MODEL_PATH = "/home/devadath/cholec80/OptSurgAI/AEApproach/ijepa/logs/vanilla_test/cholec_constrained_01-ep50.pth.tar"
SEGMENTATION_MASK_FOLDER_PATH = "/home/devadath/cholec80/OptSurgAI/sample video/processed_output"
RAW_FRAME_FOLDER_PATH = "/home/devadath/cholec80/OptSurgAI/AEApproach/Subset_folder"
JSON_PATH = "/home/devadath/cholec80/MarchPipeline/Metadata/cholec_manifest.json"
TOOL_ANNOTATION_PATH = "/home/devadath/cholec80/tool_annotations"

results_log = []


def visualize_complete_surgical_ai(model, predictor, image_path, mask_path, tool_annotation, phase_annotation, patch_size=16):
    model.eval()
    predictor.eval()
    raw_img = cv2.imread(image_path)
    raw_img = cv2.resize(raw_img, (224, 224))
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    
    raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    raw_mask = cv2.resize(raw_mask, (14, 14), interpolation=cv2.INTER_NEAREST)
    
    # Prep 4-channel input
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    dummy_channel = torch.zeros((1, 1, 224, 224)).to(device)
    img_t = torch.cat([img_t, dummy_channel], dim=1)

    total_phase_probs = np.zeros(len(PHASE_NAMES))
    total_sim_scores = 0

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
    pass


def get_ground_truth_tools(tool_file_path, target_frame):
    """
    Parses the Cholec80 tool.txt file and returns active tools for a specific frame.
    """
    if not os.path.exists(tool_file_path):
        return ["File Not Found"]
    
    tools_header = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
    active_tools = []
    
    # Read the file and find the row where 'Frame' matches our target_frame
    try:
        with open(tool_file_path, 'r') as f:
            lines = f.readlines()[1:] # Skip header
            for line in lines:
                parts = line.split() # Splits by any whitespace/tab
                if not parts: continue
                
                frame_idx = int(parts[0])
                # Since annotations are every 25 frames, find the nearest one
                if abs(frame_idx - target_frame) < 13: 
                    # Check columns 1-7 for '1' (active)
                    for i, val in enumerate(parts[1:]):
                        if val == "1":
                            active_tools.append(tools_header[i])
                    break # Found the closest frame, exit loop
    except Exception as e:
        print(f"Error reading tool file: {e}")
        
    return active_tools if active_tools else ["None"]

def folder_fetcher(frame_folder, mask_folder, json_path, test_set_ids):
    content = []
    
    with open(json_path, 'r') as f:
        data = json.load(f)
 
    target_ids = [str(id) for id in test_set_ids]
    
    print(f"--- Searching for videos: {target_ids} ---")

    for video_key, meta in data.items():
        yolo_tools = meta.get('tool_labels', [])
        phase = meta.get('phase', "Unknown")

        gt_tools = get_ground_truth_tools(os.path.join(TOOL_ANNOTATION_PATH, f"video{video_key.split('_')[0]}-tool.txt"), int(video_key.split('_')[1]))
        vid_id, frame_id = video_key.split("_")

        image_path = os.path.join(frame_folder, f"video{vid_id}", f"frame_{int(frame_id):05d}.jpg")
        mask_path = os.path.join(mask_folder, f"video{vid_id}", f"frame_{int(frame_id):05d}.jpg")
                        
        if os.path.exists(image_path):
            content.append({
                    "image": image_path,
                    "mask": mask_path,
                    "yolo_tools": yolo_tools,
                    "gt_tools": gt_tools,
                    "phase": phase
                })
        else:
            pass 
    print(f"✅ Successfully mapped {len(content)} frames from JSON to local files.")
    return content

# Run it
oru_list = folder_fetcher(RAW_FRAME_FOLDER_PATH, SEGMENTATION_MASK_FOLDER_PATH, JSON_PATH, [54, 55, 56])
print(oru_list[3], oru_list[256], oru_list[67])