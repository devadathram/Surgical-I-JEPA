import torch
import torch.nn.functional as F
from src.helper import init_model
import cv2
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_surgical_video(model, video_path, output_path="surgical_dashboard.mp4"):
    model.eval()
    device = next(model.parameters()).device
    
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # We will create a dashboard double the width of the input
    out = cv2.VideoWriter(output_path, fourcc, fps, (224 * 2, 224))

    # To smooth the phase predictions
    phase_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Preprocess Frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        
        # Add the 4th dummy channel your weights expect
        dummy = torch.zeros((1, 1, 224, 224)).to(device)
        img_t = torch.cat([img_t, dummy], dim=1)

        with torch.no_grad():
            # 2. Get Phase Prediction
            _, phase_logits = model(img_t)
            probs = F.softmax(phase_logits, dim=-1).cpu().numpy()[0]
            
            # Smooth predictions over last 5 frames
            phase_history.append(probs)
            if len(phase_history) > 5: phase_history.pop(0)
            smoothed_probs = np.mean(phase_history, axis=0)
            
            # 3. Get Attention (Tool Focus)
            # This is your "Zero-Shot" segmentation since we have no masks
            x = model.patch_embed(img_t)
            x = x + model.interpolate_pos_encoding(x, model.pos_embed)
            for blk in model.blocks[:-1]: x = blk(x)
            attn = model.blocks[-1](x, return_attention=True)
            importance = torch.mean(torch.mean(attn[0], dim=0), dim=0).reshape(14, 14).cpu().numpy()
            heatmap = cv2.resize(importance, (224, 224))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_MAGMA)

        # 4. Construct the Dashboard Frame
        # Left: Attention Overlay
        overlay = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)
        
        # Right: Phase Bars (Simple manual drawing)
        bar_panel = np.zeros((224, 224, 3), dtype=np.uint8)
        for i, p in enumerate(smoothed_probs):
            y_pos = 30 + (i * 30)
            bar_w = int(p * 150)
            color = (0, 255, 0) if i == np.argmax(smoothed_probs) else (100, 100, 100)
            cv2.rectangle(bar_panel, (10, y_pos), (10 + bar_w, y_pos + 15), color, -1)
            cv2.putText(bar_panel, f"P{i}", (170, y_pos + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # Combine
        combined = np.hstack((overlay, bar_panel))
        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    cap.release()
    out.release()
    print("Video Dashboard generated successfully!")

encoder, predictor = init_model(
    device=device,
    patch_size=16,
    crop_size=224, # Adjust to your training size
    model_name="vit_base" # Adjust to yours
)
model_path = "/home/devadath/cholec80/OptSurgAI/AEApproach/ijepa/logs/vanilla_test/cholec_constrained_01-ep50.pth.tar"
checkpoint = torch.load(model_path, map_location=device)
encoder.load_state_dict(checkpoint['encoder'])
predictor.load_state_dict(checkpoint['predictor'])

video_path = "/home/devadath/cholec80/OptSurgAI/sample video/vlc-record-2025-11-20-15h00m12s-7013HD Bin.01-H264-480-.mp4"
output_path = "results_march/surgical_dashboard.mp4"

process_surgical_video(encoder, video_path, output_path=output_path)