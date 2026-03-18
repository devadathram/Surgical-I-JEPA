import os
import pickle
from collections import Counter

label_file = '/home/devadath/cholec80/OptSurgAI/AEApproach/ijepa/ch80_labels/labels/train/1fps_100_0.pickle' 
frames_root = '/home/devadath/cholec80/frames/'

with open(label_file, 'rb') as f:
    data = pickle.load(f)

# 1. Map video_id to its actual path (handles the '000001-010000' top layer)
video_path_map = {}
for root, dirs, files in os.walk(frames_root):
    for d in dirs:
        if d.startswith('video'):
            video_path_map[d] = os.path.join(root, d)

available_frames = []

# 2. Match pickle data using the NEW filename format
for video_id, frames in data.items():
    if video_id in video_path_map:
        video_dir = video_path_map[video_id]
        
        for f_info in frames:
            f_id = f_info['Frame_id']
            
            # Use 1-based indexing for the filename if frame_00000 doesn't exist
            # Cholec80 raw frames often start at 1.
            actual_id = f_id if f_id > 0 else 1 
            
            # Logic for range folder (e.g., 250 -> 00001-10000)
            lower = ((actual_id - 1) // 10000) * 10000 + 1
            upper = ((actual_id - 1) // 10000 + 1) * 10000
            range_folder = f"{lower:05d}-{upper:05d}"
            
            # The corrected filename: frame_XXXXX.jpg
            img_name = f"frame_{actual_id:05d}.jpg"
            full_path = os.path.join(video_dir, range_folder, img_name)
            
            if os.path.exists(full_path):
                available_frames.append(f_info['Phase_gt'])

# 3. Final Analysis
if len(available_frames) > 0:
    phase_counts = Counter(available_frames)
    print(f"✅ BINGO! Ready to train with {len(available_frames)} frames.")
    total = len(available_frames)
    # 7 is the number of phases
    new_weights = [total / (7 * phase_counts.get(i, 1)) for i in range(7)]
    print("\n🔥 USE THESE WEIGHTS FOR YOUR SAMPLER:")
    print([round(w, 4) for w in new_weights])
else:
    print("❌ Still 0. Let's check the very first frame path:")
    test_path = os.path.join(video_path_map['video59'], '00001-10000', 'frame_00001.jpg')
    print(f"Checking for: {test_path}")
    print(f"Does it exist? {os.path.exists(test_path)}")