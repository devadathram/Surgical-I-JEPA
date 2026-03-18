import os
import pickle

# Same paths as before
label_file = '/home/devadath/cholec80/OptSurgAI/AEApproach/ijepa/ch80_labels/labels/train/1fps_100_0.pickle' 
frames_root = '/home/devadath/cholec80/frames/'
output_subset_txt = 'cholec_subset_15k.txt'

with open(label_file, 'rb') as f:
    data = pickle.load(f)

# Locate video folders
video_path_map = {}
for root, dirs, _ in os.walk(frames_root):
    for d in dirs:
        if d.startswith('video'):
            video_path_map[d] = os.path.join(root, d)

with open(output_subset_txt, 'w') as f_out:
    for video_id, frames in data.items():
        if video_id in video_path_map:
            v_dir = video_path_map[video_id]
            for f_info in frames:
                f_id = f_info['Frame_id']
                actual_id = f_id if f_id > 0 else 1
                
                # Range folder logic
                lower = ((actual_id - 1) // 10000) * 10000 + 1
                upper = ((actual_id - 1) // 10000 + 1) * 10000
                range_folder = f"{lower:05d}-{upper:05d}"
                
                # Build the RELATIVE path (from the root_path in your YAML)
                # If root_path is /home/devadath/cholec80/frames/, the subset file
                # needs the path starting from the video folders.
                rel_path = os.path.join(video_id, range_folder, f"frame_{actual_id:05d}.jpg")
                full_path = os.path.join(v_dir, range_folder, f"frame_{actual_id:05d}.jpg")
                
                if os.path.exists(full_path):
                    f_out.write(f"{rel_path}\n")

print(f" Subset file created: {output_subset_txt}")