import os
import shutil
from tqdm import tqdm

def create_1fps_subset(frames_root, subset_root, sampling_rate=25):
    src_file_ls = []
    if not os.path.exists(subset_root):
        os.makedirs(subset_root)

    video_folders = [d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))]
    
    print(f"Found {len(video_folders)} videos in source.")

    for video_id in tqdm(video_folders, desc="Overall Progress"):
        src_video_path = os.path.join(frames_root, video_id)
        dst_video_path = os.path.join(subset_root, video_id)
        src_file_ls.append(src_video_path)

        # RESUME LOGIC: Skip if video folder already exists in subset
        if os.path.exists(dst_video_path) and len(os.listdir(dst_video_path)) > 0:
            print(f"⏭️ Skipping {video_id}: Already processed.")
            continue

        os.makedirs(dst_video_path, exist_ok=True)

        # 2. Traverse range folders (e.g., 000001-010000)
        range_folders = sorted([d for d in os.listdir(src_video_path) 
                               if os.path.isdir(os.path.join(src_video_path, d))])

        for rf in range_folders:
            rf_path = os.path.join(src_video_path, rf)
            
            # 3. Get all frames in this range
            frames = sorted([f for f in os.listdir(rf_path) if f.endswith(".jpg")])
            
            for f_name in frames:
                # Extract the number from 'frame_00001.jpg'
                try:
                    # Splitting 'frame_00001.jpg' -> '00001'
                    frame_num = int(f_name.split('_')[1].split('.')[0])
                except (IndexError, ValueError):
                    continue

                # 4. SAMPLING LOGIC: Only keep every 25th frame
                # (frame_num % 25 == 1) ensures we get 1, 26, 51, etc.
                if frame_num % sampling_rate == 0:
                    src_file = os.path.join(rf_path, f_name)
                    dst_file = os.path.join(dst_video_path, f_name)

                    #print(f"Copying {src_file} to {dst_file}")
                    # Copy without nested range folders
                    shutil.copy2(src_file, dst_file)

    print(f"\n✅ Subset creation complete in: {subset_root}")
    print(f"Total source files collected: {len(src_file_ls)}")

# --- CONFIGURATION ---
create_1fps_subset(
    frames_root="/home/devadath/cholec80/frames",
    subset_root="/home/devadath/cholec80/OptSurgAI/AEApproach/Subset_folder",
    sampling_rate=25 # Set to 1 if you want to copy EVERYTHING flattened
)