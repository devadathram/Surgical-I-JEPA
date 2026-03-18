import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

def check_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False

def run_audit(root_path):
    # Dictionary to store { video_name: [total, corrupt] }
    stats = defaultdict(lambda: [0, 0])
    all_files = []

    print("Gathering file list...")
    for root, _, files in os.walk(root_path):
        # Assumes folder structure: .../frames/videoXX/...
        parts = root.split(os.sep)
        # Find the part that looks like 'video01', 'video02', etc.
        video_name = next((p for p in parts if p.startswith('video')), "unknown")
        
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, f)
                all_files.append((path, video_name))
                stats[video_name][0] += 1

    print(f"Auditing {len(all_files)} files across {len(stats)} folders...")
    
    def worker(file_info):
        path, v_name = file_info
        is_valid = check_image(path)
        return v_name, is_valid

    with ThreadPoolExecutor(max_workers=24) as executor:
        results = list(executor.map(worker, all_files))

    for v_name, is_valid in results:
        if not is_valid:
            stats[v_name][1] += 1

    print(f"\n{'Video ID':<15} | {'Total Frames':<15} | {'Corrupt':<10} | {'Error %':<10}")
    print("-" * 60)
    
    problematic_videos = []
    for v_name in sorted(stats.keys()):
        total, corrupt = stats[v_name]
        error_pct = (corrupt / total * 100) if total > 0 else 0
        print(f"{v_name:<15} | {total:<15} | {corrupt:<10} | {error_pct:.2f}%")
        
        if error_pct > 5.0: # Threshold for "problematic"
            problematic_videos.append(v_name)

    print("-" * 60)
    print(f"Suggested for Discarding (>5% error): {problematic_videos}")

if __name__ == "__main__":
    run_audit('/home/devadath/cholec80/frames')