import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_from_csv(csv_path="/home/devadath/cholec80/OptSurgAI/AEApproach/ijepa/logs/vanilla_test/cholec_constrained_01_r0.csv"):
    if not os.path.exists(csv_path):
        print(f"🚨 Error: {csv_path} not found.")
        return

    # 1. Load data
    df = pd.read_csv(csv_path)

    # 2. Cleanup: Convert to numeric and drop any corrupted rows (like repeated headers)
    for col in ['epoch', 'loss', 'acc']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['epoch', 'loss'])

    # 3. AGGREGATION: This is the magic step.
    # We collapse all iterations (itr) into one mean value per epoch.
    epoch_grouped = df.groupby('epoch').agg({
        'loss': 'mean',
        'acc': 'mean',
        'time (ms)': 'mean'
    }).reset_index()

    # 4. PLOTTING
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Average Loss (Left Axis)
    color_loss = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Mean Loss', color=color_loss, fontsize=12, fontweight='bold')
    ax1.plot(epoch_grouped['epoch'], epoch_grouped['loss'], 
             marker='o', linestyle='-', color=color_loss, linewidth=2, label='Avg Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Accuracy (Right Axis) - Useful if you are doing Phase Recognition
    if 'acc' in epoch_grouped.columns and epoch_grouped['acc'].max() > 0:
        ax2 = ax1.twinx()
        color_acc = 'tab:green'
        ax2.set_ylabel('Mean Accuracy', color=color_acc, fontsize=12, fontweight='bold')
        ax2.plot(epoch_grouped['epoch'], epoch_grouped['acc'], 
                 marker='s', linestyle='--', color=color_acc, alpha=0.7, label='Avg Acc')
        ax2.tick_params(axis='y', labelcolor=color_acc)

    plt.title('Surgical I-JEPA: Training Progress (Epoch Averages)', fontsize=14)
    fig.tight_layout()
    
    plt.savefig('epoch_loss_cleaned.png', dpi=300)
    print(f"✅ Plot saved! Grouped {len(df)} iterations into {len(epoch_grouped)} clean epochs.")

if __name__ == "__main__":
    plot_from_csv()