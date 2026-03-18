import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
# Use on_bad_lines='skip' because the log file has mixed formats
df = pd.read_csv('/home/devadath/cholec80/OptSurgAI/AEApproach/ijepa/logs/vanilla_test/cholec_phase_01_r0.csv', on_bad_lines='skip')

# 2. Clean the data
# Remove rows that accidentally repeat the column names
df = df[df['epoch'] != 'epoch']

# Convert columns to numeric values
df['epoch'] = pd.to_numeric(df['epoch'])
df['loss'] = pd.to_numeric(df['loss'])

# 3. Aggregate data
# Since there are many iterations (itr) per epoch, we take the average loss per epoch
epoch_loss = df.groupby('epoch')['loss'].mean().reset_index()

# 4. Create the plot
plt.figure(figsize=(10, 5))
plt.plot(epoch_loss['epoch'], epoch_loss['loss'], marker='o', color='#2c3e50', linewidth=2)

# Styling
plt.title('Training Progress: Loss vs Epoch', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Average Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Save or show
plt.savefig('loss_trend_with_phase.png')
plt.show()