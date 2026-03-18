import pickle

file_path = '/home/devadath/cholec80/OptSurgAI/AEApproach/ijepa/ch80_labels/labels/val/1fps.pickle' # Replace with your actual path
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Check the type and a sample
print(f"Data type: {type(data)}")
if isinstance(data, list):
    print("First entry sample:", data[0])
elif isinstance(data, dict):
    print("Keys found:", data.keys())

sample_video = 'video42'
print(f"Sample from {sample_video}:")
print(data[sample_video][:3])