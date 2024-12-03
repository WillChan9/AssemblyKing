import os
import random
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# Define paths
dataset_dir = 'dataset'
image_dir = os.path.join(dataset_dir, 'images')
label_dir = os.path.join(dataset_dir, 'labels')

# Create lists of image and label files
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # Ensure proper ordering

# Shuffle the dataset
random.seed(42)  # For reproducibility
random.shuffle(image_files)

# Split into training and validation sets (e.g., 80% training, 20% validation)
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# Function to copy images and labels to designated directories
def copy_files(file_list, src_image_dir, src_label_dir, dest_image_dir, dest_label_dir):
    os.makedirs(dest_image_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)
    for file_name in file_list:
        # Copy image file
        src_image_path = os.path.join(src_image_dir, file_name)
        dest_image_path = os.path.join(dest_image_dir, file_name)
        shutil.copy(src_image_path, dest_image_path)
        
        # Copy corresponding label file
        label_name = os.path.splitext(file_name)[0] + '.txt'
        src_label_path = os.path.join(src_label_dir, label_name)
        dest_label_path = os.path.join(dest_label_dir, label_name)
        shutil.copy(src_label_path, dest_label_path)

# Define destination directories for training and validation sets
train_image_dir = os.path.join(dataset_dir, 'images/train')
train_label_dir = os.path.join(dataset_dir, 'labels/train')
val_image_dir = os.path.join(dataset_dir, 'images/val')
val_label_dir = os.path.join(dataset_dir, 'labels/val')

# Copy files to the respective directories
copy_files(train_files, image_dir, label_dir, train_image_dir, train_label_dir)
copy_files(val_files, image_dir, label_dir, val_image_dir, val_label_dir)

# Create the data.yaml file for YOLO configuration
data_yaml_content = f"""
train: {os.path.abspath(train_image_dir)}
val: {os.path.abspath(val_image_dir)}

nc: 1
names: ['Part1']
"""

with open('data.yaml', 'w') as f:
    f.write(data_yaml_content.strip())

# Initialize the YOLO model
model = YOLO('yolov8n.pt')  # Using the smallest model for speed

# Training parameters for quick test
model.train(
    data='data.yaml',     # Path to the data configuration file
    epochs=20,             # Number of training epochs
    imgsz=640,            # Smaller image size for faster training
    batch=16,              # Smaller batch size due to limited data
    name='test1',    # Name of the training run
    verbose=False         # Reduce verbosity for quick test
)

# Optionally, evaluate the model
metrics = model.val()
print(metrics)