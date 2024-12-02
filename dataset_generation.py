import cv2
import os
import random
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from ..sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

def extract_and_annotate_frames(video_path, output_dir, frame_interval=1, train_split=0.8):
    """
    Extract frames from a video file, automatically annotate them, and split into training and validation sets.

    Parameters:
    - video_path: Path to the input video file.
    - output_dir: Base directory where images and labels will be saved.
    - frame_interval: Number of frames to skip between saves (default 1 saves every frame).
    - train_split: Proportion of images to include in the training set.
    """

    # Create directory structure
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')

    for split in ['train', 'val']:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

    # Initialize video capture and background subtractor
    vidcap = cv2.VideoCapture(video_path)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    
    frame_count = 0
    saved_frames = []
    success = True

    while success:
        success, frame = vidcap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            # Apply background subtraction
            fgMask = backSub.apply(frame)

            # Remove shadows (pixel value 127)
            _, fgMask = cv2.threshold(fgMask, 250, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Assume the largest contour is the object
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Save the frame
                frame_filename = f"frame_{frame_count:05d}.jpg"
                frame_path = os.path.join(images_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_frames.append(frame_filename)

                # Generate YOLO format annotation
                img_height, img_width, _ = frame.shape
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height

                label_filename = frame_filename.replace('.jpg', '.txt')
                label_path = os.path.join(labels_dir, label_filename)

                # Assuming class id is 0
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")
            else:
                print(f"No object detected in frame {frame_count}")
        frame_count += 1

    vidcap.release()
    print(f"Extracted and annotated {len(saved_frames)} frames.")

    # Shuffle and split frames into train and val
    random.shuffle(saved_frames)
    split_index = int(len(saved_frames) * train_split)
    train_frames = saved_frames[:split_index]
    val_frames = saved_frames[split_index:]

    # Move frames and labels to train and val directories
    for frame_filename in train_frames:
        # Move image
        src = os.path.join(images_dir, frame_filename)
        dst = os.path.join(images_dir, 'train', frame_filename)
        shutil.move(src, dst)
        # Move label
        label_filename = frame_filename.replace('.jpg', '.txt')
        src_label = os.path.join(labels_dir, label_filename)
        dst_label = os.path.join(labels_dir, 'train', label_filename)
        shutil.move(src_label, dst_label)

    for frame_filename in val_frames:
        # Move image
        src = os.path.join(images_dir, frame_filename)
        dst = os.path.join(images_dir, 'val', frame_filename)
        shutil.move(src, dst)
        # Move label
        label_filename = frame_filename.replace('.jpg', '.txt')
        src_label = os.path.join(labels_dir, label_filename)
        dst_label = os.path.join(labels_dir, 'val', label_filename)
        shutil.move(src_label, dst_label)

    print(f"Organized frames into {len(train_frames)} training and {len(val_frames)} validation images.")

def visualize_annotations(image_dir, label_dir, output_dir):
    """
    Draw bounding boxes on images based on YOLO annotations and save them to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(label_dir, label_file)

        # Read the image
        image = cv2.imread(image_path)
        img_height, img_width, _ = image.shape

        # Read the annotation if it exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                # Convert normalized coordinates back to pixel values
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height

                # Calculate the top-left and bottom-right coordinates
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                # Draw the bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Optionally, put class name or ID
                cv2.putText(image, str(int(class_id)), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Save the image with bounding boxes
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, image)

# Example usage
# video_path = 'object_videos/IMG_2577.MOV'
# output_dir = 'dataset'
# extract_and_annotate_frames(video_path, output_dir, frame_interval=1, train_split=0.8)
# Example usage
image_dir = 'dataset/images/train'  # or 'dataset/images/val'
label_dir = 'dataset/labels/train'  # or 'dataset/labels/val'
output_dir = 'visualizations/train'  # or 'visualizations/val'

visualize_annotations(image_dir, label_dir, output_dir)