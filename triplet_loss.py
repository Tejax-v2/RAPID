import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import albumentations as A
from sklearn.metrics import pairwise_distances
import random
import os

# Define the CNN feature extractor (simple CNN for this task)
class FeatureExtractorCNN(nn.Module):
    def __init__(self):
        super(FeatureExtractorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)  # assuming 224x224 input size after augmentations
        self.fc2 = nn.Linear(512, 128)  # output feature vector dimension

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data augmentation pipeline
augmentation_pipeline = A.Compose([
    A.Resize(224, 224),  # <-- resize instead of crop
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomRotate90(p=0.5),
    A.Blur(blur_limit=3, p=0.2),
    A.Sharpen(p=0.2),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),
    A.RandomGamma(p=0.2),
    A.CLAHE(p=0.2),
])


# Dataset for loading crops and augmenting images
class YOLODataset(Dataset):
    def __init__(self, image_folder, label_folder, augmentations=None):
        """
        Args:
            image_folder (str): Path to the folder containing images.
            label_folder (str): Path to the folder containing YOLO annotation labels.
            augmentations (albumentations.Compose): Augmentation pipeline to apply on the images and crops.
        """
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.augmentations = augmentations

        # Get the list of image files (assuming all images have a '.jpg' extension)
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        label_path = os.path.join(self.label_folder, image_name.replace('.jpg', '.txt'))

        # Read the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Read the label file
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # Get bounding box coordinates from labels
        crops = []
        for label in labels:
            # Extract class index and bounding box coordinates (normalized)
            parts = label.strip().split()
            class_idx = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])

            # Convert YOLO format to pixel coordinates (assuming img size is known)
            img_height, img_width = img.shape[:2]
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)

            # Crop the image based on the bounding box
            crop = img[y1:y2, x1:x2]
            crops.append(crop)

        # Apply augmentation if available
        if self.augmentations:
            augmented_crops = [self.augmentations(image=crop)["image"] for crop in crops]
        else:
            augmented_crops = crops

        return augmented_crops

# Set paths to your dataset
image_folder = '/DATA/tejas_2101cs78/datasets/SKU-110K/images'
label_folder = '/DATA/tejas_2101cs78/datasets/SKU-110K/labels'

# Triplet loss function
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.pairwise_distance(anchor, positive, p=2)
        neg_dist = torch.pairwise_distance(anchor, negative, p=2)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()

# Training loop
def train_triplet_model(model, dataloader, loss_fn, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, crops in enumerate(dataloader):
            # Randomly select anchor, positive, and negative samples
            anchor_idx = random.randint(0, len(crops) - 1)
            positive_idx = (anchor_idx + 1) % len(crops)  # Next crop as positive example
            negative_idx = random.randint(0, len(crops) - 1)  # Random crop from another image

            anchor_crop = crops[anchor_idx]
            positive_crop = crops[positive_idx]
            negative_crop = crops[negative_idx]

            # Convert to tensor
            anchor_tensor = torch.tensor(anchor_crop).float().permute(2, 0, 1).unsqueeze(0)
            positive_tensor = torch.tensor(positive_crop).float().permute(2, 0, 1).unsqueeze(0)
            negative_tensor = torch.tensor(negative_crop).float().permute(2, 0, 1).unsqueeze(0)

            # Forward pass
            anchor_features = model(anchor_tensor)
            positive_features = model(positive_tensor)
            negative_features = model(negative_tensor)

            # Calculate loss
            loss = loss_fn(anchor_features, positive_features, negative_features)
            running_loss += loss.item()

            # Backprop and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}')


# Prepare dataset and dataloader
dataset = YOLODataset(image_folder, label_folder, augmentations=augmentation_pipeline)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize the CNN, loss function, and optimizer
model = FeatureExtractorCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = TripletLoss()

# Train the model
train_triplet_model(model, dataloader, loss_fn, optimizer)
