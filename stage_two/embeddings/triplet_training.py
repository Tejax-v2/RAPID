import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import random
import os
from tqdm import tqdm
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for matplotlib 
import matplotlib.pyplot as plt

# 1. Extract detected objects from YOLO results
def extract_crops(results, min_size=32):
    """Extract crops with minimum size filtering"""
    all_crops = []
    for result in results:
        orig_img = Image.fromarray(result.orig_img[..., ::-1])  # Convert BGR to RGB
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # Skip small crops that might cause issues
            if (x2 - x1) > min_size and (y2 - y1) > min_size:
                crop = orig_img.crop((x1, y1, x2, y2))
                all_crops.append(crop)
    return all_crops

# 2. Enhanced Triplet Dataset with visualization
class TripletDataset(Dataset):
    def __init__(self, crops, transform=None, augment=None, min_crops=100):
        self.crops = crops
        # Ensure we have enough crops for meaningful training
        if len(self.crops) < min_crops:
            raise ValueError(f"Need at least {min_crops} crops, got {len(self.crops)}")
        self.transform = transform
        self.augment = augment
        self.min_crops = min_crops

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        anchor = self.crops[idx]
        
        # Generate positive - use augmentation with 80% probability
        if random.random() < 0.8 and self.augment:
            positive = self.augment(anchor)
        else:
            positive = anchor
        
        # Find a hard negative - different but similar looking crop
        attempts = 0
        while attempts < 10:  # Try to find a good negative
            negative_idx = random.randint(0, len(self.crops)-1)
            if negative_idx != idx:
                negative = self.crops[negative_idx]
                # Basic similarity check (could be enhanced)
                if anchor.size != negative.size:
                    break
                if abs(np.mean(np.array(anchor)) - np.mean(np.array(negative))) > 20:
                    break
            attempts += 1
            
        # Apply transforms
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
            
        return anchor, positive, negative

    def visualize_triplet(self, idx, save_path=None):
        """Visualize and save a sample triplet"""
        anchor, positive, negative = self.__getitem__(idx)
        
        # Convert tensors back to images if transformed
        if self.transform:
            anchor = self._reverse_transform(anchor)
            positive = self._reverse_transform(positive)
            negative = self._reverse_transform(negative)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(anchor)
        axes[0].set_title('Anchor')
        axes[1].imshow(positive)
        axes[1].set_title('Positive')
        axes[2].imshow(negative)
        axes[2].set_title('Negative')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _reverse_transform(self, tensor):
        """Reverse transform for visualization"""
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        tensor = inv_normalize(tensor)
        tensor = torch.clamp(tensor, 0, 1)
        return transforms.ToPILImage()(tensor.cpu())

# 3. Modified ResNet18 with L2 normalization
# class ResNetEmbedding(nn.Module):
#     def __init__(self, embedding_size=512):
#         super().__init__()
#         self.resnet = resnet18(pretrained=True)
#         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_size)
#         # L2 normalization layer
#         self.normalize = nn.functional.normalize

#     def forward(self, x):
#         x = self.resnet(x)
#         return self.normalize(x, p=2, dim=1)

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ResNetEmbedding(nn.Module):
    def __init__(self):
        super(ResNetEmbedding, self).__init__()
        # Load pretrained ResNet18 and chop off the classifier (fc) layer
        resnet18 = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet18.children())[:-1])
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize entire image to fixed size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        # self.eval()

    def forward(self, x):
        """
        Forward pass for a tensor input (already preprocessed).
        x: torch.Tensor of shape (B, 3, 224, 224)
        Returns: torch.Tensor of shape (B, 512)
        """
        features = self.feature_extractor(x)  # (B, 512, 1, 1)
        embeddings = features.view(features.size(0), -1)  # Flatten to (B, 512)
        return embeddings

    def get_embedding_from_path(self, image_path):
        """
        Accepts image path, handles preprocessing and returns embedding
        """
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        return self.forward(img_tensor).squeeze(0)  # Return (512,) tensor


# 4. Enhanced training configuration
def train_triplet_network(crops, num_epochs=20, batch_size=64, margin=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup TensorBoard with more organized logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/triplet_train_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)
    
    # Enhanced transforms with more augmentations
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Slightly larger for better quality
        transforms.RandomCrop(224),  # Random crop for better generalization
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    augmentation = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.GaussianBlur(3),
        ], p=0.7),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
    ])
    
    # Create dataset and save sample triplets
    dataset = TripletDataset(crops, transform=base_transform, augment=augmentation)
    os.makedirs('sample_triplets', exist_ok=True)
    for i in range(5):  # Save 5 sample triplets
        dataset.visualize_triplet(
            idx=random.randint(0, len(dataset)-1),
            save_path=f'sample_triplets/triplet_sample_{i}.png'
        )
    
    # Create DataLoader with balanced sampling
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Model with L2 normalization
    model = ResNetEmbedding().to(device)  # Smaller embedding size
    
    # Loss function with squared distance option
    criterion = nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
    
    # Enhanced optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Training loop with enhanced monitoring
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (anchor, positive, negative) in enumerate(progress_bar):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Forward pass
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            # Compute loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update tracking
            running_loss += loss.item()
            global_step += 1
            
            # Log batch metrics
            if global_step % 10 == 0:
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                
                # Calculate and log distances
                pos_dist = torch.mean(torch.norm(anchor_emb - positive_emb, p=2, dim=1))
                neg_dist = torch.mean(torch.norm(anchor_emb - negative_emb, p=2, dim=1))
                writer.add_scalar('Distance/positive', pos_dist.item(), global_step)
                writer.add_scalar('Distance/negative', neg_dist.item(), global_step)
                writer.add_scalar('Margin/actual', (neg_dist - pos_dist).item(), global_step)
            
            progress_bar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Save sample batch visualization every 100 batches
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    grid = make_grid(torch.cat([anchor[:4], positive[:4], negative[:4]]))
                    writer.add_image('Sample Triplets', grid, global_step)
        
        # Epoch statistics
        epoch_loss = running_loss / len(dataloader)
        writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Update learning rate
        scheduler.step(epoch_loss)
        
        # Save checkpoints
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f'best_model_epoch_{epoch}.pth')
        
        print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}, LR={optimizer.param_groups[0]["lr"]:.2e}')
    
    writer.close()
    return model

# Usage flow
if __name__ == "__main__":
    # 1. Run YOLO detection
    from ultralytics import YOLO
    model = YOLO('../../stage_one/trained_yolov10.pt')
    results = model.predict('../../images', conf=0.5)  # Add confidence threshold
    
    # 2. Extract crops with size filtering
    crops = extract_crops(results, min_size=64)
    print(f"Extracted {len(crops)} valid crops for training")
    
    # 3. Train with enhanced parameters
    trained_model = train_triplet_network(
        crops,
        num_epochs=30,
        batch_size=64,
        margin=0.3  # Smaller margin for more challenging learning
    )
    
    # 4. Save final model
    torch.save(trained_model.state_dict(), 'final_embedding_model.pth')