import os
import torch
from torchvision.datasets.folder import has_file_allowed_extension
from embedding_generator import ResNetEmbedding  # assuming you saved the class in resnet_embedding.py

# Setup
image_folder = "../../reference_dataset"
embedding_folder = "embeddings"
os.makedirs(embedding_folder, exist_ok=True)

# Allowed image extensions
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Load model
model = ResNetEmbedding()
model.load_state_dict(torch.load("best_model_epoch_27.pth")['model_state_dict'])

# Iterate through all images
for filename in os.listdir(image_folder):
    if has_file_allowed_extension(filename, IMG_EXTENSIONS):
        image_path = os.path.join(image_folder, filename)
        try:
            # Generate embedding
            embedding = model.get_embedding_from_path(image_path)  # shape: (512,)

            # Save with same name (change extension to .pt)
            name_without_ext = os.path.splitext(filename)[0]
            save_path = os.path.join(embedding_folder, f"{name_without_ext}.pt")
            torch.save(embedding, save_path)

            print(f"Saved embedding for {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
