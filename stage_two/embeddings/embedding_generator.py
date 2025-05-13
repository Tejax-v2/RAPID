import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ResNetEmbedding(nn.Module):
    def __init__(self):
        super(ResNetEmbedding, self).__init__()
        # Load pretrained ResNet18 and chop off the classifier (fc) layer
        resnet18 = models.resnet18()
        self.feature_extractor = nn.Sequential(*list(resnet18.children())[:-1])
        self.preprocess = transforms.Compose([
            transforms.Resize(256),        # Resize smaller edge to 256
            transforms.CenterCrop(224),    # Center crop to 224x224
            transforms.ToTensor(),
            transforms.Normalize(          # Normalize using ImageNet stats
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.eval()

    def forward(self, x):
        """
        Forward pass for a tensor input (already preprocessed).
        x: torch.Tensor of shape (B, 3, 224, 224)
        Returns: torch.Tensor of shape (B, 512)
        """
        with torch.no_grad():
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
