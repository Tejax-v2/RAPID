import os
import torch
from ultralytics import YOLO
from torchvision import models, transforms
import torchvision
from PIL import Image
import numpy as np
import hdbscan
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import supervision as sv
import cv2

# ========== CONFIG ========== #
image_path = "/DATA/tejas_2101cs78/datasets/SKU-110K/images/test_1.jpg"        # Your input image
save_crops_dir = "crops2"        # Where to save cropped images
yolo_weights = "yolo/runs/detect/train/weights/best.pt"        # Your YOLO trained weights
device = 'cuda' if torch.cuda.is_available() else 'cpu'
image = cv2.imread(image_path)

# ========== STEP 1: Load YOLO Model and Detect ========== #
model = YOLO(yolo_weights)
results = model(image_path)[0]
detections = sv.Detections.from_ultralytics(results)

# Make sure crops directory exists
os.makedirs(save_crops_dir, exist_ok=True)

# ========== STEP 2: Crop and Save Detections ========== #
cropped_images = []
coordinates = []  # To store normalized coordinates of centers
image_width, image_height = Image.open(image_path).size  # Get image dimensions

for i, box in enumerate(results.boxes.xyxy.cpu().numpy()):
    x1, y1, x2, y2 = map(int, box)
    
    # Crop the image
    img = Image.open(image_path)
    cropped = img.crop((x1, y1, x2, y2))
    crop_path = os.path.join(save_crops_dir, f"crop_{i}.jpg")
    cropped.save(crop_path)
    cropped_images.append(crop_path)
    
    # Calculate normalized center coordinates (x_center, y_center)
    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height
    coordinates.append([x_center, y_center])

print(f"[INFO] Saved {len(cropped_images)} cropped images.")

# ========== STEP 3: Create Embeddings using ResNet50 ========== #
# Load Pretrained ResNet50
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()  # Remove final classification layer
resnet = resnet.to(device)
resnet.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

embeddings = []

with torch.no_grad():
    for img_path in cropped_images:
        img = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        embedding = resnet(input_tensor)
        
        # Flatten the embedding and append normalized coordinates
        embedding = embedding.cpu().numpy().flatten()
        embedding_with_coordinates = np.concatenate([embedding, coordinates[len(embeddings)]])
        embeddings.append(embedding_with_coordinates)

embeddings = np.array(embeddings)

print(f"[INFO] Generated embeddings with shape: {embeddings.shape}")

# ========== STEP 4: HDBSCAN Clustering ========== #
clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
cluster_labels = clusterer.fit_predict(embeddings)

# Exclude outliers (-1) for silhouette score calculation
valid_labels = cluster_labels != -1
valid_embeddings = embeddings[valid_labels]
valid_cluster_labels = cluster_labels[valid_labels]

# Calculate Silhouette Score
if len(set(valid_cluster_labels)) > 1:  # Silhouette score requires at least 2 clusters
    score = silhouette_score(valid_embeddings, valid_cluster_labels)
    print(f"[INFO] Silhouette Score: {score:.3f}")
else:
    print("[INFO] Silhouette Score: Cannot calculate with only one cluster.")

print(f"[INFO] Cluster labels: {cluster_labels}")

# ========== STEP 5: t-SNE Visualization ========== #
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings_2d[:, 0], 
    embeddings_2d[:, 1], 
    c=cluster_labels, 
    cmap='tab10', 
    s=50
)
plt.title("t-SNE Visualization of Clusters")
plt.colorbar(scatter)
plt.savefig("tsne_clusters_2.png")
print("[INFO] Saved t-SNE plot as 'tsne_clusters_2.png'")

# ========== STEP 6: Annotate Original Image with Detections and Cluster Colors ========== #
# Create a color palette - one unique color per cluster
num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
colors = plt.cm.get_cmap('tab10', num_clusters)

# Copy the detections object
detections_with_labels = detections

# Update detections with cluster labels
detections_with_labels.class_id = cluster_labels  # Just storing cluster IDs as class IDs for coloring

# Create bounding box annotator
bounding_box_annotator = sv.BoxAnnotator()

# Prepare annotations manually
annotated_image = image.copy()

for idx, (xyxy, cluster_id) in enumerate(zip(detections_with_labels.xyxy, cluster_labels)):
    if cluster_id == -1:
        color = (128, 128, 128)  # Grey color for outliers
        label = "Outlier"
    else:
        rgb_color = colors(cluster_id)[:3]
        color = tuple(int(c * 255) for c in rgb_color)  # Convert to 0-255 range
        label = f"Cluster {cluster_id}"

    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness=6)
    cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

# Save the annotated image
save_path = "annotated_image_with_clusters.jpg"
cv2.imwrite(save_path, annotated_image)
print(f"Annotated clustered image saved at: {save_path}")