import os
from inference import get_model
import supervision as sv
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the image file
image_file = "photos/shelf9.jpg"
image = cv2.imread(image_file)

# Load a pre-trained YOLOv8 model
model = get_model(model_id="sku-110k/2", api_key="API_KEY")
print(model)

# Run inference on the image
results = model.infer(image)[0]

# Load the results into the supervision Detections API
detections = sv.Detections.from_inference(results)
#print(detections)

# Load pre-trained ResNet-18 model
resnet50 = models.resnet50(pretrained=True).to(device)
resnet50.eval()

# Define preprocessing transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Extract embeddings
embeddings = []
for bbox in detections.xyxy:
    x1, y1, x2, y2 = map(int, bbox)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        continue
    crop_tensor = transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet50(crop_tensor).flatten().cpu().numpy()
    embeddings.append(embedding)

embeddings = np.array(embeddings)
# print(embeddings)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=30.0, min_samples=2)
labels = dbscan.fit_predict(embeddings)
#print(labels)

# Assign cluster labels to detections, filtering out noise points (-1)
# Assign noise points (-1) a unique class ID (e.g., 999)
labels[labels == -1] = 999  # Assign all noise detections to class 999
detections.class_id = labels  # Apply modified labels to detections

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster Label')
plt.title("t-SNE Visualization of Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()


unique_labels = set(labels) - {999}  # Exclude noise class
if len(unique_labels) > 1:
    score = silhouette_score(embeddings, labels)
    print(f"Silhouette Score: {score}")
else:
    print("Silhouette score cannot be calculated with less than 2 clusters.")

# Define colors for annotation
# color_annotator = sv.ColorAnnotator()
box_annotator = sv.BoxAnnotator()
annotated_image = box_annotator.annotate(scene=image, detections=detections)
# label_annotator = sv.LabelAnnotator()
# annotated_image = label_annotator.annotate(scene=image, detections=detections, labels=[str(x) for x in detections.class_id])
# print(detections)

# Save and show annotated image
cv2.imwrite("annotated_image.jpg", annotated_image)
cv2.imshow("Annotated Image", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()