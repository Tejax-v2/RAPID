import os
from inference import get_model
import supervision as sv
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torchvision.transforms as transforms
from torchvision import models
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
#model.to(device)  # Move the YOLO model to GPU


# Run inference on the image
results = model.infer(image)[0]

# Load the results into the supervision Detections API
detections = sv.Detections.from_inference(results)

# Define the Embedder network (Pre-trained CNN for feature extraction)
embedder = models.resnet50(pretrained=True)
embedder.fc = torch.nn.Identity()
embedder = embedder.to(device)  # Move model to GPU
embedder.eval()

# Define transformation pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load reference descriptors (Pre-computed features of known products)
reference_images = {
    "product_1": "reference_images/product_1.jpg",
    "product_2": "reference_images/product_2.jpg",
    "product_3": "reference_images/product_3.jpg",
    "product_4": "reference_images/product_4.jpg",
    "product_5": "reference_images/product_5.jpg",
    "product_6": "reference_images/product_6.jpg",
    "product_7": "reference_images/product_7.jpg",
    "product_8": "reference_images/product_8.jpg",
    "product_9": "reference_images/product_9.jpg",
    "product_10": "reference_images/product_10.jpg",
    "product_11": "reference_images/product_11.jpg",
    "product_12": "reference_images/product_12.jpg",
    "product_13": "reference_images/product_13.jpg",
    "product_14": "reference_images/product_14.jpg",
    "product_15": "reference_images/product_15.jpg",
    "product_16": "reference_images/product_16.jpg",
    "product_17": "reference_images/product_17.jpg",
    "product_18": "reference_images/product_18.jpg",
    "product_19": "reference_images/product_19.jpg",
    "product_20": "reference_images/product_20.jpg",
    "product_21": "reference_images/product_21.jpg",
    "product_22": "reference_images/product_22.jpg",
    "product_23": "reference_images/product_23.jpg",
    "product_24": "reference_images/product_24.jpg",
    "product_25": "reference_images/product_25.jpg"
}


reference_descriptors = {}
for label, ref_image_path in reference_images.items():
    ref_image = cv2.imread(ref_image_path)
    ref_image = transform(ref_image).unsqueeze(0).to(device)  # Move image tensor to GPU
    with torch.no_grad():
        ref_descriptor = embedder(ref_image).cpu().numpy().flatten()  # Move result back to CPU
    reference_descriptors[label] = ref_descriptor


# Convert reference descriptors into a database
labels = list(reference_descriptors.keys())
descriptor_list = np.array(list(reference_descriptors.values()))
neigh = NearestNeighbors(n_neighbors=1, metric='euclidean')
neigh.fit(descriptor_list)
# colors = [
#     'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
#     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
#     '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
#     'dodgerblue', 'gold', 'mediumseagreen', 'crimson', 'darkviolet'
# ]




# Process detected product regions
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Map product labels to unique integer IDs
label_to_id = {label: idx for idx, label in enumerate(labels)}

detected_embeddings = []
detected_labels = []

for i, (x, y, w, h) in enumerate(detections.xyxy):
    cropped = image[int(y):int(h), int(x):int(w)]  # Crop detected region
    cropped = transform(cropped).unsqueeze(0).to(device)  # Move image tensor to GPU
    with torch.no_grad():
        descriptor = embedder(cropped).cpu().numpy().flatten()  # Move result back to CPU
        detected_embeddings.append(descriptor)


    # Find the nearest reference descriptor
    _, indices = neigh.kneighbors([descriptor])
    predicted_label = labels[indices[0][0]]  # Get product label
    detected_labels.append(predicted_label)
    
    # Assign integer class ID
    detections.class_id[i] = label_to_id[predicted_label]

    # Store readable labels separately
    detections.data["custom_labels"] = detections.data.get("custom_labels", [])
    detections.data["custom_labels"].append(predicted_label)

detected_embeddings = np.array(detected_embeddings)
detected_labels = np.array(detected_labels)
print(len(detected_embeddings), len(detected_labels))

colors = plt.cm.get_cmap("tab10", len(set(detected_labels)))

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
embeddings_2d = tsne.fit_transform(detected_embeddings)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 6))


for i, label in enumerate(set(detected_labels)):
    mask = detected_labels == label
    embeddings_filtered = embeddings_2d[mask]
    plt.scatter(
        embeddings_filtered[:, 0],  # X-axis
        embeddings_filtered[:, 1],  # Y-axis
        color=colors(i),  # Assign a unique color
        label=f"Cluster {label}",
        s=70,  # Marker size
        edgecolors='k',  # Black edge for better visibility
    )

# scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],c=colors, cmap='viridis', alpha=0.7)
# plt.colorbar(scatter, label='Cluster Label')
plt.title("t-SNE Visualization of Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()

annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
# annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# Display the annotated image
sv.plot_image(annotated_image)
