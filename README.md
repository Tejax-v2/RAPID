# Clustering Algorithms for Image Embeddings

This repository contains two methods for clustering objects detected in images using deep learning-based feature embeddings.

## Methods

### Method 1: Clustering Based on Distance Between Embeddings

This method follows these steps:
1. Uses a pre-trained YOLOv8 model to detect objects in an image.
2. Extracts feature embeddings using a ResNet-50 model.
3. Applies **DBSCAN** clustering to group similar objects based on Euclidean distance between embeddings.
4. Uses **t-SNE** for visualization of clustered embeddings.
5. Evaluates clustering performance using **Silhouette Score** (if applicable).

#### Dependencies
- `torch`
- `torchvision`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `opencv-python`
- `supervision`

#### Usage
```bash
python method1.py
```

### Method 2: Clustering Using a Reference Image Database

This method clusters detected objects by comparing their feature embeddings to a pre-defined reference image database.

Steps:
1. Uses a pre-trained YOLOv8 model to detect objects in an image.
2. Extracts feature embeddings using a ResNet-50 model.
3. Loads precomputed embeddings of known reference images.
4. Uses **Nearest Neighbors** search to assign detected objects to the closest reference image.
5. Uses **t-SNE** for visualization of the clustered results.

#### Dependencies
- `torch`
- `torchvision`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `opencv-python`
- `supervision`

#### Usage
```bash
python method2.py
```

## Results
Both methods produce an annotated image with detected and clustered objects. The results are visualized using t-SNE to show how well the objects are grouped.

## License
This project is open-source under the MIT License.

## Author
Tejas Tupke