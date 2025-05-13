import torch
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def classify_embeddings(input_embeddings, folder_path='embeddings'):
    """
    input_embeddings: List of torch.Tensor or numpy arrays of shape (512,)
    folder_path: Path to the folder containing .pt files named like '1.pt', '2.pt', etc.

    Returns: List of labels corresponding to nearest neighbors
    """

    # Load known embeddings from folder
    train_embeddings = []
    train_labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            label = int(filename.split('.')[0])  # assuming label is the filename
            emb = torch.load(os.path.join(folder_path, filename))

            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()

            train_embeddings.append(emb)
            train_labels.append(label)

    train_embeddings = np.array(train_embeddings)

    # Ensure input embeddings are also in numpy format
    input_embeddings_np = [
        emb.cpu().numpy() if isinstance(emb, torch.Tensor) else emb
        for emb in input_embeddings
    ]

    # Train 1-NN classifier
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn.fit(train_embeddings, train_labels)

    # Predict labels for input embeddings
    predicted_labels = knn.predict(input_embeddings_np)

    return predicted_labels.tolist()
