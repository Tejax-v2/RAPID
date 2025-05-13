import os
import torch
import numpy as np
import faiss  # Make sure you have faiss-cpu installed

def classify_embeddings_faiss(input_embeddings, folder_path='embeddings'):
    """
    input_embeddings: List of torch.Tensor or numpy arrays of shape (512,)
    folder_path: Folder containing .pt files named as '<label>.pt'

    Returns: List of labels corresponding to nearest neighbors using FAISS
    """

    index_vectors = []
    index_labels = []

    # Load .pt files and collect embeddings + labels
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            label = int(filename.split('.')[0])
            emb = torch.load(os.path.join(folder_path, filename))

            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()

            index_vectors.append(emb.astype(np.float32))  # FAISS requires float32
            index_labels.append(label)

    # Convert to numpy array
    index_vectors = np.stack(index_vectors)

    # Build FAISS index (flat L2 index for simplicity)
    dim = index_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(index_vectors)

    # Prepare input embeddings
    input_vectors = [
        emb.cpu().numpy() if isinstance(emb, torch.Tensor) else emb for emb in input_embeddings
    ]
    input_vectors = np.stack([v.astype(np.float32) for v in input_vectors])

    # Search k=1 nearest neighbor
    distances, indices = index.search(input_vectors, k=1)

    # Map indices to labels
    predicted_labels = [index_labels[i[0]] for i in indices]

    return predicted_labels
