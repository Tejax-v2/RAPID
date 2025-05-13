import torch
from PIL import Image
from torchvision.transforms.functional import crop
import os
import uuid

def generate_object_embeddings(results, model):
    """
    results: Ultralytics YOLO result object (from model.predict())
    model: An instance of ResNetEmbedding class
    Returns: List of embeddings (each is a torch.Tensor of shape (512,))
    """
    embeddings = []
    temp_dir = "temp_crops"
    os.makedirs(temp_dir, exist_ok=True)

    for result_idx, result in enumerate(results):  # in case multiple images are predicted
        image = Image.fromarray(result.orig_img)  # convert OpenCV image (np.ndarray) to PIL
        boxes = result.boxes.xyxy.cpu()  # shape: (num_boxes, 4) in [x1, y1, x2, y2]

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cropped = image.crop((x1, y1, x2, y2))

            # Save temporarily to disk (because get_embedding_from_path uses path)
            temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")
            cropped.save(temp_path)

            try:
                embedding = model.get_embedding_from_path(temp_path)
                embeddings.append(embedding)
            except Exception as e:
                print(f"❌ Failed to embed cropped object: {e}")
            finally:
                os.remove(temp_path)  # Clean up after yourself, you animal 🧹

    return embeddings
