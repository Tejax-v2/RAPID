from stage_two.embeddings.embedding_generator import ResNetEmbedding
from ultralytics import YOLO
from stage_two.evaluate.generate_embeddings import generate_object_embeddings
from stage_two.evaluate.kNN import classify_embeddings
from stage_two.evaluate.FAISS import classify_embeddings_faiss

# Load both models
yolo_model = YOLO("/DATA/tejas_2101cs78/Projects/RAPID/stage_one/trained_yolov10.pt")
resnet_model = ResNetEmbedding()

# Stage 1: YOLO detects stuff
results = yolo_model.predict("/DATA/tejas_2101cs78/Projects/RAPID/images/test_1.jpg", conf=0.5)[0]

# Stage 2: Extract embeddings for each detected object
embeddings = generate_object_embeddings(results, resnet_model)
print(f"Detected {len(embeddings)} objects. Each has embedding of shape {embeddings[0].shape}")

# labels = classify_embeddings(embeddings, folder_path='/DATA/tejas_2101cs78/Projects/RAPID/stage_two/embeddings/embeddings')
labels = classify_embeddings_faiss(embeddings, folder_path='/DATA/tejas_2101cs78/Projects/RAPID/stage_two/embeddings/embeddings')   
print(labels) 
