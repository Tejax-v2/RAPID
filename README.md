# 🛒 RAPID — Retail Analytics and Product Identification using Deep Learning

Welcome to **RAPID**!  
This project is a two-stage deep learning pipeline designed to identify retail products on shelves with high accuracy using a combination of object detection and metric learning.

> You bring the shelf image, we bring the AI magic. ✨

---


---

## 🚀 Pipeline Overview

### 🧠 Stage 1: Product-Agnostic Detection

- **Objective:** Detect all product-like objects on shelves using the [SKU110K dataset](https://github.com/eg4000/SKU110K_CVPR19).
- **Model:** YOLOv10 (You Only Look Once)
- **Training:** Run `train.py` on data defined in `data.yaml`.
- **Evaluation:** Use `evaluate.py` to view performance metrics.
- **Output:** `trained_yolov10.pt` – YOLOv10 weights trained on retail shelves.

---

### 💎 Stage 2: Product Embedding & Identification

- **Embedding Generation:**  
  `embedding_generator.py` uses a modified **ResNet18** (last FC layer removed) to extract embeddings from detected product images.

- **Triplet Loss Training:**  
  - Anchor: Detected product image  
  - Positive: Augmented version of anchor  
  - Negative: Random different product  
  Trained using `triplet_training.py`.

- **Database Management:**  
  `update_embeddings_database.py` generates embeddings for all reference dataset images and stores them in `.pt` format.

- **Identification Algorithms:**
  - **Method 1 (kNN):** Classic nearest neighbor search using cosine distance.
  - **Method 2 (FAISS):** Scalable similarity search using Facebook’s FAISS library.

- **Evaluation:**  
  `evaluation_pipeline.py` runs the full detection → embedding → classification flow on input shelf images.

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/Tejax-v2/RAPID.git
cd RAPID

# Create virtual environment
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows

# Install dependencies
pip install -r requirements.txt
