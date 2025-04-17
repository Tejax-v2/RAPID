import os
import torch
import logging
from ultralytics import YOLO

# === Configuration ===
DATA_YAML = 'configs/data.yaml'
MODEL_NAME = 'yolov10n'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
IMG_SIZE = 640
EPOCHS = 10
PATIENCE = 10
INIT_LR = 0.0001
MIN_LR = 0.000001
RESUME_MODEL = "runs/detect/train/weights/last.pt"
LOG_DIR = 'logs'

# === Logging Setup ===
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'training.log'),
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

def main():
    logging.info("Training started.")
    model = YOLO(RESUME_MODEL) if os.path.exists(RESUME_MODEL) else YOLO(MODEL_NAME)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        lr0=INIT_LR,
        lrf=MIN_LR / INIT_LR,
        save=True,
        save_period=1,
        verbose=True
    )

    logging.info("Training finished.")

    # === Hyperparameter tuning suggestions ===
    metrics = results.metrics if hasattr(results, 'metrics') else {}
    map50 = metrics.get('map50', None)
    if map50 is not None:
        if map50 < 0.3:
            logging.info("mAP@0.5 is low. Consider increasing epochs, lowering learning rate, or using a larger model.")
        elif map50 < 0.5:
            logging.info("Decent mAP@0.5, but room for improvement. Try adjusting IMG_SIZE or fine-tuning learning rate schedule.")
        else:
            logging.info("Good mAP@0.5! You might still benefit from slight batch size tuning or augmentation tweaks.")
    else:
        logging.info("No mAP metric available to evaluate performance. Check training results for issues.")

if __name__ == '__main__':
    main()
