# detection_utils.py
import os
import cv2
from ultralytics import YOLO

def load_model(model_path: str) -> YOLO:
    """Load YOLO model from specified path"""
    return YOLO(model_path)

def run_detection(model: YOLO, image_path: str):
    """Perform object detection on image"""
    return model(image_path)

def ensure_output_dir(output_folder: str):
    """Create output directory if missing"""
    os.makedirs(output_folder, exist_ok=True)

def process_detections(results, image, output_folder: str):
    """Save cropped bounding boxes from detection results"""
    for i, result in enumerate(results):
        boxes = result.boxes
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = image[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(output_folder, f'crop_{i}_{j}.jpg'), crop)

def detect_and_crop(model_path: str, image_path: str, output_folder: str):
    """End-to-end detection and cropping pipeline"""
    model = load_model(model_path)
    results = run_detection(model, image_path)
    ensure_output_dir(output_folder)
    image = cv2.imread(image_path)
    annotated_frame = results[0].plot()
    cv2.imwrite('static/predictions/'+image_path.split('/')[-1], annotated_frame)
    process_detections(results, image, output_folder)
