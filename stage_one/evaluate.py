from ultralytics import YOLO

# Load your trained model (replace with your actual model path)
model = YOLO('trained_yolov10.pt')

# Evaluate the model on the validation set
metrics = model.val(batch=4, data='configs/data.yaml', save=True)

# Print overall metrics
print("📊 Evaluation Results:")
print(f"Precision:     {metrics.box.precision:.4f}")
print(f"Recall:        {metrics.box.recall:.4f}")
print(f"mAP@0.5:       {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95:  {metrics.box.map:.4f}")
print(f"Number of Classes: {metrics.box.nc}")