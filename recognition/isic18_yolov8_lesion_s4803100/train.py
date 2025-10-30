import os
from pathlib import Path
from ultralytics import YOLO
from modules import save_curves

def main():
    data_yaml = "yolo_data.yaml"
    out_dir = Path("runs/isic18_yolov8n"); out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")  # fast start
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=32,
        device=0,
        lr0=0.002,
        workers=4,
        patience=20,
        project=str(out_dir),
        name="exp",
        pretrained=True,
        optimizer="SGD",
        close_mosaic=10
    )

    try:
        hist = {
            "precision": results.results_dict.get("metrics/precision(B)", []),
            "recall":    results.results_dict.get("metrics/recall(B)", []),
            "mAP50":     results.results_dict.get("metrics/mAP50(B)", []),
            "mAP50-95":  results.results_dict.get("metrics/mAP50-95(B)", []),
        }
        save_curves(hist, out_dir / "learning_curves.png")
    except Exception:
        pass

if __name__ == "__main__":
    os.system("python dataset.py")
    main()
