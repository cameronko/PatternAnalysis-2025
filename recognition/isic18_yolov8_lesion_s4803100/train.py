import os
from pathlib import Path
from ultralytics import YOLO
from modules import save_curves

def _find_latest_last(project_dir: Path) -> Path | None:
    """Return newest runs/.../weights/last.pt if present, else None."""
    if not project_dir.exists():
        return None
    candidates = []
    for exp in project_dir.iterdir():
        w = exp / "weights" / "last.pt"
        if w.exists():
            candidates.append(w)
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

def main():
    data_yaml = "yolo_data.yaml"
    out_dir = Path("runs/isic18_yolov8n"); out_dir.mkdir(parents=True, exist_ok=True)

    # Try to resume if a checkpoint exists
    last_ckpt = _find_latest_last(out_dir)
    if last_ckpt is not None:
        print(f"[INFO] Resuming training from: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        results = model.train(resume=True)  # uses the original args from that run
    else:
        print("[INFO] No checkpoint found. Starting fresh from yolov8n.pt")
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
            close_mosaic=10,
            save_period=5,     # save every 5 epochs so you can resume if killed
            cache=True,        # cache labels/images to reduce I/O stalls
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
    # only prepare dataset if I'm not resuming from checkpoint
    os.system("python dataset.py")
    main()
