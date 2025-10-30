from pathlib import Path
import numpy as np
from ultralytics import YOLO
from modules import compute_iou, yolo_xywh_to_xyxy

DATA_ROOT = Path("data_yolo_isic18")
TEST_IMG = DATA_ROOT / "images/test"
TEST_LBL = DATA_ROOT / "labels/test"

def evaluate(weights_path: str, conf=0.25, iou_thr=0.6, imgsz=640):
    model = YOLO(weights_path)
    img_paths = sorted([p for p in TEST_IMG.iterdir() if p.suffix.lower() in {".jpg",".png"}])

    ious = []
    for img_path in img_paths:
        results = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, iou=iou_thr, verbose=False)
        preds = results[0].boxes.xywhn.cpu().numpy() if results and results[0].boxes is not None else []

        # load GT label
        lbl_path = TEST_LBL / (img_path.with_suffix(".txt").name)
        gts = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                if not line: continue
                _, xc, yc, w, h = map(float, line.split())
                gts.append(yolo_xywh_to_xyxy(xc,yc,w,h))

        # greedy best-match IoU for each GT
        for gt in gts:
            best = 0.0
            for pr in preds:
                best = max(best, compute_iou(gt, yolo_xywh_to_xyxy(*pr)))
            ious.append(best)

    if ious:
        ious = np.array(ious)
        print(f"Test IoU mean: {ious.mean():.4f}")
        print(f"Test IoU >= 0.8 (recall of GT): {(ious>=0.8).mean():.3f}")
    else:
        print("No GT boxes found in test split (check split/labels).")

if __name__ == "__main__":
    evaluate("runs/isic18_yolov8n/exp/weights/best.pt")
