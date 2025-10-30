import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def compute_iou(boxA, boxB):
    # boxes in xyxy normalized
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter = max(0, min(ax2,bx2) - max(ax1,bx1)) * max(0, min(ay2,by2) - max(ay1,by1))
    a = max(0, ax2-ax1) * max(0, ay2-ay1)
    b = max(0, bx2-bx1) * max(0, by2-by1)
    union = a + b - inter + 1e-9
    return inter / union

def yolo_xywh_to_xyxy(xc,yc,w,h):
    return xc - w/2, yc - h/2, xc + w/2, yc + h/2

def save_curves(history: dict, out_png: Path):
    fig = plt.figure()
    for k, v in history.items():
        if isinstance(v, (list, tuple)) and len(v) > 1:
            plt.plot(v, label=k)
    plt.xlabel("epoch"); plt.ylabel("value"); plt.legend(); plt.tight_layout()
    fig.savefig(out_png); plt.close(fig)
