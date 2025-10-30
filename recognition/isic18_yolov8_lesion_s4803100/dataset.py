from pathlib import Path
import random, numpy as np
from skimage import measure, io
from utils import ensure_dir, set_seed, list_images

# Read-only source
ISIC_ROOT = Path("/home/groups/comp3710/ISIC2018")
IMG_DIR = ISIC_ROOT / "ISIC2018_Task1-2_Training_Input_x2"
MSK_DIR = ISIC_ROOT / "ISIC2018_Task1_Training_GroundTruth_x2"

# Local output (writeable)
OUT_ROOT = Path("data_yolo_isic18")
IMG_OUT = OUT_ROOT / "images"
LBL_OUT = OUT_ROOT / "labels"

def mask_to_bboxes(mask: np.ndarray, min_area: int = 50):
    """Return list of YOLO (xc,yc,w,h) normalized bboxes for connected components."""
    bboxes = []
    mask_bin = (mask > 0).astype(np.uint8)
    if mask_bin.sum() == 0:
        return bboxes
    lbl = measure.label(mask_bin, connectivity=2)
    H, W = mask.shape
    for rid in np.unique(lbl):
        if rid == 0:
            continue
        ys, xs = np.where(lbl == rid)
        if ys.size < min_area:
            continue
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
        xc = (x0 + x1 + 1) / 2.0
        yc = (y0 + y1 + 1) / 2.0
        bboxes.append((xc / W, yc / H, bw / W, bh / H))
    return bboxes

def write_yolo_label(lbl_path: Path, bboxes, class_id: int = 0):
    if not bboxes:
        lbl_path.write_text("")  # empty means no lesion
        return
    # Write the labels to 6 decimal float
    lines = [f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for (x,y,w,h) in bboxes]
    lbl_path.write_text("\n".join(lines))

def _make_mask_map():
    """Map 'ISIC_XXXXXXX' -> mask path by removing '_segmentation.png' suffix."""
    mask_map = {}
    for m in MSK_DIR.iterdir():
        if m.name.endswith("_segmentation.png"):
            base = m.name.replace("_segmentation.png", "")
            mask_map[base] = m
    return mask_map

def prepare_splits(seed: int = 1337, train_ratio=0.70, val_ratio=0.15):
    set_seed(seed)
    for sub in ["train", "val", "test"]:
        ensure_dir(IMG_OUT / sub)
        ensure_dir(LBL_OUT / sub)

    imgs = list_images(IMG_DIR)
    assert imgs, f"No images found in {IMG_DIR}"
    mask_map = _make_mask_map()

    pairs = []
    for img in imgs:
        base = img.stem  # ISIC_XXXXXXX
        msk = mask_map.get(base)
        if msk is not None:
            pairs.append((img, msk))

    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:n_train+n_val]
    test_pairs  = pairs[n_train+n_val:]

    print(f"[INFO] ISIC paired samples: {n} "
          f"(train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)})")

    from shutil import copy2
    def convert(split_name, split_pairs):
        for img_path, msk_path in split_pairs:
            out_img = IMG_OUT / split_name / img_path.name
            out_lbl = LBL_OUT / split_name / img_path.with_suffix(".txt").name
            copy2(img_path, out_img)
            mask = io.imread(msk_path)
            if mask.ndim == 3:
                mask = mask[..., 0]
            bboxes = mask_to_bboxes(mask)
            write_yolo_label(out_lbl, bboxes)

if __name__ == "__main__":
    prepare_splits()
