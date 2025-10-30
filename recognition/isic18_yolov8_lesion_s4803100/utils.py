import os, random, shutil
from pathlib import Path
import numpy as np

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def list_images(img_dir: Path):
    return sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {'.jpg', '.png'}
    ])

def copy_files(pairs, out_img_dir: Path, out_mask_dir: Path | None = None):
    ensure_dir(out_img_dir)
    if out_mask_dir:
        ensure_dir(out_mask_dir)
    for img, msk in pairs:
        shutil.copy2(img, out_img_dir / img.name)
        if out_mask_dir and msk:
            shutil.copy2(msk, out_mask_dir / msk.name)

