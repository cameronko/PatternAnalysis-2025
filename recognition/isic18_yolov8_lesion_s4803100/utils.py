from pathlib import Path
import random, numpy as np

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def list_images(img_dir: Path):
    """Return sorted list of .jpg or .png files in a directory."""
    return sorted([p for p in img_dir.iterdir()
                   if p.suffix.lower() in {'.jpg', '.png'}])
