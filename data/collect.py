import numpy as np
from pathlib import Path
import time
import sys

# Get project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from config import BASE_DIR

DATASET_ROOT = BASE_DIR / "dataset"

def save_samples(gesture_name, samples):
    """
    samples: list of np.array (21, 3)
    saves each as a separate .npy file in dataset/gesture_name/
    """
    dest = DATASET_ROOT / gesture_name
    dest.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time() * 1000)
    count = 0
    for i, arr in enumerate(samples):
        filename = dest / f"{timestamp}_{i}.npy"
        np.save(filename, arr)
        count += 1
    return count
