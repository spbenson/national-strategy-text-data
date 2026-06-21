import gc
import torch

LABELS = ["Not_Aligned", "Aligned", "Neutral/Irrelevant"]
LABEL_MAPPING = {label: idx for idx, label in enumerate(LABELS)}

def free_gpu_memory(*objs):
    """
    Explicitly deletes the given objects and clears CUDA's cached allocator.
    Call this in build.py between models to prevent GPU memory from
    accumulating across a multi-model training loop.
    """
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()