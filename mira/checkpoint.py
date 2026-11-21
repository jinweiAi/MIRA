"""Atomic checkpoint and metric writes. No base-model weights are duplicated."""
import json
import os
from pathlib import Path
import torch

def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def append_json_record(record, path):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
