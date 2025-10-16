# src/utils.py
import torch
import json
from pathlib import Path
import pickle
from typing import List

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def pad_1d(tensors: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """Pad list of 1D tensors (waveforms) to same length -> (B, T)."""
    max_len = max(t.shape[0] for t in tensors)
    out = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, : t.shape[0]] = t
    return out

def pad_input_ids(input_ids_list, padding_value: int = 0):
    """Pad list of 1D int tensors to (B, L)."""
    max_len = max(x.shape[0] for x in input_ids_list)
    out = torch.full((len(input_ids_list), max_len), padding_value, dtype=input_ids_list[0].dtype)
    attention_mask = torch.zeros_like(out)
    for i, x in enumerate(input_ids_list):
        out[i, : x.shape[0]] = x
        attention_mask[i, : x.shape[0]] = 1
    return out, attention_mask

def save_checkpoint(model, optimizer, epoch, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, path)

def load_checkpoint(path, model, optimizer=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", None)
