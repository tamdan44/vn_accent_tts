# src/dataset.py
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional
from pathlib import Path
import numpy as np
from utils import load_pickle

class ViAccentDataset(Dataset):
    """
    Wrap a HuggingFace Dataset split loaded from pickled DatasetDict.
    Each example expected fields:
      - 'text' (str)
      - 'audio' (dict with 'array' and 'sampling_rate') OR waveform array directly
      - 'accent_label' (int)
      - 'filename' optional
    """
    def __init__(self, hf_dataset_split, tokenizer_name: str, accent2id_map=None, max_waveform_len: Optional[int]=None):
        self.dataset = hf_dataset_split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.accent2id_map = accent2id_map
        self.max_waveform_len = max_waveform_len

    @classmethod
    def from_pickle(cls, pkl_path, split: str, tokenizer_name: str, max_waveform_len: Optional[int]=None):
        ds = load_pickle(pkl_path)
        return cls(ds[split], tokenizer_name, max_waveform_len=max_waveform_len)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        text = ex["text"]
        # audio can be dict or array
        audio = ex["audio"]
        if isinstance(audio, dict):
            arr = audio.get("array")
            sr = audio.get("sampling_rate", None)
        else:
            arr = audio
            sr = None
        # ensure numpy array
        wav = None
        if isinstance(arr, list):
            wav = np.array(arr, dtype=np.float32)
        elif hasattr(arr, "astype"):
            wav = arr.astype(np.float32)
        else:
            raise RuntimeError(f"Unknown audio array type: {type(arr)}")

        # crop/truncate if requested
        if self.max_waveform_len is not None and wav.shape[0] > self.max_waveform_len:
            wav = wav[: self.max_waveform_len]

        accent_label = int(ex["accent_label"]) if "accent_label" in ex else -1

        # tokenize later in collate to support dynamic padding
        return {
            "text": text,
            "waveform": wav,
            "sampling_rate": sr,
            "accent_label": accent_label
        }
