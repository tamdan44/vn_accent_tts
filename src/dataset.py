from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
import json

class ViAccentDataset(Dataset):
    def __init__(self, hf_dataset_split, accent2id_path, tokenizer_name):
        """
        Args:
            hf_dataset_split: HuggingFace Dataset split object (e.g., ds["train"])
            accent2id_path: path to JSON file mapping accent name → ID
            tokenizer_name: e.g., "facebook/mms-tts-vie"
        """
        self.dataset = hf_dataset_split
        with open(accent2id_path, "r") as f:
            self.accent2id = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = sample["text"]
        accent_name = sample["region"].strip().lower()
        accent_id = self.accent2id.get(accent_name, 0)  # fallback if not found

        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True
        )

        # Audio
        wav_array = torch.tensor(sample["audio"]["array"], dtype=torch.float32)
        sr = sample["audio"]["sampling_rate"]

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "accent_id": torch.tensor(accent_id),
            "audio": wav_array,
            "sampling_rate": sr
        }

    def __len__(self):
        return len(self.dataset)
