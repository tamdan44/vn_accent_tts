# src/dataset.py
import torch
from torch.utils.data import Dataset
import torchaudio
from torch.utils.data import DataLoader

from config import TARGET_SAMPLE_RATE

class AccentDataset(Dataset):
    """
    Wrap Hugging Face dataset into a PyTorch dataset
    """
    def __init__(self, hf_dataset, processor=None, target_sampling_rate=TARGET_SAMPLE_RATE):
        """
        Args:
            hf_dataset (datasets.Dataset)
            processor (Wav2Vec2Processor)
            target_sampling_rate (int): required sampling rate (default 16kHz)
        """
        
        self.dataset = hf_dataset
        self.processor = processor
        self.target_sampling_rate = target_sampling_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # waveform: float32 numpy [-1, 1]
        audio = item["audio"]["array"]
        sampling_rate = item["audio"]["sampling_rate"]
        waveform = torch.from_numpy(audio).float()

        # Processor expects shape (batch, time)
        if self.processor:
            audio_inputs = self.processor(
                waveform.numpy(),
                sampling_rate=self.target_sampling_rate,
                return_tensors="pt"
            )
            input_values = audio_inputs.input_values.squeeze(0)
        else:
            input_values = waveform

        accent_label = item["accent_label"]

        return {
            "input_values": input_values,
            "label": torch.tensor(accent_label, dtype=torch.long),
            "text": item["text"]
        }
    
    def load_dataloaders(self, batch_size, shuffle=True):
        train_loader = DataLoader(self.dataset["train"], batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        valid_loader = DataLoader(self.dataset["valid"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader  = DataLoader(self.dataset["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return train_loader, valid_loader, test_loader
    
    def filter_gender(self, gender):
        ds = ds.filter(lambda x: x["gender"] == gender , num_proc=1)
        return ds


def collate_fn(batch):
    """
    Pad audio sequences dynamically
    """
    input_values = [b["input_values"] for b in batch]
    labels = [b["label"] for b in batch]

    # pad audio
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    labels = torch.stack(labels)

    return {"input_values": input_values, "labels": labels}
