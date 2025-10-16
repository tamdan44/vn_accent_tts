# Check + change working dir
import os

current_directory = os.getcwd()
print(f"The current working directory is: {current_directory}")


import torch
from torch.utils.data import DataLoader

from dataset import AccentDataset, collate_fn
from accent_model import AccentEmbeddingModel, load_processor
from utils import load_pickle
from config import DATA_PKL, BATCH_SIZE

# 1. Load dataset
ds = load_pickle(DATA_PKL)
processor = load_processor()

train_ds = AccentDataset(ds["train"], processor)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
print("Number of training samples:", len(train_ds))

# 2. Init model
num_accents = len(set(ds["train"]["accent_label"]))
model = AccentEmbeddingModel(num_accents=num_accents)
model.to("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache() # clear gpu memory

print("CUDA available:", torch.cuda.is_available())

# 3. Forward pass sample
batch = next(iter(train_loader))
inputs = batch["input_values"].to(model.encoder.device)
embeddings, logits = model(inputs)

print("Accent embedding shape:", embeddings.shape)  # [B, 256]
print("Logits shape:", logits.shape)                # [B, num_accents]