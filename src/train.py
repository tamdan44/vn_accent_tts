# src/train.py
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import math
import os
from transformers import AutoTokenizer
from datasets import DatasetDict

from config import *
from utils import pad_1d, pad_input_ids, save_checkpoint, load_pickle
from dataset import ViAccentDataset
from accent_model import AccentVits

def collate_fn(batch, tokenizer) -> dict:
    """
    batch: list of dicts {"text","waveform","sampling_rate","accent_label"}
    returns dict of tensors to feed model wrapper
    """
    # tokenize texts -> produce torch LongTensor list
    encs = [tokenizer(x["text"], add_special_tokens=True, return_tensors="pt")["input_ids"].squeeze(0) for x in batch]
    input_ids_padded, attention_mask = pad_input_ids(encs, padding_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    # waveforms -> pad as float32 (B, T)
    waves = [torch.tensor(x["waveform"], dtype=torch.float32) for x in batch]
    wavs_padded = pad_1d(waves, pad_value=0.0)
    accent_ids = torch.tensor([int(x["accent_label"]) for x in batch], dtype=torch.long)
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "waveforms": wavs_padded,
        "accent_ids": accent_ids
    }

def train():
    device = torch.device("cuda" if torch.cuda.is_available() and DEVICE == "cuda" else "cpu")
    print("Using device:", device)
    print("CUDA available:", torch.cuda.is_available())

    # Load dataset
    ds = load_pickle(DATA_PKL)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(ds)
    print(tokenizer)

    # Build datasets
    train_ds = ViAccentDataset(ds["train"], tokenizer_name=TOKENIZER_NAME, max_waveform_len=None)
    valid_ds = ViAccentDataset(ds["valid"], tokenizer_name=TOKENIZER_NAME, max_waveform_len=None)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))

    # load model wrapper
    # number of accents from accent2id.json length
    import json
    with open(ACCENT2ID_JSON, "r", encoding="utf-8") as f:
        accent2id = json.load(f)
    num_accents = len(accent2id)

    model = AccentVits(base_model_name=HF_MODEL_NAME, num_accents=num_accents, accent_emb_dim=ACCENT_EMB_DIM, freeze_base=False)
    model.to(device)
    return

    # optimizer: only train accent emb + proj + maybe some final layers
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LR)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    l1 = nn.L1Loss()

    global_step = 0
    best_valid_loss = float("inf")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_wavs = batch["waveforms"].to(device)
            accent_ids = batch["accent_ids"].to(device)

            # forward: we use inputs_embeds path inside wrapper (wrapper accepts input_ids)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(input_ids=input_ids, attention_mask=attention_mask, accent_ids=accent_ids)
                # attempt to obtain waveform from model output
                if hasattr(out, "waveform"):
                    pred = out.waveform
                elif hasattr(out, "audio") and out.audio is not None:
                    pred = out.audio
                else:
                    # fallback if model returns mel; user should adapt to mel/vocoder flow
                    raise RuntimeError("Model output doesn't have waveform attribute. Adjust training to compute mel->vocoder or provide waveform target accordingly.")

                # Pad/truncate pred to match target length if needed
                # pred expected shape [B, T] or list. Ensure tensor
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)
                # If predicted length differs, pad/truncate to target length
                target_len = target_wavs.shape[1]
                if pred.shape[1] < target_len:
                    # pad
                    pad = torch.zeros((pred.shape[0], target_len - pred.shape[1]), device=pred.device, dtype=pred.dtype)
                    pred = torch.cat([pred, pad], dim=1)
                elif pred.shape[1] > target_len:
                    pred = pred[:, :target_len]

                loss = l1(pred, target_wavs)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            global_step += 1
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch} train loss: {avg_train_loss:.6f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_wavs = batch["waveforms"].to(device)
                accent_ids = batch["accent_ids"].to(device)

                out = model(input_ids=input_ids, attention_mask=attention_mask, accent_ids=accent_ids)
                if hasattr(out, "waveform"):
                    pred = out.waveform
                elif hasattr(out, "audio") and out.audio is not None:
                    pred = out.audio
                else:
                    raise RuntimeError("Model output doesn't have waveform attribute. Adjust training to compute mel->vocoder or provide waveform target accordingly.")

                # align shapes as before
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)
                target_len = target_wavs.shape[1]
                if pred.shape[1] < target_len:
                    pad = torch.zeros((pred.shape[0], target_len - pred.shape[1]), device=pred.device, dtype=pred.dtype)
                    pred = torch.cat([pred, pad], dim=1)
                elif pred.shape[1] > target_len:
                    pred = pred[:, :target_len]

                loss = l1(pred, target_wavs)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(valid_loader)
            print(f"Epoch {epoch} valid loss: {avg_val_loss:.6f}")

            # checkpoint if improved
            ckpt_path = CHECKPOINT_DIR / f"accent_vits_epoch{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, ckpt_path)
            if avg_val_loss < best_valid_loss:
                best_valid_loss = avg_val_loss
                best_path = CHECKPOINT_DIR / "best.pt"
                save_checkpoint(model, optimizer, epoch, best_path)
                print(f"Saved best checkpoint to {best_path}")

    print("Training complete.")

if __name__ == "__main__":
    train()
