# Check + change working dir
import os

current_directory = os.getcwd()
print(f"The current working directory is: {current_directory}")


import torch
from torch.utils.data import DataLoader

from dataset import AccentDataset, collate_fn
from datasets import load_from_disk

from model import VitsAccentAdapter
from transformers import VitsModel
from config import DATA_PKL, BATCH_SIZE
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F



def stft_loss(pred_audio, target_audio, n_fft=1024):
    max_len = max(pred_audio.shape[-1], target_audio.shape[-1])
    if pred_audio.shape[-1] < max_len:
        pred_audio = F.pad(pred_audio, (0, max_len - pred_audio.shape[-1]))
    if target_audio.shape[-1] < max_len:
        target_audio = F.pad(target_audio, (0, max_len - target_audio.shape[-1]))

    pred_stft = torch.abs(torch.stft(pred_audio, n_fft=n_fft, return_complex=True))
    target_stft = torch.abs(torch.stft(target_audio, n_fft=n_fft, return_complex=True))

    loss = torch.mean((pred_stft - target_stft) ** 2)
    return loss

base_model = VitsModel.from_pretrained("facebook/mms-tts-vie")
model_with_adapter = VitsAccentAdapter(base_model).to(DEVICE)
ds = AccentDataset(load_from_disk("processed_dataset16k"))

optimizer = torch.optim.Adam(
    list(model_with_adapter.accent_emb.parameters()) +
    list(model_with_adapter.proj.parameters()),
    lr=1e-3
)

for param in base_model.parameters():
    param.requires_grad = False

base_model.eval()


scaler = GradScaler()
model_with_adapter.train()

num_epochs = 5
accumulation_steps = 4
val_interval = 100  # do validation every 100 steps
global_step = 0

for epoch in range(num_epochs):
    for i, batch in enumerate(ds.train_loader):
        audio_target = batch["audio_tensors"].to(DEVICE)
        accent_ids = batch["accent_ids"].to(DEVICE)
        tokenized = batch_tokenize(batch['text'])
        input_ids = tokenized['input_ids'].to(DEVICE)
        attention_mask = tokenized['attention_mask'].to(DEVICE)

        if global_step % accumulation_steps == 0:
            optimizer.zero_grad()

        with autocast():
            outputs = model_with_adapter(input_ids, accent_ids, attention_mask)
            loss = stft_loss(outputs, audio_target)
            loss = loss / accumulation_steps  

        scaler.scale(loss).backward()

        if (global_step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()

        if global_step % 10 == 0:
            print(f"[Epoch {epoch}] Step {global_step} | Loss: {loss.item():.4f}")

        best_val_loss = float("inf")  # keep track of best model

        if global_step % val_interval == 0 and global_step != 0:
            model_with_adapter.eval()
            val_loss_total = 0.0
            val_steps = 0

            with torch.no_grad():
                for val_batch in ds.valid_loader:
                    val_audio_target = val_batch["audio_tensors"].to(DEVICE)
                    val_accent_ids = val_batch["accent_ids"].to(DEVICE)

                    val_tokenized = ds.batch_tokenize(val_batch['text'])
                    val_input_ids = val_tokenized['input_ids'].to(DEVICE)
                    val_attention_mask = val_tokenized['attention_mask'].to(DEVICE)

                    val_outputs = model_with_adapter(val_input_ids, val_accent_ids, val_attention_mask)
                    val_loss = stft_loss(val_outputs, val_audio_target)

                    val_loss_total += val_loss.item()
                    val_steps += 1

            avg_val_loss = val_loss_total / val_steps
            print(f"Validation — Step {global_step} | Avg Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'step': global_step,
                    'model_state_dict': model_with_adapter.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss
                }, "best_model.pth")
                print(f"Saved best model at step {global_step} (val_loss={best_val_loss:.4f})")

            model_with_adapter.train()

        global_step += 1

torch.save(model_with_adapter.state_dict(), "best_model.pth")
