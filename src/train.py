import torch
from torch.utils.data import DataLoader
from src.dataset import ViAccentDataset
from src.accent_model import AccentVITS
import torch.nn.functional as F

model = AccentVITS(num_accents=7).to("cuda")
dataset = ViAccentDataset("data/metadata.csv", "data/accent2id.json", "facebook/mms-tts-vie")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

for epoch in range(10):
    for batch in loader:
        input_ids = batch["input_ids"].to("cuda")
        mask = batch["attention_mask"].to("cuda")
        accent_id = batch["accent_id"].to("cuda")
        audio = batch["audio"].to("cuda")

        mel_pred = model(input_ids, mask, accent_id).waveform
        loss = F.l1_loss(mel_pred, audio)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} - Loss {loss.item():.4f}")
    torch.save(model.state_dict(), f"checkpoints/fine_tuned/model_{epoch}.pt")
