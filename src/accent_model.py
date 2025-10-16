import torch
import torch.nn as nn
from transformers import VitsModel

class AccentVITS(nn.Module):
    def __init__(self, base_model_name="facebook/mms-tts-vie", num_accents=7, accent_dim=256):
        super().__init__()
        self.tts = VitsModel.from_pretrained(base_model_name)
        self.accent_emb = nn.Embedding(num_accents, accent_dim)
        self.proj = nn.Linear(accent_dim, self.tts.config.hidden_size)

    def forward(self, input_ids, attention_mask, accent_id):
        accent_vec = self.accent_emb(accent_id)        # [B, accent_dim]
        accent_vec = self.proj(accent_vec).unsqueeze(1)  # [B,1,H]
        hidden_states = self.tts.text_encoder(input_ids, attention_mask)[0] + accent_vec
        mel = self.tts.speech_decoder(hidden_states)
        return mel
