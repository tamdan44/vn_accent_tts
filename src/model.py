import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VitsModel


class VitsAccentAdapter(nn.Module):
    def __init__(self, base_model, num_accents=NUM_ACCENTS, embed_dim=ACCENT_EMB_DIM):
        super().__init__()
        self.base_model = base_model
        self.accent_emb = nn.Embedding(num_accents, embed_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.residual_gate = nn.Parameter(torch.ones(1))
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, input_ids, accent_ids, attention_mask):
        padding_mask = (input_ids != PAD_ID).to(DEVICE)
        padding_mask = padding_mask.unsqueeze(1).transpose(1, 2)

        encoder_out = self.base_model.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            return_dict=True
        ).last_hidden_state

        accent_vec = self.accent_emb(accent_ids)
        accent_proj = self.proj(accent_vec).unsqueeze(1)
        encoder_out = encoder_out + self.residual_gate * accent_proj

        encoder_out = encoder_out.transpose(1, 2) * self.scale
        padding_mask = padding_mask.transpose(1, 2)

        z = self.base_model.flow(encoder_out, padding_mask)
        audio = self.base_model.decoder(z).squeeze(1)
        return audio
