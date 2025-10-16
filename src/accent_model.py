# src/accent_model.py
import torch
import torch.nn as nn
from transformers import VitsModel, AutoTokenizer
from typing import Optional

class AccentVits(nn.Module):
    """
    Wrapper around Hugging Face VitsModel to add explicit accent embedding.
    Mechanism:
      - Obtain token embeddings for input_ids via model.get_input_embeddings()
      - Project accent embedding to same embedding dim and add (broadcasted) to inputs_embeds
      - Call base model with inputs_embeds (instead of input_ids)
    This approach avoids directly hacking encoder internals.
    """
    def __init__(self, base_model_name: str, num_accents: int, accent_emb_dim: int = 256, freeze_base: bool = False):
        super().__init__()
        # load base VITS
        self.base = VitsModel.from_pretrained(base_model_name)
        # get embedding dimension
        # try several fallbacks
        try:
            embedding_layer = self.base.get_input_embeddings()
            self.embed_dim = embedding_layer.embedding_dim
        except Exception:
            # fallback to config.hidden_size (if present)
            self.embed_dim = getattr(self.base.config, "hidden_size", 512)

        self.accent_emb = nn.Embedding(num_accents, accent_emb_dim)
        self.proj = nn.Linear(accent_emb_dim, self.embed_dim)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, accent_ids=None, **kwargs):
        """
        Accepts either input_ids or inputs_embeds. accent_ids shape: (B,) or (B,1)
        Returns what the base model returns (likely waveform / mel inside a ModelOutput).
        """
        if inputs_embeds is None:
            # get token embeddings from base model
            # the embedding layer expects input_ids (LongTensor)
            inputs_embeds = self.base.get_input_embeddings()(input_ids)  # (B, L, E)

        # get accent vector
        if accent_ids is None:
            raise ValueError("accent_ids must be provided (shape [B])")
        if accent_ids.dim() == 1:
            accent_ids = accent_ids.unsqueeze(1)  # (B,1) for clarity
        # take first column if multiple dims
        accent_ids = accent_ids[:, 0].long()
        a_emb = self.accent_emb(accent_ids)        # (B, accent_emb_dim)
        a_proj = self.proj(a_emb)                  # (B, embed_dim)
        # expand to seq length
        seq_len = inputs_embeds.size(1)
        a_expanded = a_proj.unsqueeze(1).expand(-1, seq_len, -1)  # (B, L, E)

        # condition by addition (you may try concatenation & projection as alternative)
        conditioned_embeds = inputs_embeds + a_expanded

        # forward base model with conditioned inputs_embeds
        out = self.base(inputs_embeds=conditioned_embeds, attention_mask=attention_mask, **kwargs)
        return out

    def save_pretrained(self, save_directory):
        # Save base model & accent embeddings
        self.base.save_pretrained(save_directory)
        # save accent embedding & proj manually
        torch.save({
            "accent_emb.weight": self.accent_emb.state_dict(),
            "proj.weight": self.proj.state_dict(),
            "proj.bias": self.proj.bias.detach().cpu()
        }, f"{save_directory}/accent_head.pt")

    @classmethod
    def from_pretrained_with_accent(cls, save_directory, num_accents, accent_emb_dim=256):
        base = VitsModel.from_pretrained(save_directory)
        wrapper = cls(base_model_name=save_directory, num_accents=num_accents, accent_emb_dim=accent_emb_dim)
        # load accent_head.pt if exists
        import os, torch
        p = os.path.join(save_directory, "accent_head.pt")
        if os.path.exists(p):
            sd = torch.load(p, map_location="cpu")
            wrapper.accent_emb.load_state_dict({"weight": sd["accent_emb.weight"]} if "accent_emb.weight" in sd else sd)
            try:
                wrapper.proj.weight.data.copy_(sd["proj.weight"])
                wrapper.proj.bias.data.copy_(sd["proj.bias"])
            except Exception:
                pass
        return wrapper
