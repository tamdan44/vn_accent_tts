# src/accent_model.py
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class AccentEmbeddingModel(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", num_accents=10, embedding_dim=256):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.embedding_dim = embedding_dim

        # Pool to fixed size representation
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(hidden_size, embedding_dim)

        # optional classifier for accent label
        self.classifier = nn.Linear(embedding_dim, num_accents)

    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, T, H]

        # mean pooling over time
        pooled = hidden_states.mean(dim=1)         # [B, H]

        embeddings = self.projection(pooled)       # [B, embedding_dim]
        logits = self.classifier(embeddings)       # [B, num_accents]

        return embeddings, logits

def load_processor(model_name="facebook/wav2vec2-base"):
    return Wav2Vec2Processor.from_pretrained(model_name)
