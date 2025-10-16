# src/infer.py
import torch
import soundfile as sf
from transformers import AutoTokenizer
from pathlib import Path
import json

from accent_model import AccentVits
from config import HF_MODEL_NAME, TOKENIZER_NAME, ACCENT2ID_JSON, CHECKPOINT_DIR

def load_wrapper(checkpoint_dir: str, num_accents: int, accent_emb_dim: int = 256, device="cpu"):
    # If you saved using save_pretrained on base, load via from_pretrained_with_accent
    wrapper = AccentVits(base_model_name=checkpoint_dir, num_accents=num_accents, accent_emb_dim=accent_emb_dim, freeze_base=False)
    # try to load accent_head if present (the wrapper implementation checks)
    wrapper.to(device)
    return wrapper

def tts_generate(text: str, accent_name: str, out_path: str, device="cpu"):
    # load accent map
    with open(ACCENT2ID_JSON, "r", encoding="utf-8") as f:
        accent2id = json.load(f)
    if accent_name not in accent2id:
        raise ValueError(f"Unknown accent {accent_name}")
    accent_id = torch.tensor([accent2id[accent_name]], dtype=torch.long).to(device)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    # build model
    # if you saved a full pretrained base model folder inside CHECKPOINT_DIR, you can use that
    num_accents = len(accent2id)
    dev = torch.device(device)
    model = AccentVits(base_model_name=HF_MODEL_NAME, num_accents=num_accents, accent_emb_dim=256)
    model.to(dev)
    model.eval()

    # prepare inputs
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(dev)
    attention_mask = inputs["attention_mask"].to(dev)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, accent_ids=accent_id)
        if hasattr(out, "waveform"):
            wav = out.waveform
        elif hasattr(out, "audio") and out.audio is not None:
            wav = out.audio
        else:
            raise RuntimeError("Model output doesn't include waveform. Adjust infer to use vocoder or mel2wave.")

    # assume shape (1, T)
    wav_np = wav[0].cpu().numpy()
    sf.write(out_path, wav_np, samplerate=16000)  # change sample rate if needed
    print(f"Wrote audio to {out_path}")

if __name__ == "__main__":
    # example
    tts_generate("Xin chào, tôi tên là Thun.", "hanoi", "out_hanoi.wav", device="cuda" if torch.cuda.is_available() else "cpu")
