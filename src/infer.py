import torch
from src.accent_model import AccentVITS
from transformers import AutoTokenizer
import soundfile as sf

model = AccentVITS(num_accents=7)
model.load_state_dict(torch.load("checkpoints/fine_tuned/model_best.pt"))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
text = "Chúc bạn ngủ ngon"
inputs = tokenizer(text, return_tensors="pt")

accent_id = torch.tensor([3])  # 3 = hue
with torch.no_grad():
    audio = model(inputs.input_ids, inputs.attention_mask, accent_id).waveform

sf.write("hue_output.wav", audio.squeeze().cpu().numpy(), 16000)
