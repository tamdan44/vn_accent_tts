import torch
from transformers import VitsModel, AutoTokenizer

print("CUDA available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VitsModel.from_pretrained("facebook/mms-tts-vie").to(device)

text = "Xin chào, đây là một câu kiểm tra."
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    output = model(**inputs).waveform

print("Output device:", output.device)
