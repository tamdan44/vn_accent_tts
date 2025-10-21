from transformers import VitsModel, AutoTokenizer
import torch

model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

text = "mua hai chai, có cơ hội trúng thẻ cào :)"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform



import scipy
import numpy as np

# If output has shape (1, N), squeeze to (N,)
waveform = output[0].cpu().numpy()

# Optionally normalize if needed (some models output float32 in -1 to 1)
waveform = waveform.astype(np.float32)

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=waveform)
