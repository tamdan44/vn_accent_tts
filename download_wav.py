import soundfile as sf 
from datasets import load_from_disk

ds = load_from_disk("my_dataset")
test_ds = ds["test"]

for example in test_ds:
    if example["province_name"].strip().lower() == "nghean" and example["gender"] == 1:
    #if example["accent_label"] == 1 and example["gender"] == 1:
        audio_array = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        output_path = "nghean.wav"

        sf.write(output_path, audio_array, sampling_rate)
        print(f"Saved audio to {output_path}")
        break
else:
    print("No matching audio found.")
#torchaudio.save("generated.wav", generated_audio.unsqueeze(0), TARGET_SAMPLE_RATE)
