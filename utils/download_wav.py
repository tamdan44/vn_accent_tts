import soundfile as sf 
from datasets import load_from_disk
import os

ds = load_from_disk("filtered_dataset")
test_ds = ds["valid"]

def download_wav(ds, province_name, gender, number=5):
    def batch_filter(batch):
        return [
            p.strip().lower() == province_name and g == gender
            for p, g in zip(batch["province_name"], batch["gender"])
        ]

    subset = ds.filter(
        batch_filter,
        batched=True,
        batch_size=128,
        num_proc=1   
    )

    if len(subset) == 0:
        print(f"No matching audio found for {province_name} - gender {gender}")
        return



    if len(subset) == 0:
        print(f"No matching audio found for {province_name} - gender {gender}")
        return

    out_dir = f"test_wav/{province_name}"
    os.makedirs(out_dir, exist_ok=True)

    for i, example in enumerate(subset):
        if i >= number:
            break
        audio_array = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        output_path = f"{out_dir}/{province_name}_{gender}_{i}.wav"

        sf.write(output_path, audio_array, sampling_rate)
        print(f"Saved audio to {output_path}")


provinces = [
    "hochiminh", "quangnam", "quangbinh",
    "nghean", "hanoi", "thuathienhue", "camau"
]

# for province in provinces:
#     for gender in [0, 1]:
#         download_wav(test_ds, province, gender)

for gender in [0, 1]:
    download_wav(test_ds, "hanoi", gender)
