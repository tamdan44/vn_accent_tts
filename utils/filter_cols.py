from datasets import load_from_disk
ds = load_from_disk("my_dataset", keep_in_memory=False)

print(ds)

# 2. Define the filtering function
valid_provinces = {
    "HaNoi", "HoChiMinh", "QuangNam",
    "QuangBinh", "ThuaThienHue", "NgheAn", "CaMau"
}

def batch_filter(batch):
    return [name in valid_provinces for name in batch["province_name"]]

# 3. Filter in batches to save memory
train_filtered = ds.filter(
    batch_filter,
    batched=True,
    batch_size=128,
    num_proc=1
)

# 4. Remove unwanted columns after filtering
train_filtered = train_filtered.remove_columns(
    ["region", "province_code", "speakerID", "filename"]
)

# 5. Print result
print(train_filtered)



train_filtered.save_to_disk("filtered_dataset")