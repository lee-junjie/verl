import pickle

import torch

file_path = "tmp/batch.1.raw.pkl"

print(f"Loading {file_path}...")
with open(file_path, "rb") as f:
    data = pickle.load(f)

print("Keys:", data.keys())

if data.get("batch") is not None:
    print("\n--- Batch (TensorDict) ---")
    print(data["batch"])

if data.get("non_tensor_batch"):
    print("\n--- Non-Tensor Batch ---")
    for k, v in data["non_tensor_batch"].items():
        print(f"{k}: {type(v)}")

if data.get("meta_info"):
    print("\n--- Meta Info ---")
    print(data["meta_info"])
