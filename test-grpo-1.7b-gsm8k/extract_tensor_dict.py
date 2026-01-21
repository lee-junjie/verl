import os
import pickle
import sys

# Ensure we can load DataProto
sys.path.append(os.getcwd())

input_path = "tmp/batch.1.pkl"
output_path = "tmp/batch.1.raw.pkl"

print(f"Loading {input_path}...")
with open(input_path, "rb") as f:
    data_proto = pickle.load(f)

# Extract and move to CPU
batch = data_proto.batch
if batch is not None:
    batch = batch.cpu()

raw_data = {
    "batch": batch,
    "non_tensor_batch": data_proto.non_tensor_batch,
    "meta_info": data_proto.meta_info
}

print(f"Saving to {output_path}...")
with open(output_path, "wb") as f:
    pickle.dump(raw_data, f)
print("Done.")
