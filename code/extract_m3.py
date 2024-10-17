import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from config import *
from utils import *
from samplings import *
from accelerate import Accelerator
from transformers import BertConfig, GPT2Config
import argparse

# Parse command-line arguments for input_dir and output_dir
parser = argparse.ArgumentParser(description="Process files to extract features.")
parser.add_argument("input_dir", type=str, help="Directory with input files")
parser.add_argument("output_dir", type=str, help="Directory to save extracted features")
args = parser.parse_args()

# Use args for input and output directories
input_dir = args.input_dir
output_dir = args.output_dir

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Remove existing log files if present
for file in [
    "logs/files_extract_m3.json",
    "logs/files_shuffle_extract_m3.json",
    "logs/log_extract_m3.txt",
    "logs/pass_extract_m3.txt",
    "logs/skip_extract_m3.txt",
]:
    if os.path.exists(file):
        os.remove(file)

# Collect input files
files = []
for root, dirs, fs in os.walk(input_dir):
    for f in fs:
        if f.endswith(".abc") or f.endswith(".mtf"):
            files.append(os.path.join(root, f))

print(f"Found {len(files)} files in total")
with open("logs/files_extract_m3.json", "w", encoding="utf-8") as f:
    json.dump(files, f)

# Shuffle files and save the shuffled order
random.shuffle(files)
with open("logs/files_shuffle_extract_m3.json", "w", encoding="utf-8") as f:
    json.dump(files, f)

# Initialize accelerator and device
accelerator = Accelerator()
device = accelerator.device
print("Using device:", device)
with open("logs/log_extract_m3.txt", "a", encoding="utf-8") as f:
    f.write("Using device: " + str(device) + "\n")

# Model and configuration setup
patchilizer = M3Patchilizer()
encoder_config = BertConfig(
    vocab_size=1,
    hidden_size=M3_HIDDEN_SIZE,
    num_hidden_layers=PATCH_NUM_LAYERS,
    num_attention_heads=M3_HIDDEN_SIZE // 64,
    intermediate_size=M3_HIDDEN_SIZE * 4,
    max_position_embeddings=PATCH_LENGTH,
)
decoder_config = GPT2Config(
    vocab_size=128,
    n_positions=PATCH_SIZE,
    n_embd=M3_HIDDEN_SIZE,
    n_layer=TOKEN_NUM_LAYERS,
    n_head=M3_HIDDEN_SIZE // 64,
    n_inner=M3_HIDDEN_SIZE * 4,
)
model = M3Model(encoder_config, decoder_config).to(device)

# Print parameter count
print("Parameter Number: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

# Load model weights
model.eval()
checkpoint = torch.load(M3_WEIGHTS_PATH, map_location='cpu', weights_only=True)
print(f"Successfully Loaded Checkpoint from Epoch {checkpoint['epoch']} with loss {checkpoint['min_eval_loss']}")
model.load_state_dict(checkpoint['model'])

def extract_feature(item):
    """Extracts features from input data."""
    target_patches = patchilizer.encode(item, add_special_patches=True)
    target_patches_list = [target_patches[i:i + PATCH_LENGTH] for i in range(0, len(target_patches), PATCH_LENGTH)]
    target_patches_list[-1] = target_patches[-PATCH_LENGTH:]

    last_hidden_states_list = []
    for input_patches in target_patches_list:
        input_masks = torch.tensor([1] * len(input_patches))
        input_patches = torch.tensor(input_patches)
        last_hidden_states = model.encoder(
            input_patches.unsqueeze(0).to(device), input_masks.unsqueeze(0).to(device)
        )["last_hidden_state"][0]
        last_hidden_states_list.append(last_hidden_states)

    # Handle the last segment padding correctly
    last_hidden_states_list[-1] = last_hidden_states_list[-1][-(len(target_patches) % PATCH_LENGTH):]
    return torch.concat(last_hidden_states_list, 0)

def process_directory(input_dir, output_dir, files):
    """Processes files in the input directory and saves features to the output directory."""
    print(f"Found {len(files)} files in total")
    with open("logs/log_extract_m3.txt", "a", encoding="utf-8") as f:
        f.write("Found " + str(len(files)) + " files in total\n")

    # Distribute files across processes for parallel processing
    num_files_per_gpu = len(files) // accelerator.num_processes
    start_idx = accelerator.process_index * num_files_per_gpu
    end_idx = start_idx + num_files_per_gpu if accelerator.process_index < accelerator.num_processes - 1 else len(files)
    files_to_process = files[start_idx:end_idx]

    # Process each file
    for file in tqdm(files_to_process):
        output_subdir = output_dir + os.path.dirname(file)[len(input_dir):]
        try:
            os.makedirs(output_subdir, exist_ok=True)
        except Exception as e:
            print(f"{output_subdir} cannot be created\n{e}")
            with open("logs/log_extract_m3.txt", "a") as f:
                f.write(f"{output_subdir} cannot be created\n{e}\n")

        output_file = os.path.join(output_subdir, os.path.splitext(os.path.basename(file))[0] + ".npy")

        if os.path.exists(output_file):
            print(f"Skipping {file}, output already exists")
            with open("logs/skip_extract_m3.txt", "a", encoding="utf-8") as f:
                f.write(file + "\n")
            continue

        try:
            with open(file, "r", encoding="utf-8") as f:
                item = f.read()
                if not item.startswith("ticks_per_beat"):
                    item = item.replace("L:1/8\n", "")
                with torch.no_grad():
                    features = extract_feature(item).unsqueeze(0)
                np.save(output_file, features.detach().cpu().numpy())
                with open("logs/pass_extract_m3.txt", "a", encoding="utf-8") as f:
                    f.write(file + "\n")
        except Exception as e:
            print(f"Failed to process {file}: {e}")
            with open("logs/log_extract_m3.txt", "a", encoding="utf-8") as f:
                f.write(f"Failed to process {file}: {e}\n")

# Load shuffled files list and start processing
with open("logs/files_shuffle_extract_m3.json", "r", encoding="utf-8") as f:
    files = json.load(f)

# Process the directory
process_directory(input_dir, output_dir, files)

with open("logs/log_extract_m3.txt", "a", encoding="utf-8") as f:
    f.write("GPU ID: " + str(device) + "\n")
