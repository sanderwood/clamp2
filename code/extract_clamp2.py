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
from transformers import BertConfig, AutoTokenizer
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Feature extraction for CLaMP2.")
parser.add_argument("input_dir", type=str, help="Directory containing input data files.")
parser.add_argument("output_dir", type=str, help="Directory to save the output features.")
parser.add_argument("--normalize", action="store_true", help="Normalize the extracted features.")

args = parser.parse_args()

# Retrieve arguments
input_dir = args.input_dir
output_dir = args.output_dir
normalize = args.normalize

os.makedirs("logs", exist_ok=True)
for file in ["logs/files_extract_clamp2.json",
             "logs/files_shuffle_extract_clamp2.json",
             "logs/log_extract_clamp2.txt",
             "logs/pass_extract_clamp2.txt",
             "logs/skip_extract_clamp2.txt"]:
    if os.path.exists(file):
        os.remove(file)

files = []
for root, dirs, fs in os.walk(input_dir):
    for f in fs:
        if f.endswith(".txt") or f.endswith(".abc") or f.endswith(".mtf"):
            files.append(os.path.join(root, f))
print(f"Found {len(files)} files in total")
with open("logs/files_extract_clamp2.json", "w", encoding="utf-8") as f:
    json.dump(files, f)
random.shuffle(files)
with open("logs/files_shuffle_extract_clamp2.json", "w", encoding="utf-8") as f:
    json.dump(files, f) 

accelerator = Accelerator()
device = accelerator.device
print("Using device:", device)
with open("logs/log_extract_clamp.txt", "a", encoding="utf-8") as f:
    f.write("Using device: " + str(device) + "\n")

m3_config = BertConfig(vocab_size=1,
                        hidden_size=M3_HIDDEN_SIZE,
                        num_hidden_layers=PATCH_NUM_LAYERS,
                        num_attention_heads=M3_HIDDEN_SIZE//64,
                        intermediate_size=M3_HIDDEN_SIZE*4,
                        max_position_embeddings=PATCH_LENGTH)
model = CLaMP2Model(m3_config,
                    text_model_name=TEXT_MODEL_NAME,
                    hidden_size=CLAMP2_HIDDEN_SIZE,
                    load_m3=CLAMP2_LOAD_M3)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
patchilizer = M3Patchilizer()

# print parameter number
print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

model.eval()
checkpoint = torch.load(CLAMP2_WEIGHTS_PATH, map_location='cpu', weights_only=True)
print(f"Successfully Loaded CLaMP 2 Checkpoint from Epoch {checkpoint['epoch']} with loss {checkpoint['min_eval_loss']}")
model.load_state_dict(checkpoint['model'])

def extract_feature(filename, get_normalized=normalize):
    with open(filename, "r", encoding="utf-8") as f:
        item = f.read()

    if filename.endswith(".txt"):
        item = list(set(item.split("\n")))
        item = "\n".join(item)
        item = item.split("\n")
        item = [c for c in item if len(c) > 0]
        item = tokenizer.sep_token.join(item)
        input_data = tokenizer(item, return_tensors="pt")
        input_data = input_data['input_ids'].squeeze(0)
        max_input_length = MAX_TEXT_LENGTH
    else:
        input_data = patchilizer.encode(item, add_special_patches=True)
        input_data = torch.tensor(input_data)
        max_input_length = PATCH_LENGTH

    segment_list = []
    for i in range(0, len(input_data), max_input_length):
        segment_list.append(input_data[i:i+max_input_length])
    segment_list[-1] = input_data[-max_input_length:]

    last_hidden_states_list = []

    for input_segment in segment_list:
        input_masks = torch.tensor([1]*input_segment.size(0))
        if filename.endswith(".txt"):
            pad_indices = torch.ones(MAX_TEXT_LENGTH - input_segment.size(0)).long() * tokenizer.pad_token_id
        else:
            pad_indices = torch.ones((PATCH_LENGTH - input_segment.size(0), PATCH_SIZE)).long() * patchilizer.pad_token_id
        input_masks = torch.cat((input_masks, torch.zeros(max_input_length - input_segment.size(0))), 0)
        input_segment = torch.cat((input_segment, pad_indices), 0)

        if filename.endswith(".txt"):
            last_hidden_states = model.get_text_features(text_inputs=input_segment.unsqueeze(0).to(device),
                                                         text_masks=input_masks.unsqueeze(0).to(device),
                                                         get_normalized=get_normalized)
        else:
            last_hidden_states = model.get_music_features(music_inputs=input_segment.unsqueeze(0).to(device),
                                                          music_masks=input_masks.unsqueeze(0).to(device),
                                                          get_normalized=get_normalized)
        if not get_normalized:
            last_hidden_states = last_hidden_states[:, :input_masks.sum().long().item(), :]
        last_hidden_states_list.append(last_hidden_states)

    if not get_normalized:
        last_hidden_states_list = [last_hidden_states[0] for last_hidden_states in last_hidden_states_list]
        last_hidden_states_list[-1] = last_hidden_states_list[-1][-(len(input_data)%max_input_length):]
        last_hidden_states_list = torch.concat(last_hidden_states_list, 0)
    else:
        full_chunk_cnt = len(input_data) // max_input_length
        remain_chunk_len = len(input_data) % max_input_length
        if remain_chunk_len == 0:
            feature_weights = torch.tensor([max_input_length] * full_chunk_cnt, device=device).view(-1, 1)
        else:
            feature_weights = torch.tensor([max_input_length] * full_chunk_cnt + [remain_chunk_len], device=device).view(-1, 1)
        
        last_hidden_states_list = torch.concat(last_hidden_states_list, 0)
        last_hidden_states_list = last_hidden_states_list * feature_weights
        last_hidden_states_list = last_hidden_states_list.sum(dim=0) / feature_weights.sum()

    return last_hidden_states_list

def process_directory(input_dir, output_dir, files):
    print(f"Found {len(files)} files in total")
    with open("logs/log_extract_clamp.txt", "a", encoding="utf-8") as f:
        f.write("Found " + str(len(files)) + " files in total\n")

    # calculate the number of files to process per GPU
    num_files_per_gpu = len(files) // accelerator.num_processes

    # calculate the start and end index for the current GPU
    start_idx = accelerator.process_index * num_files_per_gpu
    end_idx = start_idx + num_files_per_gpu
    if accelerator.process_index == accelerator.num_processes - 1:
        end_idx = len(files)

    files_to_process = files[start_idx:end_idx]

    # process the files
    for file in tqdm(files_to_process):
        output_subdir = output_dir + os.path.dirname(file)[len(input_dir):]
        try:
            os.makedirs(output_subdir, exist_ok=True)
        except Exception as e:
            print(output_subdir + " can not be created\n" + str(e))
            with open("logs/log_extract_clamp.txt", "a") as f:
                f.write(output_subdir + " can not be created\n" + str(e) + "\n")

        output_file = os.path.join(output_subdir, os.path.splitext(os.path.basename(file))[0] + ".npy")

        if os.path.exists(output_file):
            print(f"Skipping {file}, output already exists")
            with open("logs/skip_extract_clamp2.txt", "a", encoding="utf-8") as f:
                f.write(file + "\n")
            continue

        try:
            with torch.no_grad():
                features = extract_feature(file).unsqueeze(0)
            np.save(output_file, features.detach().cpu().numpy())
            with open("logs/pass_extract_clamp2.txt", "a", encoding="utf-8") as f:
                f.write(file + "\n")
        except Exception as e:
            print(f"Failed to process {file}: {e}")
            with open("logs/log_extract_clamp.txt", "a", encoding="utf-8") as f:
                f.write("Failed to process " + file + ": " + str(e) + "\n")

with open("logs/files_shuffle_extract_clamp2.json", "r", encoding="utf-8") as f:
    files = json.load(f)

# process the files
process_directory(input_dir, output_dir, files)

with open("logs/log_extract_clamp.txt", "a", encoding="utf-8") as f:
    f.write("GPU ID: " + str(device) + "\n")