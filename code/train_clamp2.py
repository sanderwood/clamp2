import os
import json
import time
import wandb
import torch
import random
import numpy as np
from utils import *
from config import *
from tqdm import tqdm
from copy import deepcopy
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, BertConfig, get_constant_schedule_with_warmup

def list_files_in_json(json_path):
    file_list = []
    
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                file_list.append(item)

    return file_list

def collate_batch(batch):
    text_inputs, text_masks, music_inputs, music_masks = zip(*batch)

    text_inputs = torch.stack(text_inputs)
    text_masks = torch.stack(text_masks)
    music_inputs = torch.stack(music_inputs)
    music_masks = torch.stack(music_masks)

    return text_inputs, text_masks, music_inputs, music_masks

class TextMusicDataset(Dataset):
    def __init__(self, items, mode):
        print("The number of "+mode+" data: "+str(len(items)))
        self.items = items
        self.mode = mode
        if self.mode == 'train' or not EVAL_JSONL:
            self.datapath = os.path.dirname(TRAIN_JSONL)
        elif self.mode == 'eval':
            self.datapath = os.path.dirname(EVAL_JSONL)

    def text_dropout(self, item):
        if random.random() < 0.5:
            candidates = []
            for key in item.keys():
                if key not in ["summary_en", "summary_nen", "filepaths"]:
                    if item[key] == None:
                        continue
                    elif isinstance(item[key], str):
                        candidates.append(item[key])
                    elif isinstance(item[key], list):
                        candidates.extend(item[key])
            candidates = list(set(candidates))
            candidates = "\n".join(candidates)
            candidates = candidates.split("\n")
            selected_candidates = [c for c in candidates if len(c) > 0 and random.random() < 0.5]
            if len(selected_candidates) == 0:
                selected_candidates = candidates
            random.shuffle(selected_candidates)
            text = tokenizer.sep_token.join(selected_candidates)
        else:
            if random.random() < 0.5:
                text = random.choice(item["summary_en"])
            else:
                text = random.choice(item["summary_nen"])["summary"]
        
        return text

    def random_truncate(self, input_tensor, max_length):
        choices = ["head", "tail", "middle"]
        choice = random.choice(choices)
        if choice == "head" or self.mode == 'eval':
            input_tensor = input_tensor[:max_length]
        elif choice == "tail":
            input_tensor = input_tensor[-max_length:]
        elif choice == "middle":
            start = random.randint(1, input_tensor.size(0)-max_length)
            input_tensor = input_tensor[start:start+max_length]
        
        return input_tensor
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]

        # randomly select text from the item
        if self.mode == 'train' and TEXT_DROPOUT:
            text = self.text_dropout(item)
        else:
            text = item["summary_en"][0]

        # tokenize text and build mask for text tokens
        text_inputs = tokenizer(text, return_tensors='pt')
        text_inputs = text_inputs['input_ids'].squeeze(0)
        if text_inputs.size(0) > MAX_TEXT_LENGTH:
            text_inputs = self.random_truncate(text_inputs, MAX_TEXT_LENGTH)
        text_masks = torch.ones(text_inputs.size(0))

        # load music file
        if self.mode == 'train':
            filepath = random.choice(item["filepaths"])
        else:
            if item["filepaths"][0].endswith(".abc"):
                filepath = item["filepaths"][0]
            else:
                filepath = item["filepaths"][1]
        filepath = self.datapath + '/' + filepath

        with open(filepath, "r", encoding="utf-8") as f:
            item = f.read().replace("L:1/8\n", "") if filepath.endswith(".abc") else f.read()
        
        # randomly remove instrument info from the music file
        if random.random() < 0.9 and self.mode == 'train':
            item = remove_instrument_info(item)

        # mask music inputs
        music_inputs = patchilizer.encode(item, add_special_patches=True, truncate=True, random_truncate=(self.mode=="train"))
        music_inputs = torch.tensor(music_inputs)
        music_masks = torch.ones(music_inputs.size(0))
        
        # pad text inputs and masks
        pad_indices = torch.ones(MAX_TEXT_LENGTH - text_inputs.size(0)).long() * tokenizer.pad_token_id
        text_inputs = torch.cat((text_inputs, pad_indices), 0)
        text_masks = torch.cat((text_masks, torch.zeros(MAX_TEXT_LENGTH - text_masks.size(0))), 0)

        # pad music inputs and masks
        pad_indices = torch.ones((PATCH_LENGTH - music_inputs.size(0), PATCH_SIZE)).long() * patchilizer.pad_token_id
        music_inputs = torch.cat((music_inputs, pad_indices), 0)
        music_masks = torch.cat((music_masks, torch.zeros(PATCH_LENGTH - music_masks.size(0))), 0)
        
        return text_inputs, text_masks, music_inputs, music_masks

# call model with a batch of input
def process_one_batch(batch):
    text_inputs, text_masks, music_inputs, music_masks = batch
    
    loss = model(text_inputs,
                text_masks,
                music_inputs,
                music_masks)

    # Reduce the loss on GPU 0
    if world_size > 1:
        loss = loss.unsqueeze(0)
        dist.reduce(loss, dst=0)
        loss = loss / world_size
        dist.broadcast(loss, src=0)

    return loss.mean()

# do one epoch for training
def train_epoch(epoch):
    tqdm_train_set = tqdm(train_set)
    total_train_loss = 0
    iter_idx = 1
    model.train()
    train_steps = (epoch-1)*len(train_set)

    for batch in tqdm_train_set:
        with autocast(device_type='cuda'):
            loss = process_one_batch(batch)
        scaler.scale(loss).backward()
        total_train_loss += loss.item()
        scaler.step(optimizer)
        scaler.update()
        
        lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        tqdm_train_set.set_postfix({str(global_rank)+'_train_loss': total_train_loss / iter_idx})
        train_steps += 1
        
        # Log the training loss to wandb
        if global_rank==0 and CLAMP2_WANDB_LOG:
            wandb.log({"train_loss": total_train_loss / iter_idx}, step=train_steps)

        iter_idx += 1
        
    return total_train_loss / (iter_idx-1)

# do one epoch for eval
def eval_epoch():
    tqdm_eval_set = tqdm(eval_set)
    total_eval_loss = 0
    iter_idx = 1
    model.eval()
  
    # Evaluate data for one epoch
    for batch in tqdm_eval_set: 
        with torch.no_grad():
            loss = process_one_batch(batch)

        total_eval_loss += loss.item()
        tqdm_eval_set.set_postfix({str(global_rank)+'_eval_loss': total_eval_loss / iter_idx})
        iter_idx += 1

    return total_eval_loss / (iter_idx-1)

# train and eval
if __name__ == "__main__":

    # Set up distributed training
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend='nccl')
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
    if CLAMP2_DETERMINISTIC:
        seed = 42 + global_rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    m3_config = BertConfig(vocab_size=1,
                            hidden_size=M3_HIDDEN_SIZE,
                            num_hidden_layers=PATCH_NUM_LAYERS,
                            num_attention_heads=M3_HIDDEN_SIZE//64,
                            intermediate_size=M3_HIDDEN_SIZE*4,
                            max_position_embeddings=PATCH_LENGTH)
    model = CLaMP2Model(m3_config,
                        global_rank,
                        world_size,
                        TEXT_MODEL_NAME,
                        CLAMP2_HIDDEN_SIZE,
                        CLAMP2_LOAD_M3)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    patchilizer = M3Patchilizer()

    # print parameter number
    print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,  find_unused_parameters=True)
                       
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CLAMP2_LEARNING_RATE)

    if CLAMP2_WANDB_LOG and global_rank==0:
        # Initialize wandb
        if WANDB_KEY:
            wandb.login(key=WANDB_KEY)
        wandb.init(project="clamp2",
                   name=CLAMP2_WEIGHTS_PATH.replace("weights_", "").replace(".pth", ""))
             
    # load filenames under train and eval folder
    train_files = list_files_in_json(TRAIN_JSONL)
    eval_files = list_files_in_json(EVAL_JSONL)

    if len(eval_files)==0:
        train_files, eval_files = split_data(train_files)
       
    train_batch_nums = int(len(train_files) / CLAMP2_BATCH_SIZE)
    eval_batch_nums = int(len(eval_files) / CLAMP2_BATCH_SIZE)

    train_files = train_files[:train_batch_nums*CLAMP2_BATCH_SIZE]
    eval_files = eval_files[:eval_batch_nums*CLAMP2_BATCH_SIZE]

    train_set = TextMusicDataset(train_files, 'train')
    eval_set = TextMusicDataset(eval_files, 'eval')

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=global_rank)
    eval_sampler = DistributedSampler(eval_set, num_replicas=world_size, rank=global_rank)
    
    train_set = DataLoader(train_set, batch_size=CLAMP2_BATCH_SIZE, collate_fn=collate_batch, sampler=train_sampler, shuffle = (train_sampler is None))
    eval_set = DataLoader(eval_set, batch_size=CLAMP2_BATCH_SIZE, collate_fn=collate_batch, sampler=eval_sampler, shuffle = (train_sampler is None))

    lr_scheduler = get_constant_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = 1000)

    if CLAMP2_LOAD_CKPT and os.path.exists(CLAMP2_WEIGHTS_PATH):
        # Load checkpoint to CPU
        checkpoint = torch.load(CLAMP2_WEIGHTS_PATH, map_location='cpu', weights_only=True)

        # Here, model is assumed to be on GPU
        # Load state dict to CPU model first, then move the model to GPU
        if torch.cuda.device_count() > 1:
            # If you have a DataParallel model, you need to load to model.module instead
            cpu_model = deepcopy(model.module)
            cpu_model.load_state_dict(checkpoint['model'])
            model.module.load_state_dict(cpu_model.state_dict())
        else:
            # Load to a CPU clone of the model, then load back
            cpu_model = deepcopy(model)
            cpu_model.load_state_dict(checkpoint['model'])
            model.load_state_dict(cpu_model.state_dict())
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_sched'])
        pre_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        min_eval_loss = checkpoint['min_eval_loss']
        print(f"Successfully Loaded Checkpoint from Epoch {checkpoint['epoch']} with loss {checkpoint['min_eval_loss']}")
        checkpoint = None
    
    else:
        pre_epoch = 0
        best_epoch = 0
        min_eval_loss = float('inf')

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CLAMP2_LEARNING_RATE)

    for epoch in range(1+pre_epoch, CLAMP2_NUM_EPOCH+1):
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        print('-' * 21 + "Epoch " + str(epoch) + '-' * 21)
        train_loss = train_epoch(epoch)
        eval_loss = eval_epoch()
        if global_rank==0:
            with open(CLAMP2_LOGS_PATH,'a') as f:
                f.write("Epoch " + str(epoch) + "\ntrain_loss: " + str(train_loss) + "\neval_loss: " +str(eval_loss) + "\ntime: " + time.asctime(time.localtime(time.time())) + "\n\n")
            if eval_loss < min_eval_loss:
                best_epoch = epoch
                min_eval_loss = eval_loss
                checkpoint = { 
                                'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_sched': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                'min_eval_loss': min_eval_loss
                                }
                torch.save(checkpoint, CLAMP2_WEIGHTS_PATH)
            checkpoint = { 
                            'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_sched': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'best_epoch': best_epoch,
                            'min_eval_loss': min_eval_loss
                            }
            torch.save(checkpoint, "latest_"+CLAMP2_WEIGHTS_PATH)

        if world_size > 1:
            dist.barrier()

    if global_rank==0:
        print("Best Eval Epoch : "+str(best_epoch))
        print("Min Eval Loss : "+str(min_eval_loss))
