# import os
# import json
# from torch.utils.data import Dataset, DataLoader


import json
import pandas as pd
import os
import random
from collections import defaultdict
from torch.optim import AdamW
import pandas as pd
# from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq,TrainingArguments, DataCollatorForLanguageModeling, get_scheduler
from torch.utils.data import DataLoader, Subset, RandomSampler, SequentialSampler
from datasets import DatasetDict, Dataset
import numpy as np
from tqdm import tqdm
import torch

from accelerate import Accelerator
from typing import Any, Dict, List

_REMOVE_LINES = set(
    [
        "Share this with\n",
        "Email\n",
        "Facebook\n",
        "Messenger\n",
        "Twitter\n",
        "Pinterest\n",
        "WhatsApp\n",
        "Linkedin\n",
        "LinkedIn\n",
        "Copy this link\n",
        "These are external links and will open in a new window\n",
    ]
)

class XSumDataset(Dataset):
    def __init__(self, data_dir, split_file, split_name):
        self.data_dir = data_dir
        self.split_file = split_file
        self.split_name = split_name
        self.examples = self._load_data()

    def _load_data(self):
        with open(self.split_file, 'r') as f:
            split_data = json.load(f)
        
        examples = []
        for file_id in split_data[self.split_name]:
            file_path = os.path.join(self.data_dir, f"{file_id}.summary")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    text = "".join([line for line in f.readlines() if line not in _REMOVE_LINES and line.strip()])
                segs = text.split("[SN]")
                examples.append({
                    'document': segs[8].strip(),
                    'summary': segs[6].strip(),
                    'file_id': file_id,
                })
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]




data_dir = "xsum/data/bbc-summary-data"
split_file = "xsum/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json"

# Create dataset instances for training, validation, and test splits
train_dataset = XSumDataset(data_dir, split_file, 'train')
dev_dataset = XSumDataset(data_dir, split_file, 'validation')
test_dataset = XSumDataset(data_dir, split_file, 'test')

path = "Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(path)



sep_token = "[SEP]"
pad_token = "[PAD]"
tokenizer.pad_token = pad_token


added_tokens = []
if sep_token not in tokenizer.get_vocab():
    added_tokens.append(sep_token)
if pad_token not in tokenizer.get_vocab():
    added_tokens.append(pad_token)

if added_tokens:
    tokenizer.add_special_tokens({'additional_special_tokens': added_tokens})
    model.resize_token_embeddings(len(tokenizer))

tokenizer.pad_token = pad_token

sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)

def convert_to_hf_dataset(custom_dataset):
    return Dataset.from_dict({
        'document': [example['document'] for example in custom_dataset.examples],
        'summary': [example['summary'] for example in custom_dataset.examples]
    })

train_hf_dataset = convert_to_hf_dataset(train_dataset)
dev_hf_dataset = convert_to_hf_dataset(dev_dataset)
test_hf_dataset = convert_to_hf_dataset(test_dataset)


# def tokenize_function(examples):
#     # Concatenate the document and summary with the SEP token
#     texts = [doc + sep_token + summ for doc, summ in zip(examples['document'], examples['summary'])]
    
#     # Tokenize the concatenated texts
#     inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=640)
    
#     # Create labels, masking the input part
#     labels = inputs['input_ids'].copy()
#     for i, input_id in enumerate(inputs['input_ids']):
#         summary_len = len(tokenizer(examples['summary'][i], truncation=True, max_length=128)['input_ids'])
#         doc_len = len(input_id) - summary_len  # length of the document part
#         labels[i][:doc_len] = [-100] * doc_len  # Mask the document part
    
#     return {
#         'input_ids': inputs['input_ids'],
#         'labels': labels,
#         'attention_mask': inputs['attention_mask']
#     }

def tokenize_function(examples):
    questions = examples['document']
    answers = examples['summary']

    combined_sequences = []
    for question, answer in zip(questions, answers):
        question = question if question is not None else ""
        answer = answer if answer is not None else ""
        
        combined_sequence = question + " " + sep_token + " " + answer + tokenizer.eos_token
        combined_sequences.append(combined_sequence)

    model_inputs = tokenizer(combined_sequences, max_length=512, truncation=True, padding='max_length')

    input_ids = model_inputs["input_ids"].copy()
    
    # labels = []
    # for i, (question, input_id) in enumerate(zip(questions, input_ids)):
    #     question_tokenized = tokenizer(question, truncation=True, padding='max_length', max_length=512)
    #     question_length = len(question_tokenized["input_ids"]) - question_tokenized["input_ids"].count(tokenizer.pad_token_id)
        
    #     if question_length > len(input_id):
    #         print(f"Error: Question length {question_length} is greater than input ID length {len(input_id)}")
    #         question_length = len(input_id)
        
    #     label = [-100] * question_length + input_id[question_length:]
    #     labels.append(label)
    
    # model_inputs["labels"] = labels

    labels = []
    for input_id in input_ids:
        try:
            sep_token_position = input_id.index(sep_token_id)
        except ValueError:
            sep_token_position = -1
        
        if sep_token_position != -1:
            label = [-100] * (sep_token_position + 1) + input_id[(sep_token_position + 1):]
        else:
            label = [-100] * len(input_id) 

        labels.append(label)

    model_inputs["labels"] = labels
    
    return model_inputs

train_tokenized_datasets = train_hf_dataset.map(tokenize_function, batched=True)
dev_tokenized_datasets = dev_hf_dataset.map(tokenize_function, batched=True)
test_tokenized_datasets = test_hf_dataset.map(tokenize_function, batched=True)

accelerator = Accelerator(mixed_precision='bf16', cpu=False)

class CustomDataCollator:
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [example['input_ids'] for example in examples]
        labels = [example['labels'] for example in examples]
        attention_mask = [example['attention_mask'] for example in examples]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}

data_collator = CustomDataCollator()


# Custom DataLoader with detailed batch printing
def create_dataloader(dataset, batch_size, sampler):
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=data_collator)
    for i, batch in enumerate(dataloader):
        print(f"Batch {i} contents:")
        for key, value in batch.items():
            print(f"{key}: {value.shape}")
        if i == 0:  # Only print the first batch for debugging
            break
    return dataloader

train_dataloader = create_dataloader(train_tokenized_datasets, batch_size=4, sampler=SequentialSampler(train_tokenized_datasets))
dev_dataloader = create_dataloader(dev_tokenized_datasets, batch_size=4, sampler=SequentialSampler(dev_tokenized_datasets))


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, enable_wrap, wrap

def apply_fsdp(model, min_num_params=1e7):
    def fsdp_wrap_policy(module, recurse, nonwrapped_numel):
        return size_based_auto_wrap_policy(module, recurse, nonwrapped_numel, min_num_params)
    with enable_wrap(wrapper_cls=FSDP, auto_wrap_policy=fsdp_wrap_policy):
        model = wrap(model)
    return model


device = accelerator.device
model.to(device)

model = apply_fsdp(model)
optimizer = AdamW(model.parameters(), lr=1e-5)


model, optimizer, train_dataloader, dev_dataloader = accelerator.prepare(
    model,
    optimizer,
    train_dataloader,
    dev_dataloader,
)


num_epochs = 8
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


output_dir = "./llama3-8b-results"
os.makedirs(output_dir, exist_ok=True)

# Helper function to manage checkpoints
def manage_checkpoints(output_dir, max_checkpoints=2):
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    if len(checkpoints) > max_checkpoints:
        for ckpt in checkpoints[max_checkpoints:]:
            print(f"Removing old checkpoint {ckpt}")
            os.system(f"rm -rf {ckpt}")

# Training loop with logging, saving model, and progress bar
global_step = 0
save_every = 2500 


for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    model.train()
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Training Epoch {epoch + 1}")
    
    for step, batch in progress_bar:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        progress_bar.set_postfix(loss=loss.item())
        
        if global_step % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}")
            os.makedirs(checkpoint_path, exist_ok=True)  # Ensure the directory exists
    
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

            
            print(f"Saved checkpoint at step {global_step} to {checkpoint_path}")
            manage_checkpoints(output_dir, max_checkpoints=2)