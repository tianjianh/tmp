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

test_dataset = XSumDataset(data_dir, split_file, 'test')

path = "llama3-8b-results/checkpoint-20000"
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

# train_hf_dataset = convert_to_hf_dataset(train_dataset)
# dev_hf_dataset = convert_to_hf_dataset(dev_dataset)
test_hf_dataset = convert_to_hf_dataset(test_dataset)



def tokenize_function_test(examples):
    questions = examples['document']
    answers = examples['summary']

    combined_sequences = []
    for question in questions:
        question = question if question is not None else ""

        combined_sequence = question + " " + sep_token + " "
        combined_sequences.append(combined_sequence)

    model_inputs = tokenizer(combined_sequences, max_length=512, truncation=True, padding='max_length')

    labels = tokenizer(answers, max_length=256, truncation=True, padding='max_length')['input_ids']

    model_inputs["labels"] = labels
    
    return model_inputs


test_tokenized_datasets = test_hf_dataset.map(tokenize_function_test, batched=True)

# accelerator = Accelerator(mixed_precision='bf16', cpu=False)

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
    # for i, batch in enumerate(dataloader):
    #     print(f"Batch {i} contents:")
    #     for key, value in batch.items():
    #         print(f"{key}: {value.shape}")
    #     if i == 0:  # Only print the first batch for debugging
    #         break
    return dataloader

# train_dataloader = create_dataloader(train_tokenized_datasets, batch_size=4, sampler=SequentialSampler(train_tokenized_datasets))
# dev_dataloader = create_dataloader(dev_tokenized_datasets, batch_size=4, sampler=SequentialSampler(dev_tokenized_datasets))
test_dataloader = create_dataloader(test_tokenized_datasets, batch_size=4, sampler=SequentialSampler(test_tokenized_datasets))


from datasets import load_metric

# Load the metrics
rouge_metric = load_metric("rouge")

device = torch.device("cuda")

def evaluate_model(model, dataloader, tokenizer, device):
    
    model = model.to(device)
    model.eval()
    predictions = []
    references = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1024)
            input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            reference_texts = tokenizer.batch_decode(labels, skip_special_tokens=False)


            # generated_texts = [g.replace(i, "") for i, g in zip(input_texts, generated_texts)]

            # for t in generated_texts:
            #     print(t)
            #     print()

            for i, g in zip(input_texts, generated_texts):
                print(i)
                print("/" * 100)
                print(g)
                print()



            # print(generated_texts)
            print("-" * 100)
            for t in reference_texts:
                print(t)
                print()
            exit(0)

        predictions.extend(generated_texts)
        references.extend(reference_texts)
    
    # Calculate metrics
    rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
    # bart_scores = bartscore_metric.compute(predictions=predictions, references=references)
    # align_scores = alignscore_metric.compute(predictions=predictions, references=references)
    
    return rouge_scores

rouge_scores = evaluate_model(model, test_dataloader, tokenizer, device)

# Print the results
print(f"ROUGE Scores: {rouge_scores}")






