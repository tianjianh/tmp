import json
import pandas as pd
import os
import random
from collections import defaultdict
# from torch.optim import AdamW
import pandas as pd
from datasets import Dataset
from transformers import AdamW, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq,TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Subset, RandomSampler, SequentialSampler
import numpy as np
from tqdm import tqdm
import torch

from accelerate import Accelerator
from typing import Any, Dict, List

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Define pools for each slot value
# slot_value_pools = {
#     "attraction-name": ["Museum", "Art Gallery", "Zoo"],
#     "hotel-name": ["Hilton", "Marriott", "Holiday Inn"],
#     "restaurant-name": ["McDonald's", "Burger King", "Subway"],
#     "bus-departure": ["Station A", "Station B", "Station C"],
#     "bus-destination": ["Station X", "Station Y", "Station Z"],
#     "taxi-departure": ["Cambridge", "Oxford", "Manchester"],
#     "taxi-destination": ["London", "Birmingham", "New York"],
#     "train-departure": ["Station 1", "Station 2", "Station 3"],
#     "train-destination": ["Station A", "Station B", "Station C"]
# }




class CustomDataCollator:
    def __call__(self, examples):
        input_ids = [example['input_ids'] for example in examples]
        labels = [example['labels'] for example in examples]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {'input_ids': input_ids, 'labels': labels}


def load_multiwoz_data(data_dir):
    dialogues = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r') as file:
                dialogues.extend(json.load(file))
    return dialogues


def extract_slot_values_from_dir(dataset_dir):
    slot_pools = defaultdict(set)

    # Iterate through all files in the dataset directory
    for file_name in os.listdir(dataset_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(dataset_dir, file_name)
            print(f"Processing file: {file_path}")  # Logging file being processed
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    for dialogue in data:
                        if 'turns' in dialogue:
                            for turn in dialogue['turns']:
                                if 'frames' in turn:
                                    for frame in turn['frames']:
                                        if 'state' in frame and 'slot_values' in frame['state']:
                                            for domain_slot_type, slot_values in frame['state']['slot_values'].items():
                                                for slot_value in slot_values:
                                                    slot_pools[domain_slot_type].add(slot_value.lower())
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {file_path}: {e}")
                except KeyError as e:
                    print(f"Missing expected key {e} in file {file_path}")

    # Convert sets to lists for easier handling later
    for slot in slot_pools:
        slot_pools[slot] = list(slot_pools[slot])
    
    return slot_pools



# def get_curr_belief_states(turn):
#         belief_states = []
#         for frame in turn['frames']:
#             if 'state' in frame:
#                 if 'slot_values' in frame['state']:
#                     for domain_slot_type in frame['state']['slot_values']:
#                         for slot_value in frame['state']['slot_values'][domain_slot_type]:
#                             domain, slot_type = domain_slot_type.split("-")
#                             belief_state = f"{domain} {slot_type} {slot_value.lower()}"
#                             belief_states.append(belief_state)
#         return belief_states

# def process_dialogue(dialog):

#     examples = []
#     context = []
#     for turn in dialog['turns']:
#         curr_turn = turn['utterance'].lower()
#         curr_speaker = "<user>" if turn['speaker'] == 'USER' else "<system>"
#         curr_context = f"{curr_speaker} {curr_turn}"
#         context.append(curr_context)
#         cum_belief_states = get_curr_belief_states(turn)
#         if curr_speaker == "<user>":
#             examples.append({
#                 'dialogue_id': dialog['dialogue_id'],
#                 'turn_num': turn['turn_id'],
#                 'text': " ".join(context),
#                 'labels': ", ".join(set(cum_belief_states))
#             })

#     return examples




def load_city_names(filename):
    with open(filename, 'r') as file:
        cities = file.read().splitlines()
    return cities




sgd_data_dir = 'dstc8-schema-guided-dialogue/train'

# import random
slot_pools_full = extract_slot_values_from_dir(sgd_data_dir)

# # Define pools for each specific slot value
# slot_pools = {
#     # "attraction-name": ["attraction1", "attraction2", "attraction3"],
#     "hotel-name": ["hotel1", "hotel2", "hotel3"],
#     "restaurant-name": ["restaurant1", "restaurant2", "restaurant3"],
#     # "bus-departure": ["bus_departure1", "bus_departure2", "bus_departure3"],
#     # "bus-destination": ["bus_destination1", "bus_destination2", "bus_destination3"],
#     # "taxi-departure": ["taxi_departure1", "taxi_departure2", "taxi_departure3"],
#     # "taxi-destination": ["taxi_destination1", "taxi_destination2", "taxi_destination3"],
#     # "train-departure": ["train_departure1", "train_departure2", "train_departure3"],
#     # "train-destination": ["train_destination1", "train_destination2", "train_destination3"]
# }

slot_pools = {
    # "attraction-name": slot_pools_full.get("attraction_name", []),
    "hotel-name": slot_pools_full.get("hotel_name", []),
    "restaurant-name": slot_pools_full.get("restaurant_name", []),
    # "bus-departure": slot_pools_full.get("bus_departure", []),
    # "bus-destination": slot_pools_full.get("bus_destination", []),
    # "taxi-departure": slot_pools_full.get("taxi_departure", []),
    # "taxi-destination": slot_pools_full.get("taxi_destination", []),
    # "train-departure": slot_pools_full.get("train_departure", []),
    # "train-destination": slot_pools_full.get("train_destination", [])
}

city_names = load_city_names("cities.txt")
town_names = load_city_names("small_towns.txt")

slot_pools["bus-departure"] = city_names
slot_pools["bus-destination"] = city_names
slot_pools["train-departure"] = city_names
slot_pools["train-destination"] = city_names
slot_pools["taxi-departure"] = town_names
slot_pools["taxi-destination"] = town_names


# def replace_slot_values(turn, slot_pools):
#     replaced_slots = {}
#     altered_text = turn['utterance'].lower()
    
#     for frame in turn['frames']:
#         if 'state' in frame and 'slot_values' in frame['state']:
#             for domain_slot_type in frame['state']['slot_values']:
#                 domain, slot_type = domain_slot_type.split("-")
#                 if domain_slot_type in slot_pools:
#                     for slot_value in frame['state']['slot_values'][domain_slot_type]:
#                         if slot_value.lower() in altered_text:
#                             replacement = random.choice(slot_pools[domain_slot_type])
#                             altered_text = altered_text.replace(slot_value.lower(), replacement)
#                             replaced_slots[f"{domain} {slot_type}"] = replacement
    
#     return replaced_slots, altered_text

def get_curr_belief_states(turn):
    belief_states = []
    altered_mapping = {}
    for frame in turn['frames']:
        if 'state' in frame:
            if 'slot_values' in frame['state']:
                for domain_slot_type in frame['state']['slot_values']:
                    for slot_value in frame['state']['slot_values'][domain_slot_type]:
                        domain, slot_type = domain_slot_type.split("-")
                        belief_state = f"{domain} {slot_type} {slot_value.lower()}"
                        belief_states.append(belief_state)
                        if domain_slot_type in slot_pools:
                            replacement = random.choice(slot_pools[domain_slot_type]).lower()
                            altered_mapping[slot_value.lower()] = replacement

    return belief_states, altered_mapping

def process_dialogue(dialog):
    examples = []
    context = []
    
    for turn in dialog['turns']:
        curr_turn = turn['utterance'].lower()
        curr_speaker = "<user>" if turn['speaker'] == 'USER' else "<system>"
        curr_context = f"{curr_speaker} {curr_turn}"
        context.append(curr_context)
        
        cum_belief_states, altered_mapping = get_curr_belief_states(turn)
        
        cum_belief_states = list(set(cum_belief_states))
        # replaced_slots, altered_text = replace_slot_values(turn, slot_pools)
        
        if curr_speaker == "<user>":
            example = {
                'dialogue_id': dialog['dialogue_id'],
                'turn_num': turn['turn_id'],
                'text': " ".join(context),
                'labels': ", ".join(cum_belief_states),
                # 'text_altered': " ".join(context).replace(curr_turn, altered_text),
                # 'labels_altered': ", ".join([f"{k} {v}" for k, v in replaced_slots.items()])
            }

            text_altered = ""
            for tmp_context in context[: -1]:
                for original, replacement in altered_mapping.items():
                    tmp_context = tmp_context.replace(original, replacement)

                text_altered += tmp_context + " "

            tmp_context = context[-1]
            for original, replacement in altered_mapping.items():
                tmp_context = tmp_context.replace(original, replacement)

            text_altered += tmp_context

            example["text_altered"] = text_altered

            # print(cum_belief_states)

            if example["labels"] != "":

                labels_altered = ""
                for cum_belief_state in cum_belief_states[: -1]:
                    for original, replacement in altered_mapping.items():
                        cum_belief_state = cum_belief_state.replace(original, replacement)

                    labels_altered += cum_belief_state + ", "

                cum_belief_state = cum_belief_states[-1]
                for original, replacement in altered_mapping.items():
                    cum_belief_state = cum_belief_state.replace(original, replacement)

                labels_altered += cum_belief_state

                example["labels_altered"] = labels_altered

            else:
                
                example["labels_altered"] = ""  

            examples.append(example)
    
    return examples







    # processed_data = []
    # dialogue_id = dialogue['dialogue_id']
    # history = []
    # turns = dialogue['turns']
    
    # for turn in turns:
    #     speaker = turn["speaker"]
    #     utterance = turn["utterance"].lower()
    #     history.append(f"<{speaker.lower()}> {utterance}")
        
    #     if speaker == "USER":
    #         slot_values = {}
    #         altered_utterance = utterance
    #         for frame in turn['frames']:
    #             if 'state' in frame and 'slot_values' in frame['state']:
    #                 slots = frame['state']['slot_values']
    #                 for slot, values in slots.items():
    #                     if slot in slot_value_pools:
    #                         for value in values:
    #                             random_value = random.choice(slot_value_pools[slot])
    #                             altered_utterance = altered_utterance.replace(value.lower(), random_value.lower())
    #                             if slot not in slot_values:
    #                                 slot_values[slot] = values
    #                             else:
    #                                 slot_values[slot].extend(values)
            
    #         if slot_values:
    #             labels = ', '.join([f"{slot.replace('-', ' ')} {', '.join(values)}" for slot, values in slot_values.items()])
    #             altered_labels = ', '.join([f"{slot.replace('-', ' ')} {random.choice(slot_value_pools[slot])}" for slot, values in slot_values.items() if slot in slot_value_pools])
    #             processed_data.append({
    #                 'dialogue_id': dialogue_id,
    #                 'turn_num': turn['turn_id'],
    #                 'text': ' '.join(history),
    #                 'labels': labels,
    #                 'altered_text': ' '.join(history[:-1]) + f" <user> {altered_utterance}",
    #                 'altered_labels': altered_labels
    #             })
    
    # return processed_data

def process_data(data_dir, split):
    processed_dataset = []
    split_dir = os.path.join(data_dir, split)
    dialogues = load_multiwoz_data(split_dir)
    for dialogue in dialogues:
        processed_dialogue = process_dialogue(dialogue)
        processed_dataset.extend(processed_dialogue)
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_dataset)
    return df

# Define the directory where your MultiWOZ 2.2 JSON files are stored
data_dir = 'MultiWOZ_2.2'


# Process the data for each split
train_df = process_data(data_dir, 'train')
dev_df = process_data(data_dir, 'dev')
test_df = process_data(data_dir, 'test')

# Save to CSV if needed
train_df.to_csv('processed_multiwoz_train.csv', index=False)
dev_df.to_csv('processed_multiwoz_dev.csv', index=False)
test_df.to_csv('processed_multiwoz_test.csv', index=False)

# Convert the DataFrame to a Hugging Face Dataset
# dataset = Dataset.from_pandas(df)

# Split the dataset into train, dev, and test
# Load your datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dev_dataset = Dataset.from_pandas(dev_df)

# Print initial dataset sizes
print(f"Initial train dataset size: {len(train_dataset)}")
print(f"Initial dev dataset size: {len(dev_dataset)}")
print(f"Initial test dataset size: {len(test_dataset)}")

# Load the tokenizer and model
model_name = "facebook/opt-6.7b"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name)



sep_token = "[SEP]"

if sep_token not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({'additional_special_tokens': [sep_token]})
    model.resize_token_embeddings(len(tokenizer))

sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)

# Custom data collator
class CustomDataCollator:
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [example['input_ids'] for example in examples]
        labels = [example['labels'] for example in examples]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {'input_ids': input_ids, 'labels': labels}

data_collator = CustomDataCollator()

# Function to filter out invalid examples
def filter_valid_examples(example):
    question = example['text'] if example['text'] is not None else ""
    answer = example['labels'] if example['labels'] is not None else ""
    # combined_sequence = question + " " + answer
    return question != "" and answer != ""

# Apply the filtering
print("Filtering train dataset...")
train_dataset = train_dataset.filter(filter_valid_examples)
print("Filtering dev dataset...")
dev_dataset = dev_dataset.filter(filter_valid_examples)
print("Filtering test dataset...")
test_dataset = test_dataset.filter(filter_valid_examples)

# Print filtered dataset sizes
print(f"Filtered train dataset size: {len(train_dataset)}")
print(f"Filtered dev dataset size: {len(dev_dataset)}")
print(f"Filtered test dataset size: {len(test_dataset)}")

# Tokenize the inputs and labels
def preprocess_function(examples):
    questions = examples['text']
    answers = examples['labels']

    combined_sequences = []
    for question, answer in zip(questions, answers):
        question = question if question is not None else ""
        answer = answer if answer is not None else ""
        
        combined_sequence = question + " " + sep_token + " " + answer
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

# Apply the preprocessing with debugging
def map_with_debug(dataset, preprocess_function):
    print(f"Dataset size before mapping: {len(dataset)}")
    
    for i in range(min(5, len(dataset))):
        print(f"Sample {i} before mapping: {dataset[i]}")
    
    processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    
    print(f"Dataset size after mapping: {len(processed_dataset)}")
    
    for i in range(min(5, len(processed_dataset))):
        print(f"Sample {i} after mapping: {processed_dataset[i]}")
    
    return processed_dataset

# Apply the preprocessing with debugging
print("Applying preprocessing to train dataset...")
tokenized_train_dataset = map_with_debug(train_dataset, preprocess_function)
print("Applying preprocessing to dev dataset...")
tokenized_dev_dataset = map_with_debug(dev_dataset, preprocess_function)
print("Applying preprocessing to test dataset...")
tokenized_test_dataset = map_with_debug(test_dataset, preprocess_function)

print(f"Tokenized train dataset size: {len(tokenized_train_dataset)}")
print(f"Tokenized dev dataset size: {len(tokenized_dev_dataset)}")
print(f"Tokenized test dataset size: {len(tokenized_test_dataset)}")

print("Sample from tokenized train dataset:", tokenized_train_dataset[0])
print("Sample from tokenized dev dataset:", tokenized_dev_dataset[0])
print("Sample from tokenized test dataset:", tokenized_test_dataset[0])

accelerator = Accelerator(mixed_precision='bf16', cpu=False)


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

# Partition dataset into subsets for each GPU
num_gpus = torch.cuda.device_count()
indices = list(range(len(tokenized_train_dataset)))
partition_size = len(tokenized_train_dataset) // num_gpus
train_partitions = [Subset(tokenized_train_dataset, indices[i * partition_size: (i + 1) * partition_size]) for i in range(num_gpus)]
if len(tokenized_train_dataset) % num_gpus != 0:
    train_partitions[-1] = Subset(tokenized_train_dataset, indices[(num_gpus - 1) * partition_size:])

# train_dataloaders = [create_dataloader(partition, batch_size=8, sampler=RandomSampler(partition)) for partition in train_partitions]
train_dataloaders = create_dataloader(tokenized_train_dataset, batch_size=4, sampler=SequentialSampler(tokenized_train_dataset))
dev_dataloader = create_dataloader(tokenized_dev_dataset, batch_size=4, sampler=SequentialSampler(tokenized_dev_dataset))



from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, enable_wrap, wrap

# # Ensure the correct setup for FSDP
# def apply_fsdp(model, min_num_params=1e8):
#     auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=min_num_params)
#     with enable_wrap(wrapper_cls=FSDP, auto_wrap_policy=auto_wrap_policy):
#         model = wrap(model)
#     return model

# # Apply FSDP to the model
# model = apply_fsdp(model)

def apply_fsdp(model, min_num_params=1e8):
    def fsdp_wrap_policy(module, recurse, nonwrapped_numel):
        return size_based_auto_wrap_policy(module, recurse, nonwrapped_numel, min_num_params)

    with enable_wrap(wrapper_cls=FSDP, auto_wrap_policy=fsdp_wrap_policy):
        model = wrap(model)
    return model

# Apply FSDP to the model


device = accelerator.device
model.to(device)

model = apply_fsdp(model)


model, train_dataloaders, dev_dataloader = accelerator.prepare(
    model,
    train_dataloaders,
    dev_dataloader,
)

# Optimizer and scheduler setup
optimizer = AdamW(model.parameters(), lr=1e-6, weight_decay=1e-5)

# Training loop
num_epochs = 8
global_step = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    model.train()
    # for dataloader in train_dataloaders:
    for step, batch in enumerate(train_dataloaders):
        
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

        if step % 10 == 0:
            print(f"Epoch {epoch + 1} | Step {step} | Loss: {loss.item()}")

    # Evaluation
    model.eval()
    eval_loss = 0
    for step, batch in enumerate(dev_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.item()

    eval_loss /= len(dev_dataloader)
    print(f"Validation Loss after epoch {epoch + 1}: {eval_loss}")

print("Training completed.")
exit(0)

def _extract_slot_from_string(slots_string):
    domains = [
        "attraction",
        "hotel",
        "hospital",
        "restaurant",
        "police",
        "taxi",
        "train",
    ]
    slots_list = []

    str_split = slots_string.strip().split(",")
    if str_split[-1] == "":
        str_split = str_split[:-1]
    str_split = [slot.strip() for slot in str_split]

    for slot_ in str_split:
        slot = slot_.split()
        if len(slot) > 2 and slot[0] in domains:
            domain = slot[0]
            if slot[1] == "book" and slot[2] in ["day", "time", "people", "stay"]:
                slot_type = slot[1] + " " + slot[2]
                slot_val = " ".join(slot[3:])
            else:
                slot_type = slot[1]
                slot_val = " ".join(slot[2:])
            if not slot_val == "dontcare":
                slots_list.append(domain + "--" + slot_type + "--" + slot_val)
    return slots_list

def compute_jga_batched(model, tokenizer, tokenized_test_dataset, batch_size=8):
    model.eval()
    correct = 0
    total = 0

    model = model.to("cuda")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='max_length', max_length=512, return_tensors="pt")
    test_dataloader = DataLoader(tokenized_test_dataset, batch_size=batch_size, collate_fn=data_collator)

    for batch in tqdm(test_dataloader, desc="Evaluating", dynamic_ncols=True):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64, num_beams=5, early_stopping=True)

        for i in range(len(outputs)):
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            
            # Filter out invalid token IDs before decoding
            valid_label_ids = [token_id for token_id in labels[i] if token_id != -100]
            ground_truth_slots = tokenizer.decode(valid_label_ids, skip_special_tokens=True)

            # Extract slots
            # print(ground_truth_slots)
            # print(generated_text)
            # print("-" * 10)
            
            # Remove the input part from the generated text
            input_length = len(tokenizer.decode(input_ids[i], skip_special_tokens=True))
            generated_only = generated_text[input_length:].strip()


            print(ground_truth_slots)
            print(generated_only)
            print("-" * 10)
            
            slots_truth = _extract_slot_from_string(ground_truth_slots)
            slots_pred = _extract_slot_from_string(generated_only)

            # Compute JGA
            if set(slots_truth) == set(slots_pred):
                correct += 1
            total += 1

        # Update the progress bar with current JGA
        tqdm.write(f"Current JGA: {correct / total:.4f}")

    return correct / total



# Load the tokenized test dataset
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

# Compute JGA in batches with a progress bar
jga = compute_jga_batched(model, tokenizer, tokenized_test_dataset, batch_size=8)
print(f"Final Joint Goal Accuracy (JGA): {jga:.4f}")