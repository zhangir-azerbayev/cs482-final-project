from codeproject.data.mbpp.sorted_mbpp import MBPP
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn
from transformers.trainer_pt_utils import get_parameter_names
import json
import sys
from transformers import TrainingArguments, Trainer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import os
from torch.optim.lr_scheduler import LambdaLR
from codeproject.expert_iteration.utils import data_collator, tokens_to_programs
from codeproject.eval_utils import programs_to_passed_lst
from torch.optim import AdamW

tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
tokenizer.pad_token = '<|endoftext|>'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
run_name = sys.argv[1]

# Making Dataset
with open("../data/mbpp/mbpp_train.json") as f: 
    full_data = json.load(f)


id_text_lookup = {}
for x in full_data: 
    id_text_lookup[x["task_id"]] = x["text"]

def make_training_dataset(id_solved_pairs, tokenizer, max_length=300): 
    data_list = [{"text": id_text_lookup[x["task_id"]], 
                  "code": x["solution"]} for x in id_solved_pairs
                  if x["solution"]]

    return MBPP(data_list, tokenizer, max_length)

with open("mbpp_codex.json") as f: 
    codex_out = json.load(f)

dataset = make_training_dataset(codex_out["log"], tokenizer)

model = AutoModelForCausalLM.from_pretrained("facebook/incoder-1B")

batch_size=4

steps_per_epoch = math.ceil(len(dataset)/(2*batch_size))
num_epochs=10

output_dir = f"./results_train/{run_name}"
training_args = TrainingArguments(output_dir=output_dir, 
                                      num_train_epochs=num_epochs,
                                      per_device_train_batch_size=batch_size, 
                                      logging_strategy="epoch",
                                      save_steps=steps_per_epoch*num_epochs,
                                      )


def data_collator(data):
    return {'input_ids': torch.stack([f["input_ids"].squeeze()[1:] for f in data]),
            'attention_mask': torch.stack([f["attention_mask"].squeeze()[1:]
                    for f in data]), 
            'labels': torch.stack([f["input_ids"].squeeze()[1:] for f in data])
            }

decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
decay_parameters = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                if n in decay_parameters],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
lr_lambda = lambda step: 1
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

Trainer(model=model, args=training_args, train_dataset=dataset, 
            data_collator=data_collator, 
            optimizers=(optimizer, scheduler),
            ).train()
