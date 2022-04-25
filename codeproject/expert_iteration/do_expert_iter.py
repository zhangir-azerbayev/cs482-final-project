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
from transformers import AdamW
from torch.optim.lr_scheduler import LambdaLR
from codeproject.expert_iteration.train_zero import data_collator
from codeproject.expert_iteration.eval_zero import tokens_to_programs
from codeproject.eval_utils import *

# Initializing Data 
with open("../data/mbpp/sorted_mbpp.json") as f: 
    full_data = json.load(f)

id_text_lookup = {}
for x in full_data[500:]: 
    id_text_lookup[x["task_id"]] = x["text"]

def make_training_dataset(id_solved_pairs, tokenizer, max_length=120): 
    data_list = [{"text": id_text_lookup[x["task_id"]], 
                  "code": x["solution"]} for x in id_solved_pairs]

    return MBPP(full_data[:500] + data_list, tokenizer, max_length)


with open("results_eval_zero/01_incoder_checkpoint-500.json") as f: 
    zero_log = json.load(f)

id_solved_pairs = [{k:x[k] for k in ["task_id", "solution"]} for x in zero_log["log"] if x["solution"]]
#####################################################################################
# Training function
def do_train(model, tokenizer, dataset, experiment_name, num_epochs, batch_size, i): 
    steps_per_epoch = math.ceil(len(dataset)/batch_size)

    output_dir = f"./results_train_zero/{experiment_name}/MLE{i}"
    training_args = TrainingArguments(output_dir=output_dir,
                                      num_train_epochs=num_epochs,
                                      per_device_train_batch_size=batch_size,
                                      logging_strategy="epoch",
                                      save_steps=steps_per_epoch*num_epochs,
                                      )

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

    model = Trainer(model=model, args=training_args, train_dataset=dataset, 
                data_collator=data_collator, 
                optimizers=(optimizer, scheduler),
                ).train()

    return model
#############################################################################
def update_id_solved_pairs(model, 
                           tokenizer, 
                           id_solved_pairs
                           ): 

# Main Script
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 4
num_epochs = 2

tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
tokenizer.pad_token = '<|endoftext|>'


