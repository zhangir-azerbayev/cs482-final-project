from codeproject.data.mbpp.sorted_mbpp import MBPP
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import json
import sys
from transformers import TrainingArguments, Trainer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import math



run_name = sys.argv[1]

with open("../data/mbpp/sorted_mbpp.json") as f: 
    data_list = json.load(f)
    data_list = data_list[:500]

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

dataset = MBPP(data_list, tokenizer, max_length=120)

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

batch_size=32

steps_per_epoch = math.ceil(500/batch_size)
num_epochs=20

output_dir = f"./results_train_zero/{run_name}"
training_args = TrainingArguments(output_dir=output_dir, 
                                      num_train_epochs=num_epochs,
                                      per_device_train_batch_size=steps_per_epoch, 
                                      logging_strategy="epoch",
                                      save_steps=steps_per_epoch*num_epochs,
                                      )


def data_collator(data):
        return {'input_ids': torch.stack([f["input_ids"].squeeze() for f in data]),
                'attention_mask': torch.stack([f["attention_mask"].squeeze() 
                    for f in data]), 
                'labels': torch.stack([f["input_ids"].squeeze() for f in data])
                }

Trainer(model=model, args=training_args, train_dataset=dataset, 
            data_collator=data_collator
            ).train()







