from codeproject.data.mbpp.sorted_mbpp import MBPP
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import json
import sys
from transformers import TrainingArguments, Trainer


run_name = sys.argv[1]

with open("../data/mbpp/sorted_mbpp.json") as f: 
    data_list = json.load(f)
    data_list = data_list[:500]

tokenizer = AutoTokenizer.from_pretrained("shpotes/codegen-350M-mono")

dataset = MBPP(data_list, tokenizer, max_length=120)

model = AutoModelForCausalLM.from_pretrained("shpotes/codegen-350M-mono", 
        trust_remote_code=True, 
        )


output_dir = f"./results_train_zero/{run_name}"
training_args = TrainingArguments(output_dir=output_dir
                                      num_train_epochs=20,
                                      per_device_train_batch_size=24, 
                                      logging_strategy="epoch",
                                      save_strategy="epoch",
                                      )


def data_collator(data):
        return {'input_ids': torch.stack([f["input_ids"].squeeze() for f in data]),
                'attention_mask': torch.stack([f["attention_mask"].squeeze() 
                    for f in data])
                }

Trainer(model=model, args=training_args, train_dataset=dataset, 
            data_collator=data_collator
            ).train()







