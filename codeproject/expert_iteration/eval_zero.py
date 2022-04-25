import sys 
import os
import yaml 
import json
import math
import random
import re

from tqdm import tqdm

import torch 
import torch.nn

from transformers import GPTNeoForCausalLM, GPT2Tokenizer 
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

experiment_name = sys.argv[1]
model_path = sys.argv[2]

model = GPTNeoForCausalLM.from_pretrained(model_path).to("cuda")

with open("../data/mbpp/sorted_mbpp.json") as f: 
    data_list = json.load(f)

codes = [x["text"] for x in data_list[500:510]]
print(codes)

inputs = tokenizer(codes, 
                   return_tensors="pt", 
                   padding='max_length').to('cuda')

outs = model.generate(**inputs, 
                      do_sample=True, 
                      temperature=0.2, 
                      max_new_tokens=300, 
                      pad_token_id=tokenizer.eos_token_id, 
                      )

texts = tokenizer.decode(outs)

print(texts)
             
