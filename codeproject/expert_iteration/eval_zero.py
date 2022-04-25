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

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer 
max_prompt_length = 75
max_code_length = 300
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
eos = "<|endoftext|>"

tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
tokenizer.pad_token = "<|endoftext|>"

experiment_name = sys.argv[1]
model_path = sys.argv[2]

model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

with open("../data/mbpp/sorted_mbpp.json") as f: 
    data_list = json.load(f)

prompts = [x["text"] + "\n" + x["header"] for x in data_list[500:510]]
text_lengths = [len(x["text"]) for x in data_list[500:510]]
print(prompts)
inputs = tokenizer(prompts, 
                   return_tensors="pt", 
                   padding='max_length', 
                   max_length=max_prompt_length, 
                   truncation=True).to('cuda')

outs = model.generate(**inputs, 
                      do_sample=True, 
                      temperature=0.2, 
                      max_new_tokens=max_code_length, 
                      pad_token_id=tokenizer.eos_token_id, 
                      )

raw_texts = [tokenizer.decode(y) for y in outs]
texts = [y.replace("<|endoftext|>", "").replace("<|", "")[l:].split("\n</cell>")[0]
        for y,l in zip(raw_texts, text_lengths)]

for text in texts: 
    print('#'*20)
    print(text)
