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
from torch.utils.data import DataLoader
from tqdm import tqdm

max_prompt_length = 75
max_code_length = 300
temp=0.2
num_return_sequences=5
inference_batch_size=2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def tokens_to_programs(outs, input_length, tokenizer): 
    l=input_length

    raw_texts = [tokenizer.decode(out) for out in outs]

    progs = [y.replace("<|endoftext|>","").replace("<|","")[l:].split("\n</cell>")[0]
            for y in raw_texts]

    return progs

def programs_to_passed_lst(programs, test_cases): 
    test_cases_str = "\n".join(test_cases)

    to_execute = programs + "\n" + test_cases_str

    passed_lst = [check_correctness(x, 1)["passed"] for x in to_execute]

    return passed_lst

def pass_k(lst, k): 
    """
    lst: Boolean list 
    k: value of pass@k to calculate. 
    """
    n = len(lst)
    c = sum(lst)
    if n - c < k: return 1.0 
    return 1.0 - np.prod(1.0 - k / np.arange(n-c+1, n+1))

tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
tokenizer.pad_token = "<|endoftext|>"

experiment_name = sys.argv[1]
model_path = sys.argv[2]

model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

with open("../data/mbpp/sorted_mbpp.json") as f: 
    data_list = json.load(f)

eval_set = data_list[500:]

eval_loader = DataLoader(eval_set, batch_size=inference_batch_size, drop_last=False)

for batch in eval_loader: 
    batch_length = len(batch["text"])
    text_lengths = [len(x) for x in batch["text"]]

    prompts = [x + "\n" + y for x,y in zip(batch["text"], batch["header"])]

    inputs = tokenizer(prompts, 
                       max_length=max_prompt_length, 
                       padding='max_length', 
                       truncation=True,
                       )

    outputs = model.generate(**inputs,
                                 do_sample=True,
                                 temperature=temp,
                                 max_new_tokens = max_code_length,
                                 pad_token_id=tokenizer.eos_token_id,
                                 num_return_sequences = num_return_sequences,
                                )
    outputs = torch.reshape(outputs, (batch_length, num_return_sequences, -1))

    for out, text_length, test_cases in zip(outputs, text_lengths, batch["test_cases"]): 
        programs = tokens_to_programs(out, text_length, tokenizer)

        passed_lst = programs_to_passed_lst(programs, test_cases)

        num_success = sum(passed_lst)
        k = num_return_sequences

        for prog in programs: 
            print("#"*20)
            print(program)
    break



