import sys 
import os
import yaml 
import json
import math
import random
import re
import numpy as np

from tqdm import tqdm

import torch 
import torch.nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer 
from torch.utils.data import DataLoader
from tqdm import tqdm

from codeproject.execution import check_correctness
from codeproject.eval_utils import programs_to_passed_lst, pass_at_k

max_prompt_length = 75
max_code_length = 300
temp=0.2
num_return_sequences=10
inference_batch_size=12

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def tokens_to_programs(outs, input_length, tokenizer): 
    l=input_length

    raw_texts = [tokenizer.decode(out) for out in outs]

    progs = [y.replace("<|endoftext|>","").replace("<|","")[l:].split("\n</cell>")[0]
            for y in raw_texts]

    return progs


tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
tokenizer.pad_token = "<|endoftext|>"

experiment_name = sys.argv[1]
model_path = sys.argv[2]

model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

with open("../data/mbpp/sorted_mbpp.json") as f: 
    data_list = json.load(f)

eval_set = data_list[500:]
# prevents a dataloader fit
for i, _ in enumerate(eval_set): 
    eval_set[i]["tests"] = "\n".join(eval_set[i]["test_list"])
# prevents dataloader bug
eval_set = [{k : x[k] for k in ["text", "header", "tests", "task_id"]} for x in eval_set]

eval_loader = DataLoader(eval_set, batch_size=inference_batch_size, drop_last=False)

log = []
for batch in tqdm(eval_loader): 
    batch_length = len(batch["text"])
    text_lengths = [len(x) for x in batch["text"]]

    prompts = [x + "\n" + y for x,y in zip(batch["text"], batch["header"])]

    inputs = tokenizer(prompts, 
                       max_length=max_prompt_length, 
                       padding='max_length', 
                       truncation=True,
                       return_tensors="pt",
                       ).to('cuda')


    outputs = model.generate(**inputs,
                                 do_sample=True,
                                 temperature=temp,
                                 max_new_tokens = max_code_length,
                                 pad_token_id=tokenizer.eos_token_id,
                                 num_return_sequences = num_return_sequences,
                                )
    outputs = torch.reshape(outputs, (batch_length, num_return_sequences, -1)).to('cpu')

    for out, text_length, tests, task_id in zip(outputs, text_lengths, batch["tests"], batch["task_id"]): 
        programs = tokens_to_programs(out, text_length, tokenizer)

        passed_lst = programs_to_passed_lst(programs, tests)

        num_success = sum(passed_lst)
        if num_success > 0: 
            passed=True
            solution = programs[passed_lst.index(True)]
        else: 
            solution = None
            passed=False

        k = num_return_sequences
        pass_1 = pass_at_k(passed_lst, 1)
        pass_k = pass_at_k(passed_lst, k)

        to_log = {"task_id": int(task_id.item()), 
                "pass_1": pass_1, 
                "pass_k": pass_k, 
                "k": k, 
                "num_success": num_success, 
                "passed": passed, 
                "solution": solution, 
                }
        log.append(to_log)

pass_1s = [x["pass_1"] for x in log]
pass_1 = sum(pass_1s)/len(pass_1s)

pass_ks = [x["pass_k"] for x in log]
pass_k = sum(pass_ks)/len(pass_ks)

to_dump = {"pass_1": pass_1, "pass_k": pass_k, "log": log}

with open(f"results_eval_zero/{experiment_name}.json", "w") as f: 
    json.dump(to_dump, f)
