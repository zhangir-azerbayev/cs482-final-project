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
from codeproject.expert_iteration.utils import tokens_to_programs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

few_shot = int(sys.argv[1])
model_path = sys.argv[2]
outfile = sys.argv[3]

max_prompt_length = 300
max_code_length = 100
num_return_sequences=10
inference_batch_size = 10

tokenizer = AutoTokenizer.from_pretrained('facebook/incoder-1B')
tokenizer.pad_token = '<|endoftext|>'
model = AutoModelForCausalLM.from_pretrained('facebook/incoder-1B').to('cuda')

with open("../data/mbpp/few_shot_prompt.txt") as f: 
    few_shot_prompt = f.read()

with open("../data/mbpp/mbpp_test.json") as f: 
    eval_set = json.load(f)

# two modifications prevent dataloader bugs
for i, _ in enumerate(eval_set): 
    eval_set[i]["tests"] = "\n".join(eval_set[i]["test_list"])
eval_set = [{k : x[k] for k in ["text", "header", "tests", "task_id"]} 
        for x in eval_set]


eval_loader = DataLoader(eval_set, batch_size=inference_batch_size, 
            drop_last=False)

pass_ks = []
for batch in tqdm(eval_loader): 
    batch_length = len(batch["text"])
    if few_shot: 
        text_lengths = [len(x) for x in batch["text"]]
    else: 
        text_lengths = [len(few_shot_prompt+x) for x in batch["text"]]

    if few_shot: 
        prompts = [few_shot_prompt + x + "\n" + y 
                for x,y in zip(batch["text"], batch["header"])]
    else: 
        prompts = [x + "\n" + y for x,y in zip(batch["text"], batch["header"])]

    inputs = tokenizer(prompts, 
                       max_length=max_prompt_length, 
                       padding='max_length', 
                       truncation=True,
                       return_tensors="pt",
                       ).to('cuda')


    outputs = model.generate(**inputs,
                                 do_sample=True,
                                 temperature=.2,
                                 max_new_tokens = max_code_length,
                                 pad_token_id=tokenizer.eos_token_id,
                                 num_return_sequences = num_return_sequences,
                                )
    outputs = torch.reshape(outputs, (batch_length, 
        num_return_sequences, -1))

    for out, text_length, tests, task_id, head in zip(outputs, 
            text_lengths, batch["tests"], batch["task_id"], batch["header"]): 
        decoded = [tokenizer.decode(x) for x in out]
        lefts = [x[x.index(head):] for x in decoded]
        programs = [x.replace("<|endoftext|>", "").replace("<|", "").split("</cell>")[0] for x in lefts]

        passed_lst = programs_to_passed_lst(programs, tests)


        if True in passed_lst: 
            pass_ks.append(1)
        else: 
            pass_ks.append(0)

    print(sum(pass_ks)/len(pass_ks))

pass_k = sum(pass_ks)/len(pass_ks)

with open(outfile, "w") as f: 
    json.dump([pass_k, num_return_sequences], f)
