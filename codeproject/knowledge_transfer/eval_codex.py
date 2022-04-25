from cl.data.dataset import read_gsm8k
from cl.execution import semisafe_evaluate
from tqdm import tqdm
import re
import random
import json
import openai
from ratelimit import limits, sleep_and_retry
from codeproject.eval_utils import programs_to_passed_lst, pass_at_k

@sleep_and_retry
@limits(calls=4, period=60)
def call_api(engine, prompt, max_tokens, n, temperature):
    return openai.Completion.create(engine=engine,
            prompt=prompt, max_tokens=max_tokens, n=n,
            temperature=temperature)

random.seed(20)
k = 5
temp = 0.2

with open("../data/mbpp/few_shot_prompt.txt") as f: 
    few_shot_prompt = f.read()

with open("../data/mbpp/mbpp_train.json") as f: 
    train_list = json.load(f)

log = []
for instance in tqdm(train_list): 
    prompt = few_shot_prompt + instance["text"] + '\n' + instance["header"]

    outputs = call_api(engine="code-davinci-002", 
                       prompt=prompt, 
                       max_tokens=200, 
                       n=k, 
                       temperature=0.2, 
                       )

    outputs = [x["text"].split("[END]")[0] for x in outputs["choices"]]

    programs = [instance["header"] + "\n" + x for x in outputs]

    test_cases = "\n".join(instance["test_list"])

    passed_lst = programs_to_passed_lst(programs, test_cases)
    
    num_successes = sum(passed_lst)

    if num_successes > 0: 
        solution = programs[passed_lst.index(True)]
        passed=True
    else: 
        solution = None
        passed = False

    pass_1 = pass_at_k(passed_lst, 1)
    pass_k = pass_at_k(passed_lst, k)

    to_log = {"task_id": instance["task_id"], 
              "passed": passed, 
              "solution": solution, 
              "pass_1": pass_1,
              "pass_k": pass_k, 
              "num_successes": num_successes
              }
    print(to_log)
    log.append(to_log)


pass_1s = [x["pass_1"] for x in log]
pass_ks = [x["pass_k"] for x in log]

pass_1 = sum(pass_1s)/len(pass_1s)
pass_k = sum(pass_ks)/len(pass_ks)

to_dump = {"pass_1": pass_1, 
           "pass_k": pass_k, 
           "log": log
           }

with open("mbpp_codex.json", "w") as f:
    json.dump(to_dump, f)


