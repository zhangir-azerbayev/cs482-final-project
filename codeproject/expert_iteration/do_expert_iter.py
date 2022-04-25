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
from torch.utils.data import DataLoader
from tqdm import tqdm

# Main Script
with open("../data/mbpp/sorted_mbpp.json") as f: 
    full_data = json.load(f)

for i,_ in enumerate(full_data): 
    full_data[i]["tests"] = "\n".join(full_data[i]["test_list"])

id_text_lookup = {}
for x in full_data[500:]: 
    id_text_lookup[x["task_id"]] = x["text"]

def make_training_dataset(id_solved_pairs, tokenizer, max_length=120): 
    data_list = [{"text": id_text_lookup[x["task_id"]], 
                  "code": x["solution"]} for x in id_solved_pairs]
    
    added_back = full_data[:500] + data_list

    return MBPP(added_back, tokenizer, max_length)

with open("results_eval_zero/01_incoder_checkpoint-500.json") as f: 
    zero_log = json.load(f)

id_solved_pairs = [{k:x[k] for k in ["task_id", "solution"]} 
        for x in zero_log["log"] if x["solution"]]

# Training function
def do_mle(model, tokenizer, new_train, experiment_name, num_epochs, batch_size, i): 
    steps_per_epoch = math.ceil(len(new_train)/batch_size)

    training_args = TrainingArguments(output_dir=f"./results_train/{experiment_name}/MLE{i}",
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

    Trainer(model=model, args=training_args, train_dataset=new_train, 
                data_collator=data_collator, 
                optimizers=(optimizer, scheduler),
                ).train()

    return model



def update_id_solved_pairs(model, 
                           tokenizer, 
                           id_solved_pairs,
                           experiment_name, 
                           inference_batch_size, 
                           num_return_sequences, 
                           i
                           ): 
    max_prompt_length=100
    max_code_length=250
    print("#"*20)
    print("DOING SAMPLING")
    print("#"*20)
    solved_ids = sorted([x["task_id"] for x in id_solved_pairs])

    eval_set = [x for x in full_data[500:] if x["task_id"] not in solved_ids]
    eval_set = [{k : x[k] for k in ["text", "header", "tests", "task_id"]} 
            for x in eval_set]
    eval_loader = DataLoader(eval_set, batch_size=inference_batch_size, 
            drop_last=False)

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
                                     temperature=.2,
                                     max_new_tokens = max_code_length,
                                     pad_token_id=tokenizer.eos_token_id,
                                     num_return_sequences = num_return_sequences,
                                    )
        outputs = torch.reshape(outputs, (batch_length, 
            num_return_sequences, -1))

        for out, text_length, tests, task_id in zip(outputs, 
                text_lengths, batch["tests"], batch["task_id"]): 
            programs = tokens_to_programs(out, text_length, tokenizer)

            passed_lst = programs_to_passed_lst(programs, tests)

            if True in passed_lst: 
                solution = programs[passed_lst.index(True)]
                id_solved_pairs.append({"task_id": int(task_id.item()), 
                                        "solution": solution})
    print("TOTAL NUMBER SOLVED: ", len(id_solved_pairs))

    with open(f"results_train/{experiment_name}/S_{i}.json", "w") as f: 
        json.dump({"num_solved": len(id_solved_pairs), 
                   "log": id_solved_pairs
                  },  f)
    return id_solved_pairs




experiment_name = sys.argv[1]
os.mkdir(f"results_train/{experiment_name}")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARELLISM"] = "false"
batch_size = 4
num_epochs = 2
number_iterations = 6
num_return_sequences=10
inference_batch_size = 12

tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
tokenizer.pad_token = '<|endoftext|>'

model_path = "facebook/incoder-1B"
model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')

for i in range(1, number_iterations+1): 
    new_train = make_training_dataset(id_solved_pairs, tokenizer)

    model = do_mle(model, 
                     tokenizer, 
                     new_train, 
                     experiment_name, 
                     num_epochs, 
                     batch_size, 
                     i
                     )

    id_solved_pairs = update_id_solved_pairs(model, 
                                             tokenizer, 
                                             id_solved_pairs, 
                                             experiment_name,
                                             inference_batch_size, 
                                             num_return_sequences, 
                                             i
                                             )


