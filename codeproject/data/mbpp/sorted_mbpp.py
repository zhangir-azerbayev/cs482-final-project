import json
import torch
import re

def sort_mbpp(): 
    with open('./raw_mbpp.jsonl') as f: 
        mbpp = [json.loads(line) for line in f]
    # There is something wrong with task id 170
    for i,x in enumerate(mbpp): 
        if x["task_id"] in [170, 649]: 
            mbpp.pop(i)

    # adds END token and promp setting
    for i,x in enumerate(mbpp): 
        mbpp[i]["code"] = mbpp[i]["code"] + "\n</cell>"
        
        re_key = "def.*?:"
        if not re.search(re_key, x["code"]): 
            print(repr(x["code"]))
        header = x["code"][:re.search(re_key, x["code"]).span()[1]]
        mbpp[i]["header"] = header

    
    # sorts
    key = lambda x: x["code"].count('\n')

    sorted_mbpp = sorted(mbpp, key=key)

    with open("./sorted_mbpp.json", "w") as f: 
        json.dump(sorted_mbpp, f)

def split_mbpp(): 
    with open('./raw_mbpp.jsonl') as f: 
        mbpp = [json.loads(line) for line in f]
    # adds header
    for i,x in enumerate(mbpp):  
        re_key = "def.*?:"
        if not re.search(re_key, x["code"]): 
            print(repr(x["code"]))
        header = x["code"][:re.search(re_key, x["code"]).span()[1]]
        mbpp[i]["header"] = header

    train = mbpp[:900]
    test = mbpp[900:]

    with open("mbpp_train.json", "w") as f: 
        json.dump(train, f)
    
    with open("mbpp_test.json", "w") as f: 
        json.dump(test, f)



class MBPP(torch.utils.data.Dataset): 
    def __init__(self, instance_list, tokenizer, max_length): 
        self.data = instance_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx): 
        instance = self.data[idx]
        full = instance["text"] + "\n" + instance["code"]

        full_tokenized = self.tokenizer(full, 
                max_length = self.max_length, 
                padding = 'max_length', 
                return_tensors = 'pt',
                truncation=True, 
                )

        return full_tokenized 

    def __len__(self): 
        return len(self.data)

if __name__=="__main__": 
    split_mbpp()
