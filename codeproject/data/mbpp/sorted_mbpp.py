import json
import torch

def sort_mbpp(): 
    with open('./raw_mbpp.jsonl') as f: 
        mbpp = [json.loads(line) for line in f]
    # There is something wrong with task id 170
    for i,x in enumerate(mbpp): 
        if x["task_id"] in [170, 649]: 
            mbpp.pop(i)

    # adds END token
    for i,_ in enumerate(mbpp): 
        mbpp[i]["code"] = mbpp[i]["code"] + "\nEND"
    
    # sorts
    key = lambda x: x["code"].count('\n')

    sorted_mbpp = sorted(mbpp, key=key)

    with open("./sorted_mbpp.json", "w") as f: 
        json.dump(sorted_mbpp, f)



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
    sort_mbpp()
