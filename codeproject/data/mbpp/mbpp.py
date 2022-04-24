import torch 

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
                pad_token_id=pad_token_id
                )

        return full_tokenized 

    def __len__(self): 
        return len(self.data)
