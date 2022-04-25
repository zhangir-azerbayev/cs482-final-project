def tokens_to_programs(outs, input_length, tokenizer): 
    l=input_length

    raw_texts = [tokenizer.decode(out) for out in outs]

    progs = [y.replace("<|endoftext|>","").replace("<|","")[l:].split("\n</cell>")[0]
            for y in raw_texts]

    return progs



def data_collator(data):
    return {'input_ids': torch.stack([f["input_ids"].squeeze()[1:] for f in data]),
            'attention_mask': torch.stack([f["attention_mask"].squeeze()[1:]
                    for f in data]), 
            'labels': torch.stack([f["input_ids"].squeeze()[1:] for f in data])
            }
