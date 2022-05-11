import json
from tqdm import tqdm
from pathlib import Path

def read_data(path : Path, file : str, tokenizer, args):
    path = path / file
    title, maintext, ids = [], [], []
    with open(path) as f:
        for line in tqdm(f, desc=f"Reading {file}"):
            line = json.loads(line)
            title.append(line['title'])
            maintext.append(line['maintext'])
            ids.append(line['id'])

    model_inputs = tokenizer(
        maintext,
        max_length=args.max_input_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            title,
            max_length=args.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ) 
    labels['input_ids'] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels['input_ids']
    ]
    return {
        'labels' : labels,
        'model_inputs' : model_inputs,
        'ids' : ids,
    }
