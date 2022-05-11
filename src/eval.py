import json
from pathlib import Path
import argparse
from tw_rouge import get_rouge

def evaluate_rouge(reference, submission):
    refs, preds = {}, {}

    with open(reference) as file:
        for line in file:
            line = json.loads(line)
            refs[line['id']] = line['title'].strip() + '\n'

    if isinstance(submission, Path):
        with open(args.submission) as file:
            for line in file:
                line = json.loads(line)
                preds[line['id']] = line['title'].strip() + '\n'
    elif isinstance(submission, list):
        for sub in submission:
            preds[sub[0]] = sub[1].strip() + '\n'
    else:
        raise NotImplementedError

    keys =  refs.keys()
    refs = [refs[key] for key in keys]
    preds = [preds[key] for key in keys]

    print(json.dumps(get_rouge(preds, refs), indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference')
    parser.add_argument('-s', '--submission')
    args = parser.parse_args()
    evaluate_rouge(args.reference, args.submission)
