import pathlib
import json
import jsonlines
from collections import defaultdict
from tqdm.auto import tqdm

import config

HOME = pathlib.Path.home()
DICTIONARY_PATH = HOME / "Data/conll2003/lstm_data/word2index.json"
TRAIN_PATH = HOME / "Data/conll2003/train.json"
TEST_PATH = HOME / "Data/conll2003/test.json"
DEV_PATH = HOME / "Data/conll2003/dev.json"

TRAIN_PROCESSED_PATH = HOME / "Data/conll2003/lstm_data/train_processed.jsonl"
TEST_PROCESSED_PATH = HOME / "Data/conll2003/lstm_data/test_processed.jsonl"
DEV_PROCESSED_PATH = HOME / "Data/conll2003/lstm_data/dev_processed.jsonl"

MAX_LEN = config.MAX_LEN


def save_to_jsonlines(data, path):
    with jsonlines.open(path, mode="w") as writer:
        writer.write_all(data)


def preprocess(file_path, word2idx):
    transformed_data = []
    
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    for sample in tqdm(data, total=len(data)):
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]
        pos_tags = sample["pos_tags"]
        
        # get token ids
        token_ids = [word2idx.get(w.lower()) for w in tokens]
                            
        # apply padding
        n = len(token_ids)
        if n > MAX_LEN:
            token_ids = token_ids[:MAX_LEN]
            ner_tags = ner_tags[:MAX_LEN]
            pos_tags = ner_tags[:MAX_LEN]
        else:
            token_ids += [1] * (MAX_LEN - n)
            ner_tags += [0] * (MAX_LEN - n)
            pos_tags += [47] * (MAX_LEN -n)
        sample_dict = {
            "token_ids": token_ids,
            "ner_tags": ner_tags,
            "pos_tags": pos_tags
        }
        transformed_data.append(sample_dict)

    return transformed_data


def main():
    # load dictionary
    with open(DICTIONARY_PATH, "r") as f0:
        word2idx = json.load(f0)

    # convert dict to default dict
    word2idx = defaultdict(int, word2idx)

    # process train data
    print("Preprocessing train data...")
    train_data = preprocess(TRAIN_PATH, word2idx)

    # save train data
    save_to_jsonlines(train_data, TRAIN_PROCESSED_PATH)

    # process test data
    print("Preprocessing test data...")
    test_data = preprocess(TEST_PATH, word2idx)

    # save test data
    save_to_jsonlines(test_data, TEST_PROCESSED_PATH)

    # process dev data
    print("Preprocessing dev data...")
    dev_data = preprocess(DEV_PATH, word2idx)

    # save dev data
    save_to_jsonlines(dev_data, DEV_PROCESSED_PATH)


if __name__ == "__main__":
    main()
