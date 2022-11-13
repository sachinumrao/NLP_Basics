import json
import os

import jsonlines
from tqdm.auto import tqdm
from transformers import BertTokenizer

import ner_config

data_folder = "/home/sachin/Data/conll2003/"


def save_to_jsonlines(data, filepath):
    """
    Save data to jsonlines format
    """
    with jsonlines.open(filepath, mode="w") as writer:
        writer.write_all(data)


def preprocess(file_path, tokenizer):
    """
    Preprocessing function to generate IOB format data
    compatible with given tokenizer.
    """

    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    data_processed = []
    for sample in tqdm(data):
        token_ids = []
        tag_ids = []
        for tok, ner_tag in zip(sample["tokens"], sample["ner_tags"]):
            encoding = tokenizer.encode(tok)
            n = len(encoding) - 2
            tags = [ner_tag] + [ner_tag + 1 if ner_tag % 2 == 1 else ner_tag] * (n - 1)

            token_ids += encoding[1:-1]
            tag_ids += tags

        token_ids = [101] + token_ids + [102]
        tag_ids = [0] + tag_ids + [0]

        data_processed.append(
            {"id": sample["id"], "token_ids": token_ids, "tag_ids": tag_ids}
        )

    return data_processed


def main():
    tokenizer = BertTokenizer.from_pretrained(ner_config.BASE_MODEL)

    # preprocess training data
    train_file_path = os.path.join(data_folder, "train.json")
    train_processed = preprocess(train_file_path, tokenizer)
    save_to_jsonlines(
        train_processed, os.path.join(data_folder, "train_processed.jsonl")
    )

    # preprocess test data
    test_file_path = os.path.join(data_folder, "test.json")
    test_processed = preprocess(test_file_path, tokenizer)
    save_to_jsonlines(test_processed, os.path.join(data_folder, "test_processed.jsonl"))

    # preprocess validation data
    val_file_path = os.path.join(data_folder, "dev.json")
    val_processed = preprocess(val_file_path, tokenizer)
    save_to_jsonlines(val_processed, os.path.join(data_folder, "dev_processed.jsonl"))


if __name__ == "__main__":
    main()
