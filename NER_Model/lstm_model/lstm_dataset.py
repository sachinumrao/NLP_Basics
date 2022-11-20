import jsonlines
import torch
from torch.utils.data import Dataset


class LSTMDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.jsonl_datareader(data_path)
        self.n = len(self.data)

    def jsonl_datareader(self, path):
        data = []

        with jsonlines.open(path, mode="r") as reader:
            for line in reader:
                data.append(line)

        return data

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        sample = self.data[index]

        token_ids = sample["token_ids"]
        targets = sample["ner_tags"]

        token_ids = torch.tensor(token_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)

        return (token_ids, targets)
