import torch
from torch.data import Dataset


class NerDataset(Dataset):
    """
    Dataset module for loading ner data
    """

    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len
        self.n = len(self.data)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        input_ids = data[idx]["token_ids"]
        ner_tags = data[idx]["ner_tags"]

        # append 0 to pad to max length
        if len(token_ids) >= self.max_len:
            input_ids = input_ids[: self.max_len]
            ner_tags = ner_tags[: self.max_len]
            token_type_ids = [0] * self.max_len
            attention_masks = [1] * self.max_len
        else:
            padding_len = self.max_len - len(input_ids)
            token_type_ids = [0] * self.max_len
            attention_masks = [1] * len(input_ids) + [0] * padding_len
            input_ids = input_ids + [0] * padding_len
            ner_tags = ner_tags + [0] * padding_len

        encoding = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_masks": torch.tensor(attention_masks, dtype=torch.long),
        }

        ner_tags = torch.tensor(ner_tags, dtype=torch.long)
        return encoding, ner_tags

        pass
