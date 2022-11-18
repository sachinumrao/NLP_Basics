import os
from accelerate import Accelerator
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import ner_config
from model import BertNer
from ner_dataset import NerDataset
from utils import load_jsonl_data

# from transformers import AdamW, get_linear_schedule_with_warmup
accelearator = Accelerator()


def train_fn(data_loader, model, optimizer):
    model.train()
    final_loss = 0

    # optimize model with accelerate
    model, optimizer, data_loader = accelearator.prepare(model, optimizer, data_loader)
    for data in tqdm(data_loader, total=len(data_loader)):
        # for k, v in data.items():
            # data[k] = v.to(ner_config.DEVICE)

        optimizer.zero_grad()
        loss = model(**data)
        accelearator.backward(loss)

        optimizer.step()
        final_loss += loss.item()

    return final_loss / len(data_loader)


def eval_fn(data_loader, model):
    model.eval()
    final_loss = 0

    model, data_loader = accelearator.prepare(model, data_loader)
    for data in tqdm(data_loader, total=len(data_loader)):
        # for k, v in data.items():
            # data[k] = v.to(ner_config.DEVICE)
        loss = model(**data)
        final_loss += loss.item()
    return final_loss / len(data_loader)


def main():
    train_file_path = os.path.join(ner_config.DATA_FOLDER, "train_processed.jsonl")
    test_file_path = os.path.join(ner_config.DATA_FOLDER, "test_processed.jsonl")
    val_file_path = os.path.join(ner_config.DATA_FOLDER, "dev_processed.jsonl")

    # load jsonl data files
    train_data = load_jsonl_data(train_file_path)
    val_data = load_jsonl_data(val_file_path)
    test_data = load_jsonl_data(test_file_path)

    # create datasets
    train_dataset = NerDataset(train_data, ner_config.MAX_LEN)
    test_dataset = NerDataset(test_data, ner_config.MAX_LEN)
    val_dataset = NerDataset(val_data, ner_config.MAX_LEN)

    # create data loader
    train_dataloader = DataLoader(
        train_dataset, batch_size=ner_config.TRAIN_BS, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=ner_config.VAL_BS, shuffle=False
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=ner_config.VAL_BS, shuffle=False
    )

    # intantiate model
    model = BertNer(ner_config.NUM_TAGS, ner_config.DROPOUT)
    optimizer = AdamW(model.parameters(), lr=ner_config.LR)

    # training loops
    for i in range(ner_config.NUM_EPOCHS):
        print("Training Epoch: ", i + 1)
        train_loss = train_fn(train_dataloader, model, optimizer)
        print(train_loss)

        print("Evaluating Epoch: ", i + 1)
        val_loss = eval_fn(val_dataloader, model)
        print(val_loss)

    print("Finished Training...")
    test_loss = eval_fn(test_dataloader, model)
    print("Test Loss: ", test_loss)


if __name__ == "__main__":
    main()
