import pathlib

import numpy as np
import torch
from accelerate import Accelerator
from torch.optim.rmsprop import RMSprop
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config
from lstm_dataset import LSTMDataset
from model import LSTMModel

# create global accelerator
accelerator = Accelerator()


def train_fn(model, optimizer, dataloader, loss_fn):
    model.train()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    num_iter = len(dataloader)
    running_loss = 0.0

    for batch in tqdm(dataloader, total=num_iter):

        optimizer.zero_grad()
        inputs, targets = batch
        output = model(inputs)

        loss = loss_fn(output, targets)

        running_loss += loss.item()
        accelerator.backward(loss)

    return running_loss / num_iter


def eval_fn(model, dataloader, loss_fn):
    model.eval()
    model, dataloader = accelerator.prepare(model, dataloader)
    num_iter = len(dataloader)
    running_loss = 0.0

    for batch in tqdm(dataloader, total=num_iter):
        inputs, targets = batch
        output = model(inputs)
        loss = loss_fn(output, targets)
        running_loss += loss.item()

    return model, running_loss / num_iter


def main():
    # load data
    train_data = LSTMDataset(config.TRAIN_PATH)
    test_data = LSTMDataset(config.TEST_PATH)
    dev_data = LSTMDataset(config.DEV_PATH)

    # convert to dataloader
    train_dataloader = DataLoader(
        train_data, batch_size=config.BS, shuffle=True, num_workers=4, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data, batch_size=config.BS, shuffle=False, num_workers=4, pin_memory=True
    )
    dev_dataloader = DataLoader(
        dev_data, batch_size=config.BS, shuffle=False, num_workers=4, pin_memory=True
    )

    # load embedding matrix
    embedding_mat = np.load(config.EMBEDDING_PATH)
    vocab_size, embedding_dim = embedding_mat.shape

    # model params
    model_params = {
        "gru_layers": 1,
        "vocab_size": vocab_size,
        "seq_len": config.MAX_LEN,
        "embedding_dim": embedding_dim,
        "hidden_fc1": 256,
        "hidden_fc2": 64,
        "num_classes": config.NUM_CLASSES,
        "embedding_mat": embedding_mat,
        "dropout_factor": config.DROPOUT,
        "train_embeddings": config.IS_TRAIN_EMBEDDING,
    }

    # create Model
    model = LSTMModel(**model_params)

    # create loss_fn , optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = RMSprop(model.parameters(), lr=config.LR)

    for i in range(config.NUM_EPOCHS):
        model, train_loss = train_fn(model, optimizer, train_dataloader, loss_fn)
        print(f"Epoch: {i+1} Loss: {train_loss}")

        eval_loss = eval_fn(model, dev_dataloader, loss_fn)
        print(f"Validation Loss: {eval_loss}")

    print("Finished training...")

    test_loss = eval_fn(model, test_dataloader, loss_fn)
    print(f"Test Loss: {test_loss}")


if __name__ == "__main__":
    main()
