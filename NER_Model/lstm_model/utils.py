import json
import pathlib

import numpy as np
from tqdm.auto import tqdm

GLOVE_EMBD_PATH = pathlib.Path.home() / "Data/glove_6B/glove.6B.50d.txt"
EMBD_OUTPUT_PATH = pathlib.Path.home() / "Data/conll2003/lstm_data/embedding.npy"
DICTIONARY_OUTPUT_PATH = (
    pathlib.Path.home() / "Data/conll2003/lstm_data/word2index.json"
)
TRAIN_DATA_PATH = pathlib.Path.home() / "Data/conll2003/train.json"


def get_glove_embedding():
    """
    This function loads glvoe wmbeddings into a dictionary
    """
    embedding = {}
    N = 400_000
    print("Reading glove embedding...")
    with open(GLOVE_EMBD_PATH, "rb") as f:
        for line in tqdm(f, total=N):
            line = line.decode().split()
            word = line[0].lower()
            vector = np.array(line[1:]).astype(np.float32)
            embedding[word] = vector

    return embedding


def init_embedding(size=50):
    """
    Random initialization for embedding
    """
    vector = np.random.normal(0.0, 0.01, size)
    return vector


def generate_conll2003_embeddings():
    """
    Generate glove embeddings for conll2003 dataset
    """
    glove_embedding = get_glove_embedding()

    word2index = {}
    idx2word = {}
    embed_array = []

    word2index["<pad>"] = 1
    embed_array.append(init_embedding())

    word2index["<unk>"] = 0 
    embed_array.append(init_embedding())

    data = []
    with open(TRAIN_DATA_PATH, "r") as f:
        for line in f:
            data.append(json.loads(line))

    idx = 2

    for sample in tqdm(data, total=len(data)):
        words = sample["tokens"]

        for w in words:
            w = w.lower()

            # if word is not present in dictionary, add to dictionary and append embedding vector
            if w not in word2index.keys():
                word2index[w] = idx
                idx += 1
                if w not in glove_embedding.keys():
                    ev = init_embedding()
                else:
                    ev = glove_embedding[w]

                embed_array.append(ev)

            else:
                continue

    # save embeddings
    embed_array = np.vstack(embed_array)
    np.save(EMBD_OUTPUT_PATH, embed_array)

    # save dictionary
    print("Dicitionary Size: ", len(word2index))
    with open(DICTIONARY_OUTPUT_PATH, "w") as f:
        json.dump(word2index, f)


if __name__ == "__main__":
    generate_conll2003_embeddings()
