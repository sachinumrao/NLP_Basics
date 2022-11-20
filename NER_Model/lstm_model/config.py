import pathlib

# data processing configs
MAX_LEN = 128

BS = 32
LR = 1e-3
NUM_CLASSES = 9
NUM_EPOCHS = 50
DROPOUT = 0.2
IS_TRAIN_EMBEDDING = False

EMBEDDING_PATH = pathlib.Path.home() / "Data/conll2003/lstm_data/embedding.npy"
TRAIN_PATH = pathlib.Path.home() / "Data/conll2003/lstm_data/train_processed.jsonl"
TEST_PATH = pathlib.Path.home() / "Data/conll2003/lstm_data/test_processed.jsonl"
DEV_PATH = pathlib.Path.home() / "Data/conll2003/lstm_data/dev_processed.jsonl"
