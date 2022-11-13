import torch

TAG_MAP = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}

TRAIN_BS = 8
VAL_BS = 8
MAX_LEN = 128
NUM_TAGS = len(TAG_MAP)

LR = 1e-3
DROPOUT = 0.2
NUM_EPOCHS = 5

BASE_MODEL = "bert-base-uncased"
MODEL_OPUTPUT_DIR = "/home/sachin/Data/conll2003/model/"
DATA_FOLDER = "/home/sachin/Data/conll2003/"

DEVICE = torch.device('cpu')
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
