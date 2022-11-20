import torch 
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, **params):
        super(LSTMModel, self).__init__()

        self.num_gru_layers = params["gru_layers"]
        self.vocab_size = params["vocab_size"]
        self.seq_len = params["seq_len"]
        self.embedding_dim = params["embedding_dim"]

        self.hidden_fc1 = params["hidden_fc1"]
        self.hidden_fc2 = params["hidden_fc2"]
        self.num_classes = params["num_classes"]

        self.embedding_weights = torch.tensor(params["embedding_mat"], dtype=torch.long)
        self.train_embeddings = params["train_embeddings"]

        self.dropout_factor = params["dropout_factor"]

        # looad embeddings
        self.embedding_layer = nn.Embedding(
            self.vocab_size, self.embedding_dim, padding_idx=1
        )
        self.embedding_layer.load_state_dict({"weight": self.embedding_weights})
        if not self.train_embeddings:
            self.embedding_layer.weight.requires_grad = False

        # build gru layer
        self.gru_layer = nn.GRU(
            self.embedding_dim,
            self.hidden_fc1,
            self.num_gru_layers,
            batch_first=True,
        )

        # classifier head
        self.do1 = nn.Dropout(self.dropout_factor)
        self.fc1 = nn.Linear(self.hidden_fc1, self.hidden_fc2)
        self.fc2 = nn.Linear(self.hidden_fc2, self.num_classes)

    def forward(self, x):
        out, _ = self.gru_layer(self.embedding_layer(x))
        out = self.do1(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
