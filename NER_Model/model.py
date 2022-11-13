import torch
import torch.nn as nn
from transformers import BertModel

import ner_config


class BertNer(nn.Module):
    def __init__(self, num_tags, dropout):
        super(BertNer, self).__init__()
        self.num_tags = num_tags
        self.num_hidden = 768
        self.dropout = dropout
        self.criterion = nn.CrossEntropyLoss()

        self.bert_layer = BertModel.from_pretrained(ner_config.BASE_MODEL).to(ner_config.DEVICE)
        self.fc1 = nn.Linear(self.num_hidden, self.num_tags)
        self.bert_drop = nn.Dropout(self.dropout)

    def forward(self, ids, mask, token_type_ids, ner_tags):
        out = self.bert_layer(
            ids, attention_mask=mask, token_type_ids=token_type_ids
        )
        out = self.bert_drop(out[0])
        out = self.fc1(out)

        loss_tags = self.loss_fn(out, ner_tags, mask)
        return loss_tags

    def loss_fn(self, out, target, mask):
        active_idx = mask.view(-1) == 1
        active_logits = out.view(-1, self.num_tags)
        active_labels = torch.where(
            active_idx,
            target.view(-1),
            torch.tensor(self.criterion.ignore_index).type_as(target),
        )
        loss_val = self.criterion(active_logits, active_labels)
        return loss_val
