import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cel = nn.CrossEntropyLoss()

    def forward(self, yhat, y):
        return self.cel(yhat, y)


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bcel = nn.BCEWithLogitsLoss()

    def forward(self, yhat, y):
        return self.bcel(yhat, y)
