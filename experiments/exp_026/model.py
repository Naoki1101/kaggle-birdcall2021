from typing import Dict, Optional

import timm
import torch.nn as nn

from src import layer


class CustomModel(nn.Module):
    def __init__(
        self,
        n_classes: int,
        model_name: str = "resnet50",
        pooling_name: str = "GeM",
        args_pooling: Optional[Dict] = None,
    ):
        super(CustomModel, self).__init__()

        self.backbone = timm.create_model(model_name, pretrained=True)

        final_in_features = list(self.backbone.children())[-1].in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.pooling = getattr(layer, pooling_name)(**args_pooling)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(final_in_features, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pooling(x)
        x = x.view(len(x), -1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc(x)
        return x
