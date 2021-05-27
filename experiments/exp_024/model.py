from typing import Dict, Optional

import timm
import torch
import torch.nn as nn

from src import layer


class CustomModel(nn.Module):
    def __init__(
        self,
        n_classes: int,
        model_name: str = "resnet50",
        pooling_name: str = "GeM",
        args_pooling: Optional[Dict] = None,
        middle_fc_features: int = 128,
    ):
        super(CustomModel, self).__init__()

        self.backbone = timm.create_model(model_name, pretrained=True)

        final_in_features = list(self.backbone.children())[-1].in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.pooling = getattr(layer, pooling_name)(**args_pooling)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(final_in_features, middle_fc_features)

        self.fc_pos1 = nn.Linear(397, 256)
        self.fc_pos2 = nn.Linear(256, middle_fc_features)

        self.final_fc = nn.Linear(middle_fc_features * 2, n_classes)

    def forward(self, feats):
        x = feats["image"]
        d = feats["distance"]

        x = self.backbone(x)
        x = self.pooling(x)
        x = x.view(len(x), -1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc(x)

        d = self.fc_pos1(d)
        d = self.act(d)
        d = self.drop(d)
        d = self.fc_pos2(d)

        x = torch.cat((x, d), dim=1)
        x = self.final_fc(x)

        return x
