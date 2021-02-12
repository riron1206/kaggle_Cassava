# https://pypi.org/project/microsoftvision/
import os
import sys
import torch.nn as nn

if os.getcwd() in ["/kaggle/working", "/content"]:
    sys.path.append("../input/microsoftvision")
else:
    sys.path.append(r"C:\Users\81908\Git\microsoftvision")
import microsoftvision


class MicrosoftVisionResnet50(nn.Module):
    def __init__(self, pretrained=False, n_classes=5):
        super().__init__()
        self.model = microsoftvision.models.resnet50(pretrained=pretrained)
        self.fc = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
