import torch
import torch.nn as nn
from pathlib import Path
import torchvision.models as models

class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)
    

    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given

        '''
        model = models.resnet18(pretrained=True)

        torch.save(model.state_dict(), save_dir)


    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        model = torch.load(path)
        model.eval()
