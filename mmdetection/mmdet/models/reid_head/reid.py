import torch
import torch.nn.functional as F
from torch import nn
from .loss import make_reid_loss_evaluator

class REIDModule(torch.nn.Module):
    def __init__(self, cfg):
        super(REIDModule, self).__init__()
        self.cfg = cfg
        self.loss_evaluator = make_reid_loss_evaluator(cfg)
        self.fc = nn.Linear(256 * 7 * 7, 2048)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        feats = F.normalize(x, dim=-1)
        return feats


def build_reid(cfg):
    return REIDModule(cfg)