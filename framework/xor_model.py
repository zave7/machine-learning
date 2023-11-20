import torch.nn as nn
from functools import reduce

class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        layer_list  = []
        for in_feat, out_feat in layers:
            layer_list.append(nn.Linear(in_features=in_feat, out_features=out_feat))
            layer_list.append(nn.Sigmoid())
        self.layers = nn.ModuleList(layer_list)
        
    def forward(self, x):
        y = reduce(lambda acc, layer: layer.forward(acc),
                self.layers,
                x)
        return y.view(-1)