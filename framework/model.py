import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=3)
        self.act_1 = nn.Sigmoid()
        self.layer_2 = nn.Linear(in_features=3, out_features=2)
        self.act_2 = nn.Sigmoid()
        self.layer_3 = nn.Linear(in_features=2, out_features=1)
        self.act_3 = nn.Sigmoid()

    def forward(self, x):
        l1 = self.layer_1(x)
        a1 = self.act_1(l1)
        l2 = self.layer_2(a1)
        a2 = self.act_2(l2)
        l3 = self.layer_2(a2)
        a3 = self.act_2(l3)
        return a3.view(-1)