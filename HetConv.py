import torch.nn as nn
import torch
class HetConv2d(nn.Module):
    def __init__(self, n_feats, p=2):
        super(HetConv2d, self).__init__()
        self.n_feats = n_feats
        self.blocks = nn.ModuleList()
        for i in range(n_feats):
            self.blocks.append(self.make_HetConv2d(i,p))

    def make_HetConv2d(self, n, p):
        layers = []
        for i in range(self.n_feats):
            if ((i-n) % (p)) == 0 :
                layers.append(nn.Conv2d(1,1,3,1,1))

            else:
                layers.append(nn.Conv2d(1,1,1,1,0))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = []
        for i in range(0, self.n_feats):
            out.append(self.blocks[i](x[:, i:i+1, :, :]))
        return torch.cat(out, 1)

