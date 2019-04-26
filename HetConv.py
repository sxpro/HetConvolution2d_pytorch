import torch.nn as nn
import torch
class HetConv2d(nn.Module):
    def __init__(self, in_feats, out_feats, groups=1, p=2):
        super(HetConv2d_v2, self).__init__()
        if in_feats % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.groups = groups
        self.blocks = nn.ModuleList()
        for i in range(out_feats):
            self.blocks.append(self.make_HetConv2d(i, p))

    def make_HetConv2d(self, n, p):
        layers = nn.ModuleList()
        for i in range(self.in_feats):
            if ((i - n) % (p)) == 0:
                layers.append(nn.Conv2d(1, 1, 3, 1, 1))

            else:
                layers.append(nn.Conv2d(1, 1, 1, 1, 0))
        return layers

    def forward(self, x):
        out = []
        for i in range(0, self.out_feats):
            out_ = self.blocks[i][0](x[:, 0: 1, :, :])
            for j in range(1, self.in_feats):
               out_ += self.blocks[i][j](x[:, j:j + 1, :, :])
            out.append(out_)
        return torch.cat(out, 1)

class HetConv2d_v2(nn.Module):
    def __init__(self, in_feats, out_feats, p=4, groups=1):
        super(HetConv2d_v2, self).__init__()
        if in_feats % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        self.p = p
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.groups = groups
        self.blocks = nn.ModuleList()
        for i in range(out_feats):
            self.blocks.append(self.make_HetConv2d(p))

    def make_HetConv2d(self, p):
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(self.in_feats//p, 1, 3, 1, 1))
        layers.append(nn.Conv2d(self.in_feats-self.in_feats//p, 1, 1, 1, 0))
        return layers

    def forward(self, x):
        out = []
        for i in range(0, self.out_feats):
            out1x1 = torch.zeros([0])
            out3x3 = torch.zeros([0])
            for j in range(0, self.in_feats):
                if ((j - i) % (self.p)) == 0:
                    if out3x3.shape[0] == 0:
                        out3x3 = x[:, j:j + 1, :, :]
                    else:
                        out3x3 = torch.cat((out3x3, x[:, j:j + 1, :, :]), 1)
                else:
                    if out1x1.shape[0] == 0:
                        out1x1 = x[:, j:j + 1, :, :]
                    else:
                        out1x1 = torch.cat((out1x1, x[:, j:j + 1, :, :]), 1)
            out3x3 = self.blocks[i][0](out3x3)
            out1x1 = self.blocks[i][1](out1x1)
            out_ = out1x1 + out3x3
            out.append(out_)
        return torch.cat(out, 1)


