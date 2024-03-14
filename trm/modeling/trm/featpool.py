import torch
from torch import nn


class FeatAvgPool(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride):
        super(FeatAvgPool, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.pool = nn.AvgPool1d(kernel_size, stride)

    def forward(self, x):
        x = x.transpose(1, 2)  # B, C, T
        return self.pool(self.conv(x).relu())

def build_featpool(cfg):
    input_size = cfg.MODEL.TRM.FEATPOOL.INPUT_SIZE
    hidden_size = cfg.MODEL.TRM.FEATPOOL.HIDDEN_SIZE
    kernel_size = cfg.MODEL.TRM.FEATPOOL.KERNEL_SIZE  # 4 for anet, 2 for tacos, 16 for charades
    stride = cfg.INPUT.NUM_PRE_CLIPS // cfg.MODEL.TRM.NUM_CLIPS
    return FeatAvgPool(input_size, hidden_size, kernel_size, stride)

if __name__ == '__main__':
    featpool = FeatAvgPool(4096, 512, 2, 2)
    x = torch.rand(1,512, 4096)
    print(featpool(x).shape)  # torch.Size([2, 512, 32])