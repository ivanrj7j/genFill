import torch.nn as nn
import torch
# torch related stuff 

from src.blocks.res import ResBlock
from src.blocks.downscale import DownScaleBlock

class HolePredictorModel(nn.Module):
    def __init__(self, downScaleFactor:int, resNets:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        downBlocks = []
        for i in range(downScaleFactor):
            inChannels = 3 if i == 0 else 32 * (2**i)
            outChannels = 32 * (2**(i+1))
            norm = i % 2 == 0
            downBlocks.append(DownScaleBlock(inChannels, outChannels, norm))

        self.downscalers = nn.Sequential(*downBlocks)

        resBlocks = []
        for i in range(resNets):
            resBlocks.append(ResBlock(outChannels, i%2 == 0))

        self.resBlocks = nn.Sequential(*resBlocks)

        self.finalConv = nn.Conv2d(outChannels, 3, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x:torch.Tensor):
        x = self.downscalers(x)
        x = self.resBlocks(x)
        x = self.finalConv(x)
        x = self.sigmoid(x)
        return x