import torch.nn as nn
import torch
# torch related stuff 

from src.blocks.res import ResBlock
from src.blocks.downscale import DownScaleBlock

class HolePredictorModel(nn.Module):
    """
    # HolePredictorModel

    This model predicts divides the image into smaller chunks and checks if there is a hole in the image
    """
    def __init__(self, downScaleFactor:int, resNets:int, *args, **kwargs) -> None:
        """
        Initializes the HolePredictorModel with given downscale factor and number of resnets.

        Parameters:
        downScaleFactor (int): Downscale factor of the image
        resNets (int): Number of ResBlock layers to be used in the model
        """
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

        self.finalConv = nn.Conv2d(outChannels, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x:torch.Tensor):
        """
        Performs a forward pass through the HolePredictorModel.

        Parameters:
        x (torch.Tensor): Input tensor with shape (batch_size, inChannels, height, width).
        """
        x = self.downscalers(x)
        x = self.resBlocks(x)
        x = self.finalConv(x)
        x = self.sigmoid(x)
        return x