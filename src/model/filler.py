import torch.nn as nn
import torch
# torch related stuff 

from src.blocks.res import ResBlock
from src.blocks.downscale import DownScaleBlock
from src.blocks.upscale import Upscale2xBlock


class FillerModel(nn.Module):
    def __init__(self, sampleBlocks:int, resNets:int, *args, **kwargs) -> None:
        """
        Initialize a FillerModel

        Parameters:
        sampleBlocks (int): Number of SampleBlock layers to be used in the model
        resNets (int): Number of ResBlock layers to be used in the model
        """
        super().__init__(*args, **kwargs)

        downBlocks = []
        for i in range(sampleBlocks):
            inChannels = 3 if i == 0 else 32 * (2**i)
            outChannels = 32 * (2**(i+1))
            norm = i % 2 == 0
            downBlocks.append(DownScaleBlock(inChannels, outChannels, norm))

        self.downscalers = nn.ModuleList(downBlocks)
        # initializing downscalers 

        resBlocks = []
        for i in range(resNets):
            resBlocks.append(ResBlock(outChannels+1, i%2 == 0))

        self.resBlocks = nn.Sequential(*resBlocks)
        # initializing residual blocks 

        upBlocks = []
        for i in range(sampleBlocks):
            inChannels = 32 * (2**(sampleBlocks - i))
            outChannels = 32 * (2**(sampleBlocks - 1 - i))
            
            if i == 0:
                inChannels += 1

            norm = i % 2 == 0
            upBlocks.append(Upscale2xBlock(inChannels, outChannels, norm))

        self.upscalers = nn.ModuleList(upBlocks)
        # initializing upscalers 

        self.finalConv = nn.Conv2d(outChannels, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, holePredictions:torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the FillerModel

        Parameters:
        x (torch.Tensor): Input image tensor with shape (batch_size, channels, height, width)
        holePredictions (torch.Tensor): Hole predictions by `HolePredictorModel` tensor with shape (batch_size, 1, height, width)
        """
        downScaleOutputs = []
        y = x
        for downscaler in self.downscalers:
            y = downscaler(y)
            downScaleOutputs.append(y)
        # downscaling the image 

        resOutput = self.resBlocks(torch.cat((y, holePredictions), 1))
        # passing through residual blocks

        
        y = resOutput
        for i, upscaler in enumerate(self.upscalers):
            downIndex = len(downScaleOutputs) - i - 2
            y = upscaler(y)
            if i < len(self.upscalers) - 1:
                y += downScaleOutputs[downIndex]
        # upscaling image to og resolution 

        return self.tanh(self.finalConv(y))