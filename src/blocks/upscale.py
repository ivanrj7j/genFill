import torch.nn as nn
import torch

class Upscale2xBlock(nn.Module):
    """
    # Upscale2xBlock

    This block is used to upscale the image size by 2 times
    """
    def __init__(self, inChannels:int, outChannels:int, norm:bool=True, *args, **kwargs) -> None:
        """
        Initializes the Upscale2XBlock with given input and output channels.

        Parameters:
        inChannels (int): Number of input channels.
        outChannels (int): Number of output channels.
        norm (bool, optional): Whether to include batch normalization. Defaults to True.
        """
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(inChannels, outChannels*(4*4), 2, 2)
        self.normBlock = nn.BatchNorm2d(outChannels*(4*4)) if norm else nn.Identity()
        self.upsample = nn.PixelShuffle(4)
        self.relu = nn.PReLU(1)


    def forward(self, x:torch.Tensor):
        """
        Performs a forward pass through the Upscale2XBlock.
        
        Parameters:
        x (torch.Tensor): Input tensor with shape (batch_size, inChannels, height, width).
        """

        x = self.conv1(x)
        x = self.normBlock(x)
        x = self.upsample(x)
        x = self.relu(x)

        return x