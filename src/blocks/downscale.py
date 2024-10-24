import torch.nn as nn
import torch

class DownScaleBlock(nn.Module):
    """
    # DownScaleBlock

    DownScaleBlock is used to scale down an image to half its original size.
    """
    def __init__(self, inChannels:int, outChannels:int, norm:bool=True, *args, **kwargs) -> None:
        """
        Initializes the DownScaleBlock with given input and output channels.

        Parameters:
        inChannels (int): Number of input channels.
        outChannels (int): Number of output channels.
        norm (bool, optional): Whether to include batch normalization. Defaults to True.
        """
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(inChannels, outChannels, 2, 2, bias=False)
        self.relu = nn.PReLU(1)
        self.normBlock = nn.BatchNorm2d(outChannels) if norm else nn.Identity()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the DownScaleBlock.

        Parameters:
        x (torch.Tensor): Input tensor with shape (batch_size, inChannels, height, width).
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.normBlock(x)

        return x