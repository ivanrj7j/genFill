import torch.nn as nn
import torch

class ResBlock(nn.Module):
    """
    # ResBlock

    Modified implementation of ResBlock with PReLU
    """
    def __init__(self, channels:int, norm:bool=True, *args, **kwargs) -> None:
        """
        Initializes the ResBlock

        Parameters
        channels (int): Number of input and output channels
        norm (bool): Whether to include batch normalization layers
        """
        super().__init__(*args, **kwargs)
        self.norm = norm
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU(1)
        self.norm1 = nn.BatchNorm2d(channels) if norm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU(1)
        self.norm2 = nn.BatchNorm2d(channels) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the DownScaleBlock.

        Parameters:
        x (torch.Tensor): Input tensor with shape (batch_size, inChannels, height, width).        
        """
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.norm1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.norm2(out)
        return out + x