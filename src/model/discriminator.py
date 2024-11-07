from torch import Tensor
from src.model.holePredictor import HolePredictorModel
import torch

class Discriminator(HolePredictorModel):
    """
    # Discriminator

    This is the discriminator model that checks if the generated image is real or fake, slightly modified holePredictor
    """
    def __init__(self, downScaleFactor: int, resNets: int, *args, **kwargs) -> None:
        super().__init__(downScaleFactor, resNets, *args, **kwargs)
        downscalers = list(self.downscalers)
        outputSize = downscalers[0].conv1.out_channels
        kernelSize = downscalers[0].conv1.kernel_size
        stride = downscalers[0].conv1.stride
        padding = downscalers[0].conv1.padding
        bias = downscalers[0].conv1.bias
        
        downscalers[0].conv1 = torch.nn.Conv2d(6, outputSize, kernelSize, stride, padding, bias=bias)

    def forward(self, x: Tensor, y: Tensor):
        inp = torch.cat((x, y), 1)
        return super().forward(inp)