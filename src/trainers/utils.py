import torch
import os
from torchvision.utils import save_image
from src.model.filler import FillerModel
from src.model.discriminator import Discriminator

def saveModel(model:torch.nn.Module, path:str, epoch:int|str):
    """
    Save the model to the given path.
    """

    torch.save(model.state_dict(), os.path.join(path, f"model_{epoch}.pth"))

def saveImage(image:torch.Tensor, path:str, epoch:int|str):
    """
    Save the image to the given path.
    """
    image = (image + 1) / 2
    imagePath = os.path.join(path, f"preview_{epoch}.png")

    save_image(image, imagePath)

def loadModels(fillerPath:str, discriminatorPath:str, downscaleFactor:int, resNets:int, device:str):
    filler = FillerModel(downscaleFactor, resNets)
    discriminator = Discriminator(downscaleFactor, resNets)

    if fillerPath != "":
        print(f"Loading {fillerPath}")
        filler.load_state_dict(torch.load(fillerPath, weights_only=True))
    if discriminatorPath != "":
        print(f"Loading {discriminatorPath}")
        discriminator.load_state_dict(torch.load(discriminatorPath, weights_only=True))

    return filler.to(device), discriminator.to(device)