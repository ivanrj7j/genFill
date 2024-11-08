import torch
import os
from torchvision.utils import save_image

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
    imagePath = os.path.join(path, f"preview_{epoch}.")

    save_image(image, imagePath)