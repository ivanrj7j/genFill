import torch
import os

def saveModel(model:torch.nn.Module, path:str, epoch:int|str):
    """
    Save the model to the given path.
    """

    torch.save(model.state_dict(), os.path.join(path, f"model_{epoch}.pth"))