"""
This file contains some loss functions for models
"""

import torch

def discriminatorLoss(ouptputReal:torch.Tensor, ouptputFake:torch.Tensor, entropyNoise:float = 5e-2):
    """
    Loss function for the discriminator.
    
    Parameters:
    ouptputReal (torch.Tensor): Output tensor from the discriminator for real inputs with shape (batchSize, 1, m, n).
    ouptputFake (torch.Tensor): Output tensor from the discriminator for fake inputs with shape (batchSize, 1, m, n).
    entropyNoise (float): Entropy noise to add to the loss. Defaults to 5e-2.
    """

    noise = torch.rand_like(ouptputReal) * entropyNoise
    realLoss = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(ouptputReal, torch.ones_like(ouptputReal)-noise))
    fakeLoss = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(ouptputFake, torch.zeros_like(ouptputFake)+noise))

    return realLoss+fakeLoss

def generatorLoss(discriminatorOutput:torch.Tensor):
    """
    Loss function for the generator.

    Parameter:
    discriminatorOutput (torch.Tensor): Output tensor from the discriminator for the generated image with shape (batchSize, 1, m, n).
    """
    return torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(discriminatorOutput, torch.ones_like(discriminatorOutput)))