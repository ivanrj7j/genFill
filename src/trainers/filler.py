import config
# importing config files 

from src.model.holePredictor import HolePredictorModel
from src.model.filler import FillerModel
from src.model.discriminator import Discriminator
# importing models 

from src.loader.filler import FillerDataLoader, FillerDataset
# importing data loader and dataset 

import src.losses as losses
# importing custom loss function 

import torch
# other stuff

holePredictor = HolePredictorModel(4, 2).to(config.device)
filler = FillerModel(4, 8).to(config.device)
discriminator = Discriminator(4, 8).to(config.device)
# initializing models 

holePredictorWeights = torch.load(config.holePredictorModeSave, weights_only=True)
holePredictor.load_state_dict(holePredictorWeights)
# loading in weights for hole predictor 

def optimize():
    pass

def train():
    pass