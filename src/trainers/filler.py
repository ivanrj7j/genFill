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
from torch.optim.adam import Adam
# other stuff

holePredictor = HolePredictorModel(4, 2).to(config.device)
filler = FillerModel(4, 8).to(config.device)
discriminator = Discriminator(4, 8).to(config.device)
# initializing models 

holePredictorWeights = torch.load(config.holePredictorModeSave, weights_only=True)
holePredictor.load_state_dict(holePredictorWeights)
# loading in weights for hole predictor 

fillerDatasetTrain = FillerDataset(config.trainPath, config.downScaleFactor, config.resolution)
fillerDatasetTest = FillerDataset(config.testPath, config.downScaleFactor, config.resolution)
# initializing datsets 

fillerDataloaderTrain = FillerDataLoader(fillerDatasetTrain, holePredictor, config.device, config.batchSize)
fillerDataloaderTest = FillerDataLoader(fillerDatasetTest, holePredictor, config.device, config.batchSize)
# initializing dataloaders 

fillerOptimizer = Adam(filler.parameters(), config.lr, config.betas)
discriminatorOptimizer = Adam(discriminator.parameters(), config.lr, config.betas)
# initializing optimizers 

def optimize():
    pass

def train():
    pass