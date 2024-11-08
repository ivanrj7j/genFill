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
from tqdm import tqdm
from src.trainers.utils import saveModel, saveImage
# importing other stuff

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

def optimize(holedImage:torch.Tensor, holePredictions:torch.Tensor, targetImage:torch.Tensor) -> tuple[float, float]:
    pass

def train():
    if not config.training:
        print("Change config.training to True to start training...")
        return
    
    for epoch in range(1, config.epochs+1):
        filler.train()
        discriminator.train()
        # setting model to training mode 

        fillerLossTotal, discriminatorLossTotal = 0, 0
        batch = tqdm(fillerDataloaderTrain, f"EPOCH {epoch}", len(fillerDataloaderTrain))
        for i, ((holedImage, holePredictions), targetImage) in enumerate(batch):
            holedImage, holePredictions, targetImage = holedImage.to(config.device), holePredictions.to(config.device), targetImage.to(config.device)
            # moving the data to device 

            fillerLoss, discriminatorLoss = optimize(holedImage, holePredictions, targetImage)
            fillerLossTotal += fillerLoss / config.batchSize
            discriminatorLossTotal += discriminatorLoss / config.batchSize
            # calculating loss 
            
            batch.set_postfix({"FILLER LOSS":fillerLossTotal, "DISCRIMINATOR LOSS":discriminatorLossTotal,  "BATCH":i+1})

        if epoch % config.saveEvery == 0:
            saveModel(filler, config.savePath, f"filler_{epoch}")
            saveModel(discriminator, config.savePath, f"filler_{epoch}")

        for (x, holePredictions), y in fillerDataloaderTest:
            with torch.no_grad():
                x, holePredictions = x.to(config.device), holePredictions.to(config.device)
                # moving the data to device
                filler.eval()
                discriminator.eval()
                # setting models to evaluation mode 

                prediction = filler.forward(x, holePredictions)
                saveImage(prediction, config.ouputPreviewPath, epoch)
            break
        
