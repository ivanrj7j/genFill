from uuid import uuid4
import os

epochs = 4
batchSize = 32
lr = 1e-4
# general training hyperparameter

resolution = (128, 128)
downScaleFactor = 4
trainPath = "data/train"
testPath = "data/test"
numWorkers = 1
# hole prediction dataloader specific 

resNets = 2
device = "cuda"
# model specific parameters

savePath = f"saves/{uuid4()}"
saveEvery = 11
# saving settings 

training = True
# if the model is currently training or not 

if not os.path.exists(savePath) and training:
    os.makedirs(savePath)