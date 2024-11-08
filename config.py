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

training = False
# if the model is currently training or not 

holePredictorModeSave = "saves/cad8b0be-b647-4fa9-8a0a-f15bf5bbb69f/model_final.pth"

if not os.path.exists(savePath) and training:
    os.makedirs(savePath)