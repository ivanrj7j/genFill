from uuid import uuid4
import os

trainingID = str(uuid4())

epochs = 4
batchSize = 32
lr = 1e-4
betas = (0.5, 0.999)
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

savePath = os.path.join("saves", trainingID)
saveEvery = 11
# saving settings 

training = False
# if the model is currently training or not 

holePredictorModeSave = "saves/cad8b0be-b647-4fa9-8a0a-f15bf5bbb69f/model_final.pth"
# specifying where the saved holePredictor is 

ouputPreviewPath = os.path.join("imagePreviews", trainingID)
# specifying where the output previews are saved

if not os.path.exists(savePath) and training:
    os.mkdir(savePath)

if not os.path.exists(ouputPreviewPath) and training:
    os.mkdir(ouputPreviewPath)

if training:
    print(f"Training ID: {trainingID}")