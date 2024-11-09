from uuid import uuid4
import os

trainingID = str(uuid4())

epochs = 150
batchSize = 16
lr = 1e-4
betas = (0.5, 0.999)
# general training hyperparameter

resolution = (128, 128)
downScaleFactor = 4
resNets = 12
trainPath = "data/train"
testPath = "data/test"
numWorkers = 1
# hole prediction dataloader specific 

device = "cuda"
# model specific parameters

savePath = os.path.join("saves", trainingID)
saveEvery = 5
# saving settings 

training = True
# if the model is currently training or not 

holePredictorModeSave = "saves/cad8b0be-b647-4fa9-8a0a-f15bf5bbb69f/model_final.pth"
# specifying where the saved holePredictor is 

ouputPreviewPath = os.path.join("imagePreviews", trainingID)
# specifying where the output previews are saved

fillerLoadPath = "saves/ff609a92-03e4-458e-85ce-374979d221d2/model_filler_25.pth"
discriminatorLoadPath = "saves/ff609a92-03e4-458e-85ce-374979d221d2/model_discriminator_25.pth"


decayEvery = 2

if not os.path.exists(savePath) and training:
    os.mkdir(savePath)

if not os.path.exists(ouputPreviewPath) and training:
    os.mkdir(ouputPreviewPath)

if training:
    print(f"Training ID: {trainingID}")