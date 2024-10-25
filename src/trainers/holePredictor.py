import config
from src.model.holePredictor import HolePredictorModel
from src.loader.holePredictor import getDataLoader
import torch
from torch.optim.adam import Adam
from torch.nn import BCELoss
from tqdm import tqdm
from src.trainers.utils import saveModel

trainLoader = getDataLoader(config.trainPath, config.downScaleFactor, config.resolution, batchSize=config.batchSize, numWorkers=config.numWorkers)
testLoader = getDataLoader(config.testPath, config.downScaleFactor, config.resolution, batchSize=config.batchSize, numWorkers=config.numWorkers)
model = HolePredictorModel(config.downScaleFactor, config.resNets).to(config.device)
optimizer = Adam(model.parameters(), config.lr)
lossFunc = BCELoss()

def optimize(x:torch.Tensor, y:torch.Tensor):
    predictions = model.forward(x)
    loss_value = lossFunc.forward(predictions, y)

    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()

    return loss_value.item()


def train():
        for epoch in range(1, config.epochs+1):
            loss = 0
            batch = tqdm(trainLoader, f"EPOCH {epoch}", len(trainLoader))
            for i, (x, y) in enumerate(batch):
                x, y = x.to(config.device), y.to(config.device)
                loss += optimize(x, y) / config.batchSize
                batch.set_postfix({"LOSS":loss, "BATCH":i})

            if epoch % config.saveEvery == 0:
                saveModel(model, config.savePath, epoch)

            for x, y in testLoader:
                with torch.no_grad():
                    x, y = x.to(config.device), y.to(config.device)
                    predictions = model.forward(x)
                    loss = lossFunc.forward(predictions, y)
                    print(f"Test Loss: {loss.item():.4f}\n")
                break

        saveModel(model, config.savePath, "final")           