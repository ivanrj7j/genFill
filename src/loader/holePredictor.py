from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from PIL import Image
import torch
import os

class HolePredictorDataset(Dataset):
    def __init__(self, path:str, downScaleFactor:int, resolution:tuple[int, int], minHoleSize:float = 5e-3, maxHoleSize:float = 2.5e-2) -> None:
        super().__init__()

        self.path = path
        self.images = os.listdir(path)
        self.factor = 2 ** downScaleFactor
        self.downscleResolution = resolution[0] // self.factor, resolution[1] // self.factor

        self.transform = Compose([
            Resize(resolution),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.minHoleSize = minHoleSize
        self.maxHoleSize = maxHoleSize

    def getHoles(self, image:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.tensor([0, 1], dtype=torch.float32)
        probablities = torch.tensor([0.2, 0.8])
        indices = torch.multinomial(probablities, torch.prod(torch.tensor(self.downscleResolution)), replacement=True)
        y = values[indices].reshape(self.downscleResolution)

        multiplicator = y.repeat_interleave(self.factor, 1).repeat_interleave(self.factor, 0)
        
        return image * multiplicator, y.unsqueeze(0)

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        imagePath = os.path.join(self.path, self.images[idx])

        image = Image.open(imagePath)
        image = self.transform(image) 

        return self.getHoles(image) 