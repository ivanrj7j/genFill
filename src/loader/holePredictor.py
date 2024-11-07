from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from PIL import Image
import torch
import os

class HolePredictorDataset(Dataset):
    """
    # HolePredictorDataset

    This loads the dataset for predicting holes in image. This makes holes in the image and gives the labels for the regions in which the holes are present
    """
    def __init__(self, path:str, downScaleFactor:int, resolution:tuple[int, int], noiseProbablity:float=0.2, additionalNoiseIntensity:float=2e-1, noiseIntensity:float=1.0) -> None:
        """
        Initializes the HolePredictorDataset

        Parameters:
        path (str): Path to the dataset folder
        downScaleFactor (int): Downscale factor for the images
        resolution (tuple[int, int]): Desired resolution for the images
        noiseProbablity (float): Probability of adding noise to the a certain block in the image, a hole of noise is created. Defaults to 0.2
        additionalNoiseIntensity (float): Additional intensity of noise to be added, in the non-hole parts of the images. Defaults to 5e-3
        noiseIntensity (float): Intensity of noise to be added in the holes. Defaults to 1.0. Can be any real number 1 for simple averaging.
        """
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

        self.noiseProbablity = noiseProbablity
        self.additionalNoiseIntensity = additionalNoiseIntensity
        self.noiseIntensity = noiseIntensity

    def getHoles(self, image:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add random holes to the image.

        Parameters:
        image (torch.Tensor): Input image tensor
        """
        values = torch.tensor([0, self.noiseIntensity], dtype=torch.float32)
        probablities = torch.tensor([1-self.noiseProbablity, self.noiseProbablity])
        indices = torch.multinomial(probablities, torch.prod(torch.tensor(self.downscleResolution)), replacement=True)
        y = values[indices].reshape(self.downscleResolution)
        # deciding where to put the holes 

        multiplicator = y.repeat_interleave(self.factor, 1).repeat_interleave(self.factor, 0)
        multiplicator = multiplicator + (((torch.rand_like(multiplicator)*2) -1)*self.additionalNoiseIntensity)
        noise = (torch.rand_like(image) * 2) - 1
        noise *= multiplicator
        # adding random noise 
        
        return (image + noise)/2, y.unsqueeze(0)

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        imagePath = os.path.join(self.path, self.images[idx])

        image = Image.open(imagePath)
        image = self.transform(image) 

        return self.getHoles(image)


def getDataLoader(path:str, downScaleFactor:int, resolution:tuple[int, int], noiseProbablity:float=0.2, additionalNoiseIntensity:float=2e-1, batchSize:int=32, numWorkers:int=1):
    """
    Returns a DataLoader for the HolePredictorDataset
    """

    dataset = HolePredictorDataset(path, downScaleFactor, resolution, noiseProbablity, additionalNoiseIntensity)
    return DataLoader(dataset, batch_size=batchSize, shuffle=True)