from typing import List
from torch._tensor import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import _BaseDataLoaderIter
from src.loader.holePredictor import HolePredictorDataset
from src.model.holePredictor import HolePredictorModel
import os
from PIL import Image
from typing import Callable, Any

class  FillerDataset(HolePredictorDataset):
    """
    # FillerDataset

    This loads the dataset for filling the holes in image. This makes holes in the image feeds the data to filler model. The dataset provides, original image (in the desired resolution) and image with the holes.
    """
    def __init__(self, path: str, downScaleFactor: int, resolution: tuple[int, int], noiseProbablity: float = 0.2, additionalNoiseIntensity: float = 0.2) -> None:
        """
        Initializes the FillerDataset

        Parameters:
        path (str): Path to the dataset folder
        downScaleFactor (int): Downscale factor for the images
        resolution (tuple[int, int]): Desired resolution for the images
        noiseProbablity (float): Probability of adding noise to the a certain block in the image, a hole of noise is created. Defaults to 0.2
        additionalNoiseIntensity (float): Additional intensity of noise to be added, in the non-hole parts of the images. Defaults to 5e-3
        """

        super().__init__(path, downScaleFactor, resolution, noiseProbablity, additionalNoiseIntensity)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        imagePath = os.path.join(self.path, self.images[idx])

        image = Image.open(imagePath)
        image = self.transform(image) 

        holedImage, y = self.getHoles(image)

        return holedImage, image  
    

class FillerDataLoader(DataLoader):
    def __init__(self, dataset: FillerDataset, holePredictorModel:HolePredictorModel, device:str, batch_size: int | None = 1, shuffle: bool | None = None, num_workers: int = 0, **kwargs):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)
        
        self.holePredictorModel = holePredictorModel
        self.device = device

    def __iter__(self):
        modelOut = super().__iter__()

        for x, y in modelOut:
            x, y = x.to(self.device), y.to(self.device)
            modelPredictions = self.holePredictorModel.forward(x)
            yield (x, modelPredictions), y