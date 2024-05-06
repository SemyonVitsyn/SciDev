from torchvision import transforms
from PIL import Image
import cv2 as cv
from torch.utils.data import Dataset


class SemanticSegmentationDataset(Dataset):
    def __init__(self, paths, resolution=(512,512)):
        self.paths = paths
        self.resolution = resolution

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        src = cv.cvtColor(cv.imread(self.paths[idx][0]), cv.COLOR_BGR2RGB)
        src = transforms.ToTensor()(src)
        src = transforms.Resize(self.resolution)(src)
        
        trg = cv.cvtColor(cv.imread(self.paths[idx][1]), cv.COLOR_BGR2RGB)
        trg = transforms.ToTensor()(trg)
        trg = transforms.Resize(self.resolution, interpolation=Image.NEAREST)(trg)
                
        return src, trg