import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class MissileDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.annotations = []
        for file in os.listdir(root_dir):
            if file.endswith('.jpg') or file.endswith('.png'):
                self.image_files.append(os.path.join(root_dir, file))
                annotation_file = file.rsplit('.', 1)[0] + '.txt'
                self.annotations.append(os.path.join(root_dir, annotation_file))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        with open(self.annotations[idx], 'r') as f:
            annotation = f.readline().strip().split()
            bbox = list(map(float, annotation))
        sample = {'image': img, 'bbox': torch.tensor(bbox)}
        if self.transform:
            sample = self.transform(sample)
        return sample
