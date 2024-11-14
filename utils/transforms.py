import torch
import torchvision.transforms.functional as F
import random

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']
        w, h = image.size
        image = F.resize(image, self.size)
        scale_w = self.size[0] / w
        scale_h = self.size[1] / h
        bbox = bbox * torch.tensor([scale_w, scale_h, scale_w, scale_h])
        return {'image': image, 'bbox': bbox}

class ToTensor(object):
    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']
        image = F.to_tensor(image)
        return {'image': image, 'bbox': bbox}
