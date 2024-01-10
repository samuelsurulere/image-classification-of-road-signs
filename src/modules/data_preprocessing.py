import torch
import os
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Subset


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)

batch_size = 5
image_size = 224
num_workers = os.cpu_count()

def get_data():
    data_dir_train = '../data/train'
    data_dir_test = '../data/test'
    
    transform_train = transforms.Compose([
        transforms.CenterCrop(100),
        transforms.Resize((image_size,image_size), antialias=True),
        transforms.RandomPerspective(p=0.2),
        transforms.RandomInvert(p=0.2),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ElasticTransform(alpha=20.0,sigma=10.0),
        transforms.RandomRotation(45),
        transforms.Pad(padding=1),
        transforms.ToTensor(),
        ])
    
    transform_test = transforms.Compose([
        transforms.Resize((image_size,image_size), antialias=True),
        transforms.ToTensor(),
        ])
    
    data_set_train_raw = ImageFolder(data_dir_train)
    data_set_test_raw = ImageFolder(data_dir_test)
    
    N_data_train = len(data_set_train_raw)
    N_data_test = len(data_set_test_raw)
    
    data_set_train = CustomDataset(data_set_train_raw, transform_train)
    data_set_test = CustomDataset(data_set_test_raw, transform_test)
    
    train_idx = list(range(N_data_train))
    np.random.shuffle(train_idx)
    
    test_idx = list(range(N_data_test))
    np.random.shuffle(test_idx)
    
    data_set_train_sub = Subset(data_set_train, indices=train_idx)
    data_set_test_sub = Subset(data_set_test, indices=test_idx)
    
    train = DataLoader(data_set_train_sub, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    num_workers=num_workers, 
                    pin_memory=True)
    test = DataLoader(data_set_test_sub, 
                    batch_size=batch_size, 
                    shuffle=False, 
                    num_workers=num_workers, 
                    pin_memory=True)
    
    return train, test