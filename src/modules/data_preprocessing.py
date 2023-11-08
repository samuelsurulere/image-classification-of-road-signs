import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
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

batch_size = 7
image_size = 160

def get_data():
    data_dir_train = '/home/sammie/Capstone Project/image-classification-of-road-signs/data/train'
    data_dir_test = '/home/sammie/Capstone Project/image-classification-of-road-signs/data/test'
    
    transform_train = transforms.Compose([
        transforms.CenterCrop(100),
        transforms.RandomInvert(0.2),
        transforms.RandomPerspective(),
        transforms.Resize((image_size,image_size)),
        transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
        transforms.ElasticTransform(alpha=20.0,sigma=10.0),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ])
    
    transform_test = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            ),
        ])
    
    data_set_train_raw = datasets.ImageFolder(data_dir_train)
    data_set_test_raw = datasets.ImageFolder(data_dir_test)
    
    N_data_train = len(data_set_train_raw)
    N_data_test = len(data_set_test_raw)
    
    data_set_train = Dataset(data_set_train_raw,transform_train)
    data_set_test = Dataset(data_set_test_raw,transform_test)
    
    train_idx = list(range(N_data_train))
    np.random.shuffle(train_idx)
    
    test_idx = list(range(N_data_test))
    np.random.shuffle(test_idx)
    
    data_set_train_sub = torch.utils.data.Subset(data_set_train, indices=train_idx)
    data_set_test_sub = torch.utils.data.Subset(data_set_test, indices=test_idx)
    
    train = DataLoader(data_set_train_sub, batch_size=batch_size, shuffle=True, num_workers=8)
    test = DataLoader(data_set_test_sub, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return train, test