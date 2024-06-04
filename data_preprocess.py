import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def data_preprocess(train_data_dir, val_data_dir, batch_size):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_data_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_data_dir, transform=transform_val)
    
    return train_dataset, val_dataset
