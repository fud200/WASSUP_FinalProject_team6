import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 하이퍼파라미터 설정
batch_size = 32

# 데이터 전처리 및 증강 설정
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 랜덤한 좌우 반전
    transforms.RandomRotation(10),  # 랜덤한 각도로 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터 로드
def load_data(train_data_dir, val_data_dir):
    train_dataset = datasets.ImageFolder(train_data_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_data_dir, transform=transform_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader
