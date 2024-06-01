import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import initialize_model

# 하이퍼파라미터 설정
batch_size = 32
num_classes = 5  # 감정 클래스 수

# 데이터 디렉토리 설정
test_data_dir = "./data/test/"

# 데이터 전처리 설정
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터 로드
test_dataset = datasets.ImageFolder(test_data_dir, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 초기화 및 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = initialize_model(num_classes, device)

# 학습된 모델 가중치 로드
model.load_state_dict(torch.load('best_hybrid_model_densenet_neutral.pth'))

# 모델 평가 모드로 전환
model.eval()

# 성능 측정
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
