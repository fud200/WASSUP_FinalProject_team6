import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from model import HybridModel

def test(batch_size=32):
    # 하이퍼파라미터 설정
    num_classes = 4  # 감정 클래스 수

    # 데이터 전처리 설정
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 테스트 데이터셋 로드
    test_data_dir = "./data/test/"
    test_dataset = datasets.ImageFolder(test_data_dir, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 모델 초기화 및 가중치 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_version = 'b1'  # 'b1' 또는 'b2'로 변경 가능
    model = HybridModel(num_classes=num_classes, model_version=model_version).to(device)
    model.load_state_dict(torch.load(f'best_hybrid_model_efficientnet_{model_version}.pth'))

    # 손실 함수 설정
    criterion = nn.CrossEntropyLoss()

    # 테스트 데이터셋 성능 평가
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {running_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    test(batch_size=32)  # 여기에 원하는 배치 사이즈를 넣어주세요.
