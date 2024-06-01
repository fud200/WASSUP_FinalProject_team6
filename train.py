import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from data_preprocess import load_data
from model import initialize_model

# 하이퍼파라미터 설정
batch_size = 32
learning_rate = 0.001
num_epochs = 100
num_classes = 5  # 감정 클래스 수

# 데이터 디렉토리 설정
train_data_dir = "./data/train/"
val_data_dir = "./data/val/"

# 데이터 로드
train_loader, val_loader = load_data(train_data_dir, val_data_dir)

# 모델 초기화 및 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = initialize_model(num_classes, device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # 30 epoch마다 학습률을 0.1배씩 감소시킴

# 학습 루프
best_val_accuracy = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # 검증
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    val_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {running_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")

    scheduler.step()  # 학습률 감소 적용

    # 모델 저장
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_hybrid_model_densenet_neutral.pth')
        print('Model saved!')