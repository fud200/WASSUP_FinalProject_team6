import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from vit_pytorch import ViT
from torchvision.models import efficientnet_b1, efficientnet_b2
from torch.optim.lr_scheduler import StepLR
from data_preprocess import data_preprocess

def train(batch_size, learning_rate, num_epochs, patience):
    # 데이터 전처리 및 증강 설정
    train_data_dir = "./data/train/"
    val_data_dir = "./data/val/"
    train_dataset, val_dataset = data_preprocess(train_data_dir, val_data_dir, batch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Hybrid Model 정의: EfficientNet + ViT
    class HybridModel(nn.Module):
        def __init__(self, num_classes, model_version='b1'):
            super(HybridModel, self).__init__()
            if model_version == 'b1':
                self.efficientnet = efficientnet_b1(pretrained=True)
            elif model_version == 'b2':
                self.efficientnet = efficientnet_b2(pretrained=True)
            else:
                raise ValueError("model_version should be 'b1' or 'b2'")
            
            num_ftrs = self.efficientnet.classifier[1].in_features
            self.efficientnet.classifier = nn.Identity()  # Remove the fully connected layer
            
            self.vit = ViT(
                image_size=224,
                patch_size=32,
                num_classes=1024,  # Use 1024 as an intermediate dimension
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1
            )
            
            self.fc = nn.Linear(num_ftrs + 1024, num_classes)  # Adjusted for combined features

        def forward(self, x):
            efficientnet_features = self.efficientnet(x)  # Shape: (batch_size, num_ftrs)
            vit_features = self.vit(x)  # Shape: (batch_size, 1024)
            combined_features = torch.cat((efficientnet_features, vit_features), dim=1)  # Shape: (batch_size, num_ftrs + 1024)
            out = self.fc(combined_features)
            return out

    # 모델 초기화 및 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_version = 'b1'  # 'b1' 또는 'b2'로 변경 가능
    model = HybridModel(num_classes=4, model_version=model_version).to(device)

    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # 30 epoch마다 학습률을 0.1배씩 감소시킴

    # 학습 루프
    best_val_accuracy = 0.0
    best_epoch = 0
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

        # 모델 저장 및 조기 종료 조건 확인
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), f'best_hybrid_model_efficientnet_{model_version}.pth')
            print('Model saved!')
        elif epoch - best_epoch >= patience:
            print("Early stopping!")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Model Training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    args = parser.parse_args()

    train(args.batch_size, args.learning_rate, args.num_epochs, args.patience)

