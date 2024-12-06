# GPU 성능 비교를 위한 코드
# RTX 3090      - Epoch 당 6.2~6.4초
# AGX Orin CPU  - Epoch 당 18.3초  

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# 하이퍼파라미터 설정: 학습 설정에 필요한 기본 값들 (배치 크기, 학습률, 에포크 수 등)
batch_size = 64
learning_rate = 0.01
epochs = 5

# MNIST 데이터셋 로드: 손글씨 숫자 이미지 데이터를 로드하고 전처리
transform = transforms.Compose([
    transforms.ToTensor(),  # 데이터를 텐서로 변환
    transforms.Normalize((0.5,), (0.5,))  # 데이터 정규화
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 학습 데이터 로더
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 테스트 데이터 로더

# 간단한 신경망 정의: 입력 이미지를 분류하기 위한 신경망 구조 정의
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()  # 2D 이미지를 1D 벡터로 변환
        self.fc1 = nn.Linear(28 * 28, 128)  # 첫 번째 완전연결 레이어
        self.relu = nn.ReLU()  # 활성화 함수
        self.fc2 = nn.Linear(128, 10)  # 출력 레이어 (10개의 클래스)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 모델 초기화: 모델, 손실 함수, 최적화 알고리즘 설정
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류를 위한 손실 함수
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 확률적 경사 하강법 사용

# 학습 함수: 모델을 학습 데이터로 훈련
# 각 배치에 대해 손실을 계산하고 모델의 가중치를 업데이트
# 훈련 손실, 정확도 및 학습 시간을 출력
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    start_time = time.time()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(1) == target).sum().item()
    end_time = time.time()
    accuracy = correct / len(loader.dataset)
    elapsed_time = end_time - start_time
    print(f"Train Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, Time: {elapsed_time:.2f}s")

# 평가 함수: 테스트 데이터로 모델 성능 평가
# 손실과 정확도를 계산하여 출력
def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            correct += (output.argmax(1) == target).sum().item()
    accuracy = correct / len(loader.dataset)
    print(f"Test Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

# 학습 및 평가 루프: 에포크 수만큼 반복하며 학습과 평가 진행
# 각 에포크마다 학습 및 테스트 성능 출력
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 또는 CPU 사용
print(f"Using device: {device}")  # CPU 또는 CUDA 출력

model.to(device)

for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}")
    train(model, train_loader, criterion, optimizer, device)
    test(model, test_loader, criterion, device)
