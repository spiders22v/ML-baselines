# GPU vs. CPU 성능 비교
# (1) AMD 5975WX RTX4090             GPU(0.80s) vs CPU(  7.95s)    9.93배 차이
# (2) AMD 3965WX RTX3090             GPU(1.05s) vs CPU(  8.62s)    8.11배 차이 
# (3) i9-13980HX RTX4080(Laptop)     GPU(1.25s) vs CPU( 20.12s)   16.12배 차이 
# (4) AMD 5600 RTX3060               GPU(2.17s) vs CPU( 32.32s)   14.89배 차이 
# (5) NVIDIA AGX ORIN                GPU(5.06s) vs CPU(213.38s)   42.15배 차이

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np

# 랜덤 데이터 생성 (32x32 RGB 이미지 10,000개, 0-9 레이블)
x_train = torch.rand(10000, 3, 32, 32)  # (N, C, H, W)
y_train = torch.randint(0, 10, (10000,))  # 0-9 레이블

# PyTorch Dataset 및 DataLoader 설정
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 복잡한 모델 정의 (CNN)
class LargeModel(nn.Module):
    def __init__(self):
        super(LargeModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 학습 함수 정의
def train_and_time(device):
    model = LargeModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    try:
        for epoch in range(5):  # 5 에포크 동안 학습
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
    except KeyboardInterrupt:
        print("\n학습이 중단되었습니다.")

    end_time = time.time()
    return end_time - start_time

# GPU 사용 여부 확인
gpu_available = torch.cuda.is_available()
if gpu_available:
    print("GPU 사용 가능. 성능 비교를 시작합니다...")
    gpu_time = train_and_time('cuda')
    print(f"GPU 학습 시간: {gpu_time:.2f}초")
else:
    print("GPU를 사용할 수 없습니다.")

# CPU 학습
print("CPU 학습을 시작합니다...")
cpu_time = train_and_time('cpu')
print(f"CPU 학습 시간: {cpu_time:.2f}초")

# 결과 비교
if gpu_available:
    print(f"GPU가 CPU보다 {cpu_time / gpu_time:.2f}배 빠릅니다.")
else:
    print("GPU가 사용 불가능하므로 CPU만 비교하였습니다.")
