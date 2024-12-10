# CPU, GPU 성능 비교
# AGX ORIN CPU(25.15s) vs GPU(11.62s)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np
import time

# 대규모 데이터셋 생성
x_train = np.random.rand(10000, 32, 32, 3).astype('float32')  # 32x32 RGB 이미지 10,000개
y_train = np.random.randint(0, 10, size=(10000,))  # 0-9 레이블

# One-hot 인코딩
y_train = tf.keras.utils.to_categorical(y_train, 10)

# 복잡한 모델 정의
def create_large_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 학습 시간 측정 함수
def train_and_time(device_name):
    with tf.device(device_name):
        model = create_large_model()
        start_time = time.time()
        model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=0)
        end_time = time.time()
    return end_time - start_time

# GPU 사용 여부 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU 사용 가능. 성능 비교를 시작합니다...")
    gpu_time = train_and_time('/GPU:0')
    print(f"GPU 학습 시간: {gpu_time:.2f}초")
else:
    print("GPU를 사용할 수 없습니다.")

# CPU 학습
print("CPU 학습을 시작합니다...")
cpu_time = train_and_time('/CPU:0')
print(f"CPU 학습 시간: {cpu_time:.2f}초")

# 결과 비교
if gpus:
    print(f"GPU가 CPU보다 {cpu_time / gpu_time:.2f}배 빠릅니다.")
else:
    print("GPU가 사용 불가능하므로 CPU만 비교하였습니다.")
