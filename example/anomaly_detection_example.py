import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 정상 데이터 생성
np.random.seed(42)
normal_data = np.concatenate([np.random.normal(loc=0, scale=1, size=(500, 2)),
                              np.random.normal(loc=5, scale=1, size=(500, 2))])

# 이상 데이터 생성
anomaly_data = np.random.normal(loc=5, scale=1, size=(50, 2))

# 데이터 결합
data = np.concatenate([normal_data, anomaly_data])

# Isolation Forest 모델 생성 및 학습
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(data)

# 예측 결과 확인
labels = clf.predict(data)
anomaly_indices = np.where(labels == -1)[0]

# 시각화
plt.scatter(data[:, 0], data[:, 1], c='green', label='Normal Data', alpha=0.7)
plt.scatter(data[anomaly_indices, 0], data[anomaly_indices, 1], c='red', label='Anomaly Data', alpha=0.7)
plt.title("Anomaly Detection with Isolation Forest")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
