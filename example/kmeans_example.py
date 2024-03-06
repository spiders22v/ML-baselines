import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 랜덤 데이터 생성
np.random.seed(46)
data = np.concatenate([np.random.normal(loc=0, scale=1, size=(100, 2)),
                       np.random.normal(loc=5, scale=1, size=(100, 2)),
                       np.random.normal(loc=10, scale=1, size=(100, 2))])

# KMeans 모델 생성 및 학습
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)

# 클러스터 중심 확인
centers = kmeans.cluster_centers_
print("Cluster Centers:\n", centers)

# 각 데이터 포인트의 클러스터 할당 확인
labels = kmeans.labels_

# 시각화
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
