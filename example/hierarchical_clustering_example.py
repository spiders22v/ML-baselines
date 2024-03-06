# hierarchical_clustering_example.py

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 랜덤 데이터 생성
np.random.seed(42)
data = np.concatenate([np.random.normal(loc=0, scale=1, size=(50, 2)),
                       np.random.normal(loc=5, scale=1, size=(50, 2)),
                       np.random.normal(loc=10, scale=1, size=(50, 2))])

# 계층적 군집화 모델 생성 및 학습
agg_cluster = AgglomerativeClustering(n_clusters=3)
labels = agg_cluster.fit_predict(data)

# 덴드로그램 생성
linkage_matrix = linkage(data, method='ward')  # ward 방법 사용
dendrogram(linkage_matrix, truncate_mode='level', p=3, color_threshold=20)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Cluster Distance")
plt.show()

# 군집화 결과 시각화
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.title("Hierarchical Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
