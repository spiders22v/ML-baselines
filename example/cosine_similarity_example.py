from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 샘플 데이터 생성
vector1 = np.array([1, 2, 3, 4, 5])
vector2 = np.array([2, 3, 4, 5, 6])

# 코사인 유사도 계산
similarity_score = cosine_similarity([vector1], [vector2])[0, 0]

# 결과 출력
print("Vector 1:", vector1)
print("Vector 2:", vector2)
print("Cosine Similarity:", similarity_score)
