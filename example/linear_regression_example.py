import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 샘플 데이터 생성
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 학습 결과 확인
slope = model.coef_[0][0]
intercept = model.intercept_[0]
print("Slope (기울기):", slope)
print("Intercept (절편):", intercept)

# 새로운 데이터에 대한 예측
new_data = np.array([[2.5]])
predicted_value = model.predict(new_data)
print("예측 값:", predicted_value[0][0])

# 시각화
plt.scatter(X, y, alpha=0.7, label='실제 데이터')
plt.plot(X, model.predict(X), color='red', label='선형 회귀 모델')
plt.scatter(new_data[0][0], predicted_value[0][0], color='green', marker='X', s=100, label='새로운 데이터 예측값')
plt.title("Linear Regression Example")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.legend()
plt.show()
