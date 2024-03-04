import pandas as pd
import numpy as np

# 1. 시간 범위 설정
time_range = pd.date_range(start='2024-03-04 00:00', end='2024-03-04 01:00', freq='T')

# 2. 각 센서 데이터 생성
# 센서 A: 매 1분마다 데이터 수집
sensor_a = pd.Series(np.random.rand(len(time_range)), index=time_range)

# 센서 B: 매 2분마다 데이터 수집
sensor_b_index = time_range[::2]  # 2분 간격으로 인덱스 조정
sensor_b = pd.Series(np.random.rand(len(sensor_b_index)), index=sensor_b_index)

# 센서 C: 매 3분마다 데이터 수집
sensor_c_index = time_range[::3]  # 3분 간격으로 인덱스 조정
sensor_c = pd.Series(np.random.rand(len(sensor_c_index)), index=sensor_c_index)

# 3. 데이터 프레임 통합
df = pd.DataFrame(index=time_range)
df['Sensor A'] = sensor_a
df['Sensor B'] = sensor_b.reindex(time_range).interpolate(method='linear')
df['Sensor C'] = sensor_c.reindex(time_range).interpolate(method='linear')

# 4. 결과 확인
df.head(10)  # 결과의 처음 10개 행을 출력

import matplotlib.pyplot as plt

# 데이터 프레임의 각 센서 데이터를 시각화
plt.figure(figsize=(14, 8))
plt.plot(df.index, df['Sensor A'], label='Sensor A', marker='o')
plt.plot(df.index, df['Sensor B'], label='Sensor B', marker='x')
plt.plot(df.index, df['Sensor C'], label='Sensor C', marker='+')

# 그래프 제목과 레이블 설정
plt.title('Sensor Data Interpolation')
plt.xlabel('Time')
plt.ylabel('Sensor Values')
plt.legend()

# 그래프 표시
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
