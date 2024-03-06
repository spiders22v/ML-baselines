# CPU와 GPU에서 각각 두 행렬을 곱한 연산 시간을 측정하고 결과를 비교합니다. GPU가 사용 가능한 환경에서는 GPU를 사용하는 것이 더 빠를 것
import torch
import time

# 텐서 생성 및 초기화
cpu_tensor = torch.rand(1000, 1000)
gpu_tensor = cpu_tensor.to('cuda')

# CPU에서의 텐서 연산 시간 측정
start_time_cpu = time.time()
result_cpu = cpu_tensor.mm(cpu_tensor.t())
end_time_cpu = time.time()
elapsed_time_cpu = end_time_cpu - start_time_cpu
print(f"CPU Time: {elapsed_time_cpu:.6f} seconds")

# GPU에서의 텐서 연산 시간 측정
start_time_gpu = time.time()
result_gpu = gpu_tensor.mm(gpu_tensor.t())
end_time_gpu = time.time()
elapsed_time_gpu = end_time_gpu - start_time_gpu
print(f"GPU Time: {elapsed_time_gpu:.6f} seconds")

# CPU와 GPU의 연산 결과 비교
print("Result Difference:", torch.max(torch.abs(result_cpu - result_gpu)))
