# 파이토치(PyTorch)에서 CPU와 GPU 사용 시 연산 시간 차이를 확인할 수 있는 간단한 예제 코드
import torch
import time

# 행렬의 크기 설정
size = 10000

# CPU에서의 연산
tensor_cpu = torch.rand(size, size)

start_time_cpu = time.time()
result_cpu = tensor_cpu @ tensor_cpu
end_time_cpu = time.time()

cpu_time = end_time_cpu - start_time_cpu
print(f"CPU에서의 연산 시간: {cpu_time:.5f}초")

# GPU가 사용 가능한 경우 GPU에서의 연산
if torch.cuda.is_available():
    tensor_gpu = tensor_cpu.cuda()

    start_time_gpu = time.time()
    result_gpu = tensor_gpu @ tensor_gpu
    end_time_gpu = time.time()

    gpu_time = end_time_gpu - start_time_gpu
    print(f"GPU에서의 연산 시간: {gpu_time:.5f}초")
else:
    print("CUDA를 사용할 수 없습니다. GPU 연산을 수행할 수 없습니다.")
