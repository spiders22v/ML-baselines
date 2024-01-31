import torch

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    # 현재 사용 중인 GPU의 개수 출력
    print(f"GPU가 {torch.cuda.device_count()}개 사용 가능합니다.")
    
    # 현재 선택된 GPU의 장치 이름 출력
    print(f"현재 선택된 GPU: {torch.cuda.get_device_name(0)}")
    
    # 현재 사용 중인 GPU의 메모리 상태 출력
    print(f"현재 GPU 메모리 상태: {torch.cuda.get_device_properties(0)}")
else:
    print("GPU를 사용할 수 없습니다. CPU 모드로 계속합니다.")