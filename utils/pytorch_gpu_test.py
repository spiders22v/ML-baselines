"""
PyTorch 및 CUDA/GPU 상태 확인 테스트 스크립트
"""

import sys
import platform

def test_python_version():
    """Python 버전 확인"""
    print(f"🐍 Python 버전: {sys.version}")
    print(f"📋 플랫폼: {platform.platform()}")
    print()

def test_pytorch():
    """PyTorch 설치 및 버전 확인"""
    try:
        import torch
        print(f"🔥 PyTorch 버전: {torch.__version__}")
        print(f"📦 PyTorch 경로: {torch.__file__}")
        return True
    except ImportError as e:
        print(f"❌ PyTorch를 불러올 수 없습니다: {e}")
        return False

def test_cuda():
    """CUDA 지원 및 GPU 상태 확인"""
    try:
        import torch
        
        print("🔍 CUDA 지원 상태:")
        print(f"   - CUDA 사용 가능: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   - CUDA 버전: {torch.version.cuda}")
            print(f"   - cuDNN 버전: {torch.backends.cudnn.version()}")
            print(f"   - GPU 개수: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   - GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
            # 현재 GPU 설정
            if torch.cuda.device_count() > 0:
                current_device = torch.cuda.current_device()
                print(f"   - 현재 GPU: {current_device}")
                
                # 간단한 GPU 연산 테스트
                print("   - GPU 연산 테스트 중...")
                device = torch.device('cuda')
                x = torch.randn(1000, 1000).to(device)
                y = torch.randn(1000, 1000).to(device)
                z = torch.mm(x, y)
                print(f"   - GPU 연산 성공! 결과 텐서 크기: {z.shape}")
        else:
            print("   - CUDA를 사용할 수 없습니다.")
            
    except Exception as e:
        print(f"❌ CUDA 테스트 중 오류 발생: {e}")

def test_torchvision():
    """torchvision 설치 확인"""
    try:
        import torchvision
        print(f"🖼️  torchvision 버전: {torchvision.__version__}")
        return True
    except ImportError:
        print("❌ torchvision이 설치되지 않았습니다.")
        return False

def test_torchaudio():
    """torchaudio 설치 확인"""
    try:
        import torchaudio
        print(f"🎵 torchaudio 버전: {torchaudio.__version__}")
        return True
    except ImportError:
        print("❌ torchaudio가 설치되지 않았습니다.")
        return False

def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("🚀 PyTorch 및 GPU 상태 종합 테스트")
    print("=" * 60)
    print()
    
    # Python 버전 확인
    test_python_version()
    
    # PyTorch 테스트
    if test_pytorch():
        print()
        # CUDA/GPU 테스트
        test_cuda()
        print()
        
        # 추가 라이브러리 테스트
        test_torchvision()
        test_torchaudio()
        print()
        
        print("✅ 모든 테스트가 완료되었습니다!")
    else:
        print("❌ PyTorch가 설치되지 않아 테스트를 계속할 수 없습니다.")
        print("다음 명령어로 PyTorch를 설치하세요:")
        print("   pip install torch torchvision torchaudio")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
