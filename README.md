# 개발환경 구성

## Nvidia 그래픽 드라이버 설치

1) 빌드 필수 패키지 설치
```bash
sudo apt update && sudo apt install build-essential
```

2) Compatible 드라이브 버전 확인
```bash
sudo ubuntu-drivers devices 
```

3) 원하는 버전 설치 
```bash
sudo apt install nvidia-driver-535      # 535 버전 설치
sudo reboot                             # 설치후 재부팅
```

4) 설치 확인 
```bash
nvidia-smi
```

## CUDA 설치

1) https://developer.nvidia.com/cuda-toolkit-archive 에서 필요한 버전으로 설치
- CUDA 12.4.1, ubuntu 22.04, runfile(local) 기준
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
```
2) 다운받은 파일 설치

```bash
sudo sh cuda_12.4.1_550.54.15_linux.run
```
- 설치 옵션에서 Driver는 해제(호환 Nvidia 드라이브를 설치했다면)

3) 설치 후 bashrc에서 경로 설정
```bash
sudo nano ~/.bashrc     # nano 에디터로 bashrc 열기
```
열린 bashrc에서 제일 아래에 두줄 추가

```bashrc
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

수정된 bashrc 적용
```bash
source ~/.bashrc
```

4) 아래 명령어 제대로 출력되면 성공
```bash
nvcc -V
```

## cuDNN 설치
1) https://developer.nvidia.com/rdp/cudnn-archive 로 이동 후 로그인 및 동의하고 설치한 CUDA 버전에 맞는 cuDNN 다운로드
2) tar.xz로 압축된 파일을 풀기 (v8.9.6 기준)
```bash
tar -xvf cudnn-linux-x86_64-8.9.6.50_cuda12-archive.tar.xz
```
3) 압축 풀린 파일들을 /usr/local/로 붙여놓고 권한설정
```bash
sudo cp cudnn-linux-x86_64-8.9.6.50_cuda12-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp cudnn-linux-x86_64-8.9.6.50_cuda12-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```
4) 아래 명령어 입력시 제대로 출력되면 성공
```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

## NVIDIA 패키지 저장소를 설정하고 CUDA, cuDNN 설치하는 법
1) NVIDIA의 CUDA 저장소 키를 포함하는 cuda-keyring_1.1-1_all.deb 파일을 다운로드
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
```
- ubuntu2204/x86_64는 설치버전에 맞게 변경
- ex) ubuntu2004/arm64, ubuntu2004/x86_64, ubuntu2404/x86_64, ubuntu2404/sbsa 등

2) dpkg 명령으로 다운로드한 cuda-keyring 패키지를 설치
```bash
sudo dpkg -i cuda-keyring_1.1-1_all.deb
```

3) 로컬 apt 패키지 인덱스를 업데이트
```bash
sudo apt-get update
```

4) CUDA SDK 설치 (가장 최신버전으로 설치되므로 조심)
```bash
sudo apt-get install cuda-toolkit
```
- apt-cache search cuda-toolkit 으로 설치 버전을 조회하고 해당 버전 설치
- ex) sudo apt-get install cuda-toolkit-12-4

5) PATH추가
```bashrc
export PATH="/usr/local/cuda-12.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"
```

6) cuDNN 설치
```bash
sudo apt-get -y install cudnn9-cuda-12
```

7) cuDNN 설치 확인
```bash
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

