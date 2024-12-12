
---

# Ollama (https://ollama.com/)
대규모 언어 모델(LLM)을 로컬 머신 상에서 실행하기 위한 도구

## 1. Ollama 설치
Ollama는 공식적으로 제공되는 패키지를 통해 설치할 수 있습니다. 아래 명령어를 사용하여 Ollama를 설치하세요.

### Windows

[Download](https://ollama.com/download/OllamaSetup.exe)

### macOS

[Download](https://ollama.com/download/Ollama-darwin.zip)

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Docker

The official [Ollama Docker image](https://hub.docker.com/r/ollama/ollama) `ollama/ollama` is available on Docker Hub.

### Libraries

- [ollama-python](https://github.com/ollama/ollama-python)
- [ollama-js](https://github.com/ollama/ollama-js)


## 2. Ollama 설치 확인
설치가 완료되면, 올라마가 제대로 설치되었는지 확인하기 위해 아래 명령어를 실행하여 버전 확인 가능

```bash
ollama --version
```
## 3. 모델 다운로드 및 설치
### 모델 다운로드(로컬)

```bash
ollama pull llama3.2
```
### 모델 실행 & 채팅:

```bash
ollama run llama3.2
```
### 모델 제거:

```bash
ollama rm llama3.2
```


## 모델 라이브러리
Ollama 모델 목록: [ollama.com/library](https://ollama.com/library 'ollama model library')

다운로드 가능한 모델('24.12.12 기준)
| Model              | Parameters | Size  | Download                         |
| ------------------ | ---------- | ----- | -------------------------------- |
| Llama 3.3          | 70B        | 43GB  | `ollama run llama3.3`            |
| Llama 3.2          | 3B         | 2.0GB | `ollama run llama3.2`            |
| Llama 3.2          | 1B         | 1.3GB | `ollama run llama3.2:1b`         |
| Llama 3.2 Vision   | 11B        | 7.9GB | `ollama run llama3.2-vision`     |
| Llama 3.2 Vision   | 90B        | 55GB  | `ollama run llama3.2-vision:90b` |
| Llama 3.1          | 8B         | 4.7GB | `ollama run llama3.1`            |
| Llama 3.1          | 405B       | 231GB | `ollama run llama3.1:405b`       |
| Phi 3 Mini         | 3.8B       | 2.3GB | `ollama run phi3`                |
| Phi 3 Medium       | 14B        | 7.9GB | `ollama run phi3:medium`         |
| Gemma 2            | 2B         | 1.6GB | `ollama run gemma2:2b`           |
| Gemma 2            | 9B         | 5.5GB | `ollama run gemma2`              |
| Gemma 2            | 27B        | 16GB  | `ollama run gemma2:27b`          |
| Mistral            | 7B         | 4.1GB | `ollama run mistral`             |
| Moondream 2        | 1.4B       | 829MB | `ollama run moondream`           |
| Neural Chat        | 7B         | 4.1GB | `ollama run neural-chat`         |
| Starling           | 7B         | 4.1GB | `ollama run starling-lm`         |
| Code Llama         | 7B         | 3.8GB | `ollama run codellama`           |
| Llama 2 Uncensored | 7B         | 3.8GB | `ollama run llama2-uncensored`   |
| LLaVA              | 7B         | 4.5GB | `ollama run llava`               |
| Solar              | 10.7B      | 6.1GB | `ollama run solar`               |

> [!NOTE]
> 7B 모델을 실행하려면 최소 8GB의 RAM이 필요하고, 13B 모델을 실행하려면 16GB, 33B 모델을 실행하려면 32GB가 필요