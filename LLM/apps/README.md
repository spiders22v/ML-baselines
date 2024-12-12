# LLM applications with GUI

이 프로젝트는 두 가지 AI 모델 인터페이스를 제공합니다:
1. **Llama3 모델과의 대화** (Streamlit을 사용)
2. **Google Generative AI 챗봇** (PyQt5를 사용한 GUI)

## 요구 사항

이 코드를 실행하기 위해 필요한 패키지들을 아래 명령어로 설치할 수 있습니다:

```bash
pip install streamlit google-generativeai PyQt5 langchain-ollama streamlit-chat
```

## 1. Llama3 모델과 대화하기 (Streamlit 기반)
이 예제에서는 Llama3 모델을 사용하여 간단한 대화형 챗봇을 구현합니다. streamlit을 사용해 사용자와의 대화 기록을 보여주며, llama3.2 또는 llama3.3 모델을 통해 응답을 생성합니다.

## 요구 사항
이 코드를 실행하기 위해 필요한 패키지들을 아래 명령어로 설치할 수 있습니다:
```bash
pip install streamlit langchain-ollama streamlit-chat
```

### 실행 방법
코드 다운로드: 프로젝트 파일을 다운로드하고, 해당 폴더로 이동합니다.

### 모델 설정
OllamaLLM 인스턴스를 사용하여 모델을 설정합니다. 이 예제에서는 기본적으로 llama3.2 모델을 사용하지만, 필요에 따라 다른 모델로 변경할 수 있습니다.

### 앱 실행
Streamlit 애플리케이션을 실행하려면 아래 명령어를 실행합니다.
```bash
streamlit run llama3_chatbot_app.py
```

### 사용 방법
웹 브라우저에서 Streamlit 인터페이스가 열립니다.
텍스트 상자에 질문을 입력하고, '보내기' 버튼을 눌러 응답을 받습니다.
대화 기록은 화면에 출력됩니다.

## 2. Google Generative AI 챗봇 (PyQt5 기반)
이 예제에서는 PyQt5를 사용하여 Google의 Generative AI API와 연동된 간단한 챗봇 GUI를 구현합니다. 사용자는 텍스트를 입력하고, 챗봇은 이에 대한 응답을 생성합니다.

## 요구 사항
이 코드를 실행하기 위해 필요한 패키지들을 아래 명령어로 설치할 수 있습니다:
```bash
pip google-generativeai PyQt5
```

### 실행 방법
코드 다운로드: 프로젝트 파일을 다운로드하고, 해당 폴더로 이동합니다.

### API 키 설정
YOUR_API_KEY를 실제 Google Generative AI API 키로 교체해야 합니다. 이 키는 Google Cloud Console에서 생성할 수 있습니다.

### 앱 실행
아래 명령어로 앱을 실행합니다:
```bash
python chatbot_gui.py
```

### 사용 방법
챗봇 UI가 열리면 텍스트 상자에 질문을 입력하고 'Send' 버튼을 클릭합니다.
챗봇의 응답이 출력됩니다.

## 3. Google Generative AI 챗봇 (Flask 기반)
이 예제에서는 Llama3 모델을 사용하여 간단한 대화형 챗봇을 구현합니다. streamlit을 사용해 사용자와의 대화 기록을 보여주며, llama3.2 또는 llama3.3 모델을 통해 응답을 생성합니다.

### 필수 패키지 설치
아래 명령어를 통해 필요한 패키지를 설치하세요.
```bash
pip install google-generativeai flask
```

### Google Generative AI API 설정
YOUR_API_KEY를 실제 API 키로 교체합니다. Google Cloud에서 API 키를 발급받은 후 해당 값을 아래 코드에 추가해야 합니다.

```python
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")
```

### 코드 파일 준비
chatbot_gemini.html 파일은 챗봇 사용자 인터페이스(UI)를 위한 HTML 파일로, Flask 애플리케이션의 / 경로에서 렌더링됩니다.
app.py는 Flask 웹 서버를 실행하며, 사용자가 입력한 메시지를 받아서 Google Generative AI 모델을 통해 응답을 생성하고 반환합니다.

### 앱 실행
- app.py 파일을 실행하여 서버를 시작합니다. 이 서버는 로컬에서 실행되며, 브라우저에서 접근할 수 있습니다.
```bash
python app.py
```
- 서버가 실행되면, 브라우저에서 http://127.0.0.1:5000을 열어 챗봇과 상호작용할 수 있습니다.