# AGX ORIN 기준
# llama3.3, 70B 파라미터, 응답시간 66초
# llama3.2,  3B 파라미터, 응답시간 6~11초

import os
import streamlit as st
from langchain_ollama import OllamaLLM
from streamlit_chat import message
import time  # 시간 측정을 위한 모듈

# Streamlit 초기 설정
st.set_page_config(page_title='LLama3 채팅')
st.title('Llama3와 대화하세요')

# 세션 상태 초기화
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'response' not in st.session_state:
    st.session_state['response'] = []

# OllamaLLM 인스턴스 초기화 (필요에 따라 변경 가능)
llm = OllamaLLM(model="llama3.2", temperature=0)

def gptResponse(txt):
    start_time = time.time()  # 시작 시간 측정
    result = llm.invoke(txt)  # Llama 모델 호출
    end_time = time.time()  # 끝 시간 측정
    
    # 처리 시간 계산 및 출력
    elapsed_time = end_time - start_time
    print(f"프롬프트 처리 시간: {elapsed_time:.2f}초")  # 콘솔에 출력
    return result

response_container = st.container()

# 대화 입력 폼
with st.form('convchat', clear_on_submit=True):
    user_input = st.text_input('질문을 입력하세요:')
    btn = st.form_submit_button('보내기')
    if btn and user_input:
        # 사용자 입력 처리
        response = gptResponse(user_input)  # Llama 모델로부터 응답 생성
        st.session_state['generated'].append(user_input)
        st.session_state['response'].append(response)

# 대화 UI 표시
with response_container:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['generated'][i], key=str(i) + '_user', is_user=True,
                avatar_style='fun-emoji', seed='kim')
        message(st.session_state['response'][i], key=str(i),
                avatar_style='bottts', seed='lee')
