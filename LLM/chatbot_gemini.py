"""
pip를 이용한 패키지 설치(미설치 경우): 
$ pip install google-generativeai
"""

import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

# 모델 셋업
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)


from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__, static_folder='static')


@app.route('/')
def index():
    return render_template('chatbot_gemini.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_script = request.form['user_script']
    
    response = model.generate_content(user_script)
    
    chatbot_response = response.text

    return jsonify({'chatbot_response': chatbot_response})

if __name__ == '__main__':
    app.run(debug=True)
