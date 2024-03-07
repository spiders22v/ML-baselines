
This project uses the Google Gemini API to create a simple chatbot with a Flask web interface.

## Prerequisites

Before you start, make sure you have Python and pip installed on your system.

Install required packages:
```bash
pip install flask google-generativeai
```

## Configuration

1. Obtain a [Google Gemini API key](https://aistudio.google.com/) and replace `"YOUR_API_KEY"` with your actual API key in the `genai.configure(api_key="YOUR_API_KEY")` line.

## Running the Application

1. Open a terminal and navigate to the project directory.

2. Run the Flask application:
```bash
python chatbot_gemini.py
```


3. Open your web browser and go to [http://localhost:5000](http://localhost:5000) to interact with the chatbot.

4. (Option) If you want to run the Flask application, run the PyQt5-based frontend:

```bash
python chatbot_gui_pyqt5.py
```

## Usage

1. Type your message in the input field and press "Send" or hit Enter.

2. The chatbot's response will appear in the chat window.

3. Enjoy chatting with the Google Gemini-powered chatbot!

## Notes

- The chat history is displayed in the chat window.

- Feel free to customize the HTML, CSS, and JavaScript files to enhance the user interface.
