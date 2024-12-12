import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCursor

# Configure the Google Generative AI API
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")

# Set up the model configuration
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

# Define safety settings to control harmful content
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the GenerativeModel
model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    generation_config=generation_config,
    safety_settings=safety_settings
)

class ChatBot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('PyQt5 ChatBot')
        self.setGeometry(100, 100, 400, 500)

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        self.layout = QVBoxLayout()

        # Chat log display area
        self.chatLog = QTextEdit()
        self.chatLog.setReadOnly(True)
        self.layout.addWidget(self.chatLog)

        # User input textbox
        self.textbox = QLineEdit(self)
        self.textbox.returnPressed.connect(self.on_send)
        self.layout.addWidget(self.textbox)

        # Send button
        self.sendButton = QPushButton('Send')
        self.sendButton.clicked.connect(self.on_send)
        self.layout.addWidget(self.sendButton)

        self.centralWidget.setLayout(self.layout)

    def on_send(self):
        userText = self.textbox.text()
        if userText:  # Only process non-empty text
            # Update chat log with user input
            self.update_chat_log(f"You: {userText}\n")
            self.textbox.clear()

            # Generate chatbot response
            botResponse = self.generate_response(userText)
            
            # Update chat log with chatbot response
            self.update_chat_log(f"Bot: \n {botResponse.text}\n  \n \n")

    def update_chat_log(self, message):
        self.chatLog.moveCursor(QTextCursor.End)  # Move the cursor to the end of the text
        self.chatLog.insertPlainText(message)
        self.chatLog.moveCursor(QTextCursor.End)  # Scroll to the bottom

    def generate_response(self, user_input):
        # Implement chatbot response generation logic here
        # This example returns the user's input as the response
        response = model.generate_content(user_input)
        return response

def main():
    app = QApplication(sys.argv)
    chatbot = ChatBot()
    chatbot.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
