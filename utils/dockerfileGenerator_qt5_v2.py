import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QPushButton, QTextEdit, QComboBox, QCheckBox, QFileDialog

class DockerfileGenerator(QWidget):
    def __init__(self):
        super().__init__()       


        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Dockerfile Generator')
        self.setGeometry(100, 100, 400, 800)

        self.label_from = QLabel('FROM:)')
        self.combo_from = QComboBox(self)
        self.combo_from.addItems(['pytorch/pytorch:latest', 'pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel', 'pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime', 'tensorflow/tensorflow:latest','tensorflow/tensorflow:latest-jupyter'])
        
        self.label_dir = QLabel('WORKDIR:')
        self.edit_dir = QLineEdit('/app', self)  # 기본으로 '/app' 입력     
       
        self.label_add = QLabel('ADD:')
        self.edit_add = QLineEdit(self)

        self.label_copy = QLabel('COPY:')
        self.edit_copy = QLineEdit(self)

        self.label_run = QLabel('RUN:')
        self.edit_run = QLineEdit('pip install jupyter', self)  # 기본으로 'pip install jupyter' 입력     

        self.label_cmd = QLabel('CMD:')
        self.edit_cmd = QLineEdit('["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]', self)

        self.label_expose = QLabel('EXPOSE:')
        self.edit_expose = QLineEdit('8888', self)   # 기본으로 '8888' 입력     

        self.label_env = QLabel('ENV:')
        self.edit_env = QLineEdit(self)
        
        # 체크박스 생성
        self.checkbox_numpy = QCheckBox('Install numpy', self)
        self.checkbox_numpy.setChecked(True)  # numpy는 기본으로 선택
        self.checkbox_pandas = QCheckBox('Install pandas', self)
        self.checkbox_matplotlib = QCheckBox('Install matplotlib', self)

        self.output_text = QTextEdit(self)
        self.output_text.setReadOnly(True)

        self.btn_generate = QPushButton('Generate Dockerfile', self)
        self.btn_generate.clicked.connect(self.generate_dockerfile)
        
        self.btn_save_dockerfile = QPushButton('Save Dockerfile', self)
        self.btn_save_dockerfile.clicked.connect(self.save_dockerfile)


        self.btn_build_image = QPushButton('Build Docker Image', self)
        self.btn_build_image.clicked.connect(self.build_docker_image)

        layout = QVBoxLayout()
        layout.addWidget(self.label_from)
        layout.addWidget(self.combo_from)
        
        layout.addWidget(self.label_dir)
        layout.addWidget(self.edit_dir)        
        
        layout.addWidget(self.label_add)
        layout.addWidget(self.edit_add)
        layout.addWidget(self.label_copy)
        layout.addWidget(self.edit_copy)
        layout.addWidget(self.label_run)
        layout.addWidget(self.edit_run)
        layout.addWidget(self.label_cmd)
        layout.addWidget(self.edit_cmd)
        layout.addWidget(self.label_expose)
        layout.addWidget(self.edit_expose)
        layout.addWidget(self.label_env)
        layout.addWidget(self.edit_env)
        
        layout.addWidget(self.checkbox_numpy)
        layout.addWidget(self.checkbox_pandas)
        layout.addWidget(self.checkbox_matplotlib)
        
        layout.addWidget(self.btn_generate)
        
        layout.addWidget(self.btn_save_dockerfile)
        
        layout.addWidget(self.btn_build_image)
        layout.addWidget(self.output_text)
        
        # 위젯에 레이아웃 설정
        self.setLayout(layout)        


    def generate_dockerfile(self):
        dockerfile_content = f'FROM {self.combo_from.currentText()}\n'
        
        if self.edit_dir.text():
            dockerfile_content += f'WORKDIR {self.edit_dir.text()}\n'      


        if self.edit_add.text():
            dockerfile_content += f'ADD {self.edit_add.text()}\n'

        if self.edit_copy.text():
            dockerfile_content += f'COPY {self.edit_copy.text()}\n'

        if self.edit_run.text():
            dockerfile_content += f'RUN {self.edit_run.text()}\n'

        if self.edit_cmd.text():
            dockerfile_content += f'CMD {self.edit_cmd.text()}\n'

        if self.edit_expose.text():
            dockerfile_content += f'EXPOSE {self.edit_expose.text()}\n'

        if self.edit_env.text():
            dockerfile_content += f'ENV {self.edit_env.text()}\n'
            
        if self.checkbox_numpy.isChecked():
            dockerfile_content += f'RUN pip install numpy\n'
            
        if self.checkbox_pandas.isChecked():
            dockerfile_content += f'RUN pip install pandas\n'
            
        if self.checkbox_matplotlib.isChecked():
            dockerfile_content += f'RUN pip install matplotlib\n'

        self.output_text.setPlainText(dockerfile_content.strip())
        
    def save_dockerfile(self):
        dockerfile_content = self.output_text.toPlainText()

        if not dockerfile_content:
            return  # No Dockerfile content to save

        default_file_name = "Dockerfile"
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Dockerfile", default_file_name, "Text Files (*.txt);;All Files (*)", options=options)

        if file_name:
            try:
                with open(file_name, 'w') as dockerfile:
                    dockerfile.write(dockerfile_content)
            except Exception as e:
                print(f"Error saving Dockerfile: {e}")
    
    # def save_dockerfile(self):
    #     dockerfile_content = self.output_text.toPlainText()

    #     if not dockerfile_content:
    #         return  # No Dockerfile content to save

    #     options = QFileDialog.Options()
    #     options |= QFileDialog.DontUseNativeDialog
    #     file_name, _ = QFileDialog.getSaveFileName(self, "Save Dockerfile", "", "Text Files (*.txt);;All Files (*)", options=options)

    #     if file_name:
    #         try:
    #             with open(file_name, 'w') as dockerfile:
    #                 dockerfile.write(dockerfile_content)
    #         except Exception as e:
    #             print(f"Error saving Dockerfile: {e}")

    def build_docker_image(self):
        dockerfile_content = self.output_text.toPlainText()
        
        if not dockerfile_content:
            return  # No Dockerfile content to build

        try:
            # Save Dockerfile content to a temporary file
            with open('Dockerfile', 'w') as dockerfile:
                dockerfile.write(dockerfile_content)

            # Build Docker image using subprocess
            subprocess.run(['docker', 'build', '-t', 'custom-image:latest', '.'])
        except Exception as e:
            print(f"Error building Docker image: {e}")
        finally:
            # Remove the temporary Dockerfile
            subprocess.run(['rm', 'Dockerfile'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Quit the application after image build
            QApplication.instance().quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DockerfileGenerator()
    window.show()
    sys.exit(app.exec_())
