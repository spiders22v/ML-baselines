import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,QWidget, QTextEdit, QGroupBox, QRadioButton, QButtonGroup, QComboBox, QLabel
import pandas as pd

class DataImputer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):        
        # self.data = None

        self.setWindowTitle('데이터 전처리기: 결측 처리')
        self.setGeometry(100, 100, 600, 600)
        
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.load_button = QPushButton('파일 불러오기', self)
        self.load_button.clicked.connect(self.showDialog)
        
        self.layout.addWidget(self.load_button)

        self.text_output = QTextEdit(self)
        self.text_output.setReadOnly(True)
        
        self.layout.addWidget(self.text_output)
        
        # 그룹박스 1: 결측값 처리 방법     
        self.imputation_groupbox = QGroupBox('결측값 처리 방법', self)
        self.imputation_layout = QVBoxLayout(self.imputation_groupbox)
        
        # 첫번째 Inner 그룹박스
        self.inner_groupbox_1 = QGroupBox('목록 삭제(Listwise)', self)
        self.inner_layout_1 = QHBoxLayout()
        
        self.radio_1_1 = QRadioButton('행(rows)')
        self.radio_1_2 = QRadioButton('열(columns)')
        
        self.inner_layout_1.addWidget(self.radio_1_1)    
        self.inner_layout_1.addWidget(self.radio_1_2)
        
        self.inner_groupbox_1.setLayout(self.inner_layout_1)   
        
        # 기본으로 "Remove Rows" 옵션 체크
        self.radio_1_1.setChecked(True)     
        
        # 두번째 Inner 그룹박스
        self.inner_groupbox_2 = QGroupBox('대표값으로 결측 대체', self)
        self.inner_layout_2 = QHBoxLayout()
        
        self.radio_2_1 = QRadioButton('평균값(mean)')
        self.radio_2_2 = QRadioButton('중앙값(median)')
        self.radio_2_3 = QRadioButton('최빈값(mode)')
        self.radio_2_4 = QRadioButton('최다빈도값(most frequent)')
        
        self.inner_layout_2.addWidget(self.radio_2_1)    
        self.inner_layout_2.addWidget(self.radio_2_2)
        self.inner_layout_2.addWidget(self.radio_2_3)
        self.inner_layout_2.addWidget(self.radio_2_4)
        
        self.inner_groupbox_2.setLayout(self.inner_layout_2)   
        
        # 세번째 Inner 그룹박스
        self.inner_groupbox_3 = QGroupBox('이웃값으로 결측 보간', self)
        self.inner_layout_3 = QHBoxLayout()         # Use QHBoxLayout for horizontal alignment
        
        self.radio_3_2 = QRadioButton('이전 이웃값')
        self.radio_3_3 = QRadioButton('이후 이웃값')
        self.radio_3_1 = QRadioButton('가까운 이웃값 평균')
        self.combo_3_1 = QComboBox()                
        self.display_items = ["이웃 수(초기값: 5)", '1','2','3','4','5','10','20','50','100']    # 사용자에게 보이는 리스트
        self.actual_items = ['5', '1','2','3','4','5','10','20','50','100']      # 실제 리스트                       
        self.combo_3_1.addItems(self.display_items)         # 사용자에게 보이는 리스트 설정               
        self.combo_3_1.actual_items = self.actual_items     # 내부적으로 사용할 실제 리스트 설정
        self.combo_3_1.setEnabled(False)  # 초기에는 비활성화 상태

        self.inner_layout_3.addWidget(self.radio_3_2)
        self.inner_layout_3.addWidget(self.radio_3_3)        
        self.inner_layout_3.addWidget(self.radio_3_1)
        self.inner_layout_3.addWidget(self.combo_3_1)        
        
        self.inner_groupbox_3.setLayout(self.inner_layout_3)         
        
        # 라디오 버튼에 대한 연결 설정
        self.radio_3_1.toggled.connect(lambda: self.enableCombo(self.combo_3_1, self.radio_3_1))
        
        # 콤보박스에서 항목이 선택되었을 때의 이벤트 처리
        # self.combo_3_1.currentIndexChanged.connect(self.print_selected_item)
        
        # QButtonGroup을 사용하여 그룹박스 내 라디오버튼을 하나로 묶음
        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.radio_1_1)
        self.button_group.addButton(self.radio_1_2)
        self.button_group.addButton(self.radio_2_1)
        self.button_group.addButton(self.radio_2_2)
        self.button_group.addButton(self.radio_2_3)
        self.button_group.addButton(self.radio_2_4)
        self.button_group.addButton(self.radio_3_1)
        self.button_group.addButton(self.radio_3_2)
        self.button_group.addButton(self.radio_3_3)
        
        
        # 최상위 그룹박스에 첫 번째, 두 번째 그룹박스 추가
        self.imputation_layout.addWidget(self.inner_groupbox_1)
        self.imputation_layout.addWidget(self.inner_groupbox_2)
        self.imputation_layout.addWidget(self.inner_groupbox_3)
        
        self.imputation_groupbox.setLayout(self.imputation_layout) 

        # 전체 레이아웃 설정       
        self.layout.addWidget(self.imputation_groupbox)
        
        
        # 그룹박스 2: 수정 데이터 저장 형식
        self.save_groupbox = QGroupBox('수정된 데이터 저장 형식', self)
        self.save_layout = QHBoxLayout(self.save_groupbox)
        
        self.save_options = ['csv', 'excel', 'numpy', 'pickle', 'json']
        self.save_buttons = {method: QRadioButton(method) for method in self.save_options}       

        self.save_buttons['csv'].setChecked(True)
        
        for save_buttons in self.save_buttons.values():            
            self.save_layout.addWidget(save_buttons)
            
        self.layout.addWidget(self.save_groupbox)
        
        # 버튼        
        self.check_button = QPushButton("결측값 존재 여부 확인", self)
        self.check_button.clicked.connect(self.check_missing_values)
        self.check_button.setEnabled(False)  # 초기에는 비활성화 상태
        self.layout.addWidget(self.check_button)

        self.process_button = QPushButton('결측값 처리', self)
        self.process_button.clicked.connect(self.process_data)
        self.process_button.setEnabled(False)  # 초기에는 비활성화 상태
        self.layout.addWidget(self.process_button)

        self.save_button = QPushButton('처리된 데이터 파일 저장', self)
        self.save_button.clicked.connect(self.save_processed_data)
        self.save_button.setEnabled(False)  # 초기에는 비활성화 상태
        self.layout.addWidget(self.save_button)        

        self.data = pd.DataFrame()
        self.processed_data = pd.DataFrame()
        # self.selected_imputation_method = 'Remove Rows with Missing Values'

    def showDialog(self):
        # 파일 선택 대화 상자 열기
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, '파일 선택', '', 'CSV 파일 (*.csv);;Excel 파일 (*.xlsx);;NumPy 파일 (*.npy);;pickle 파일 (*.pkl);;json 파일 (*.json)', options=options)

        if file_name:
            # 선택된 파일 읽어오기
            self.load_file(file_name)
            
    def load_file(self, file_name):
        try:
            # CSV 파일인 경우
            if file_name.endswith('.csv'):
                self.data = pd.read_csv(file_name)
            # Excel 파일인 경우
            elif file_name.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(file_name)
            # NumPy 파일인 경우
            elif file_name.endswith('.npy'):
                import numpy as np
                # self.loaded_data = np.load(file_name)
                self.data = pd.DataFrame(np.load(file_name, allow_pickle=True))
            # pickle 파일인 경우
            elif file_name.endswith('.pkl'):
                import pickle
                self.data = pd.read_pickle(file_name)
            # json 파일인 경우
            elif file_name.endswith('.json'):
                self.data = pd.read_json(file_name)
            else:
                raise ValueError("지원되지 않는 파일 형식입니다.")

            # 읽어온 데이터를 출력창에 표시
            self.text_output.clear()
            self.text_output.setPlainText(str(self.data.head()))
            print("데이터 읽어옴")
            self.check_button.setEnabled(True)
            
        except Exception as e:
            self.text_output.clear()
            self.text_output.append(f"파일 불러오기 실패: {e}")
            
    def enableCombo(self, combo, radio):
        combo.setEnabled(radio.isChecked())     

    # 결측 체크
    def check_missing_values(self):
        if self.data is None or self.data.empty:
            self.text_output.setPlainText("데이터가 없습니다.")            
        elif self.data.isnull().values.any():
            missing_values_count = self.data.isnull().sum()
            result_text = "결측값이 존재합니다.\n결측값 개수:\n" + missing_values_count.to_string()
            self.text_output.setPlainText(result_text)
            self.process_button.setEnabled(True)           
        else:
            self.text_output.setPlainText("결측값이 존재하지 않습니다.")

    # 결측 처리
    def process_data(self):
        if not self.data.empty:
            selected_button = self.button_group.checkedButton()
            if selected_button:
                selected_text = selected_button.text()
                print("Selected Radio Button:", selected_text)
                if selected_text == "가까운 이웃값 평균":
                    selected_index = self.combo_3_1.currentIndex()
                    actual_selected_item = self.combo_3_1.actual_items[selected_index]
                    print(f"선택된 이웃 갯수: {actual_selected_item}")      
            

            # 선택된 방법으로 결측값 처리
            # self.impute_missing_values()

            # 수정된 파일로 저장 버튼 활성화
            self.save_button.setEnabled(True)

    def save_processed_data(self):
        if not self.processed_data.empty:
            # 수정된 파일로 저장
            save_file_name, _ = QFileDialog.getSaveFileName(self, 'Save Processed Data', '', 'All Files (*);;CSV Files (*.csv);;NumPy Files (*.npy);;Excel Files (*.xlsx)')
            
            if save_file_name:
                if self.csv_checkbox.isChecked():
                    self.processed_data.to_csv(save_file_name + '.csv', index=False)
                    print(f"Processed data saved as CSV to {save_file_name}.csv")

                if self.numpy_checkbox.isChecked():
                    np.save(save_file_name + '.npy', self.processed_data.values)
                    print(f"Processed data saved as NumPy to {save_file_name}.npy")

                if self.excel_checkbox.isChecked():
                    self.processed_data.to_excel(save_file_name + '.xlsx', index=False)
                    print(f"Processed data saved as Excel to {save_file_name}.xlsx")
        else:
            print("No processed data to save.")

    def impute_missing_values(self):
        if self.selected_imputation_method == 'Remove Rows with Missing Values':
            # 결측값이 있는 행 삭제
            self.processed_data = self.data.dropna()
        else:
            # 결측값 처리
            imputer = self.get_imputer(self.selected_imputation_method)
            self.processed_data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)

    def get_imputer(self, method):
        if method == 'Replace with Mean':
            return pd.DataFrame(self.data.mean()).transpose()
        elif method == 'Replace with Median':
            return pd.DataFrame(self.data.median()).transpose()
        elif method == 'Replace with KNN':
            return pd.DataFrame(self.data.fillna(self.data.median()))  # You can adjust the strategy as needed

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DataImputer()
    window.show()
    sys.exit(app.exec_())
