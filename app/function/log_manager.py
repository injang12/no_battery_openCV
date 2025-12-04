import tkinter as tk
import os
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from datetime import datetime

class LogManager:
    def __init__(self, parent_frame):
        """
        LogManager를 초기화하고 부모 프레임에 로그 위젯을 생성합니다.
        :param parent_frame: 로그 위젯이 속할 부모 ttk.Frame
        """
        log_text_frame = ttk.Frame(parent_frame)
        log_text_frame.pack(fill=BOTH, expand=YES, padx=5, pady=5)
        
        self.log_text = tk.Text(log_text_frame, height=10, state=DISABLED, wrap=WORD)
        self.log_text.pack(side=LEFT, fill=BOTH, expand=YES)

        scrollbar = ttk.Scrollbar(log_text_frame, orient=VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        # 로그 색상 설정
        self.log_text.tag_config('info', foreground='black')
        self.log_text.tag_config('warning', foreground='orange')
        self.log_text.tag_config('error', foreground='red')
        self.log_text.tag_config('success', foreground='green')

    def log(self, message, level='info'):
        """
        로그 메시지를 로그 창에 기록합니다.
        :param message: 기록할 메시지
        :param level: 메시지 레벨 ('info', 'warning', 'error', 'success')
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        self.log_text.config(state=NORMAL)
        self.log_text.insert(END, formatted_message, level)
        self.log_text.see(END)
        self.log_text.config(state=DISABLED)

    def save_result_to_file(self, result_data, image_path):
        """
        검사 결과를 매일의 로그 파일에 추가합니다.
        :param result_data: 저장할 결과 데이터 딕셔너리
        :param image_path: 원본 이미지 경로 (파일 이름 포함)
        """
        if not result_data:
            return # 저장할 데이터가 없으면 아무것도 하지 않음

        try:
            # 결과 폴더 생성
            results_dir = "results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # 파일 이름 생성 (년_월_일.txt)
            date_str = datetime.now().strftime("%Y_%m_%d")
            result_filename = f"{date_str}.txt"
            result_filepath = os.path.join(results_dir, result_filename)

            # 저장할 내용 포맷팅
            time_str = datetime.now().strftime("%H:%M:%S")
            base_name = os.path.basename(image_path) # 이미지 파일 이름 추출
            radius = result_data['radius']
            center_x, center_y = result_data['center']
            dx, dy = result_data['offset']
            
            # f-string 포맷팅을 사용하여 소수점 3자리까지 표현
            formatted_line = (
                f"[{time_str}] [{base_name}] Radius: {radius:.3f}, 좌표값: ({center_x:.3f}, {center_y:.3f}), "
                f"offset(mm): dx={dx:.3f}, dy={dy:.3f}\n"
            )

            # 파일에 이어서 쓰기 (append mode)
            with open(result_filepath, "a", encoding="utf-8") as f:
                f.write(formatted_line)

            self.log(f"결과가 '{result_filepath}'에 추가되었습니다.", "success")

        except Exception as e:
            self.log(f"결과 파일 저장 중 오류 발생: {e}", "error")