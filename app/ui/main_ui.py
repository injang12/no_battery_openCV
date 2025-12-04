import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import cv2
from datetime import datetime
from PIL import Image, ImageTk
from tkinter import filedialog
import re
import os

from .calibration_ui import CalibrationWindow
from ..function.log_manager import LogManager
from ..calibration.calibration import load_calibration_data as load_calib_data_logic
from ..function.image_processing import load_and_undistort_image, detect_circle

    
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

class CircleDetectorApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="yeti")
        self.title("Circle Detector")
        self.geometry("2500x1005")

        # --- 변수 선언 ---
        self.original_img = None
        self.processed_img = None
        self.img_path = None
        self.image_list = []
        self.current_image_index = -1
        self.mtx = None
        self.dist = None
        self.calibration_window = None
        self.is_testing = False

        # 파라미터 변수
        self.pixel_per_mm_var = tk.StringVar(value="0.117")
        self.window_size_var = tk.StringVar(value="250")
        self.approx_x_var = tk.StringVar(value="1820")
        self.approx_y_var = tk.StringVar(value="760")

        # HoughCircles 파라미터 변수
        self.h_dp_var = tk.StringVar(value="1")
        self.h_min_dist_var = tk.StringVar(value="75")
        self.h_param1_var = tk.StringVar(value="110")
        self.h_param2_var = tk.StringVar(value="20")
        self.h_min_radius_var = tk.StringVar(value="125")
        self.h_max_radius_var = tk.StringVar(value="135")
        
        # --- UI 구성 ---
        self._setup_ui()
        self.load_calibration_data()

        # 좌표 변경 시 마커 업데이트 (trace는 _setup_ui 호출 후)
        self.approx_x_var.trace_add("write", self._draw_approx_center_on_original)
        self.approx_y_var.trace_add("write", self._draw_approx_center_on_original)

    def _setup_ui(self):
        # 최상위 프레임을 상단과 하단(로그)으로 나눔
        top_frame = ttk.Frame(self)
        top_frame.pack(fill=BOTH, expand=YES, padx=10, pady=(10, 5))

        log_frame = ttk.Labelframe(self, text="Log", height=150)
        log_frame.pack(fill=X, expand=NO, padx=10, pady=(5, 10))
        log_frame.pack_propagate(False)
        self.logger = LogManager(log_frame)

        # --- 상단 프레임 구성 ---
        # 컨트롤 프레임 (좌측)
        control_frame = ttk.Frame(top_frame, width=300)
        control_frame.pack(side=LEFT, fill=Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # 이미지 프레임 (우측)
        image_frame = ttk.Frame(top_frame)
        image_frame.pack(side=LEFT, fill=BOTH, expand=YES)

        # --- 컨트롤 위젯 ---
        self.load_button = ttk.Button(control_frame, text="Load Folder", command=self.load_folder_callback, bootstyle="primary")
        self.load_button.pack(fill=X, pady=5)
        
        # 이미지 네비게이션 프레임
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=X, pady=5)
        nav_frame.columnconfigure(0, weight=1)
        nav_frame.columnconfigure(1, weight=1)

        self.prev_button = ttk.Button(nav_frame, text="< Previous", command=self.prev_image_callback, state=DISABLED)
        self.prev_button.grid(row=0, column=0, sticky=EW, padx=(0, 2))

        self.next_button = ttk.Button(nav_frame, text="Next >", command=self.next_image_callback, state=DISABLED)
        self.next_button.grid(row=0, column=1, sticky=EW, padx=(2, 0))

        # 검사 버튼 프레임
        detection_frame = ttk.Frame(control_frame)
        detection_frame.pack(fill=X, pady=5)
        detection_frame.columnconfigure(0, weight=1)
        detection_frame.columnconfigure(1, weight=1)

        self.start_button = ttk.Button(detection_frame, text="Start Detection", command=self.start_detection_callback, state=DISABLED, bootstyle="success")
        self.start_button.grid(row=0, column=0, sticky=EW, padx=(0, 2))

        self.test_button = ttk.Button(detection_frame, text="Test Detection", command=self.test_detection_callback, state=DISABLED, bootstyle="info")
        self.test_button.grid(row=0, column=1, sticky=EW, padx=(2, 0))

        self.image_info_label = ttk.Label(control_frame, text="No folder loaded.", anchor=CENTER)
        self.image_info_label.pack(fill=X, pady=5)

        self.calib_button = ttk.Button(control_frame, text="Run Calibration", command=self.open_calibration_window, bootstyle="secondary")
        self.calib_button.pack(fill=X, pady=(0, 10))

        param_frame = ttk.Labelframe(control_frame, text="Parameters")
        param_frame.pack(fill=X, pady=10)

        ttk.Label(param_frame, text="Pixel per MM:").pack(anchor=W, padx=5)
        ttk.Entry(param_frame, textvariable=self.pixel_per_mm_var).pack(fill=X, padx=5, pady=(0, 5))

        ttk.Label(param_frame, text="Search Window Size:").pack(anchor=W, padx=5)
        ttk.Entry(param_frame, textvariable=self.window_size_var).pack(fill=X, padx=5, pady=(0, 5))

        hough_frame = ttk.Labelframe(control_frame, text="HoughCircles Parameters")
        hough_frame.pack(fill=X, pady=10, padx=5)

        # dp
        ttk.Label(hough_frame, text="dp:").pack(anchor=W)
        ttk.Entry(hough_frame, textvariable=self.h_dp_var).pack(fill=X, pady=(0, 5))
        # minDist
        ttk.Label(hough_frame, text="minDist:").pack(anchor=W)
        ttk.Entry(hough_frame, textvariable=self.h_min_dist_var).pack(fill=X, pady=(0, 5))
        # param1
        ttk.Label(hough_frame, text="param1:").pack(anchor=W)
        ttk.Entry(hough_frame, textvariable=self.h_param1_var).pack(fill=X, pady=(0, 5))
        # param2
        ttk.Label(hough_frame, text="param2:").pack(anchor=W)
        ttk.Entry(hough_frame, textvariable=self.h_param2_var).pack(fill=X, pady=(0, 5))
        # minRadius
        ttk.Label(hough_frame, text="minRadius:").pack(anchor=W)
        ttk.Entry(hough_frame, textvariable=self.h_min_radius_var).pack(fill=X, pady=(0, 5))
        # maxRadius
        ttk.Label(hough_frame, text="maxRadius:").pack(anchor=W)
        ttk.Entry(hough_frame, textvariable=self.h_max_radius_var).pack(fill=X, pady=(0, 5))
        
        self.result_label = ttk.Label(control_frame, text="Result:", justify=LEFT, anchor=W)
        self.result_label.pack(fill=X, pady=5)

        # --- 이미지 표시 위젯 ---
        original_image_frame = ttk.Labelframe(image_frame, text="Original Image")
        original_image_frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=(0, 5))
        original_image_frame.pack_propagate(False)
        self.original_image_label = ttk.Label(original_image_frame)
        self.original_image_label.pack(fill=BOTH, expand=YES)
        self.original_image_label.bind("<Button-1>", self._on_image_click)

        processed_image_frame = ttk.Labelframe(image_frame, text="Processed Image")
        processed_image_frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=(5, 0))
        processed_image_frame.pack_propagate(False)
        self.processed_image_label = ttk.Label(processed_image_frame)
        self.processed_image_label.pack(fill=BOTH, expand=YES)

    def _on_image_click(self, event):
        if self.original_img is None: return
        widget_w, widget_h = self.original_image_label.winfo_width(), self.original_image_label.winfo_height()
        orig_h, orig_w = self.original_img.shape[:2]
        if orig_w == 0 or orig_h == 0: return
        ratio = min(widget_w / orig_w, widget_h / orig_h)
        resized_w, resized_h = int(orig_w * ratio), int(orig_h * ratio)
        pad_x, pad_y = (widget_w - resized_w) / 2, (widget_h - resized_h) / 2
        click_x_on_resized, click_y_on_resized = event.x - pad_x, event.y - pad_y
        if not (0 <= click_x_on_resized < resized_w and 0 <= click_y_on_resized < resized_h): return
        original_x, original_y = int(click_x_on_resized / ratio), int(click_y_on_resized / ratio)
        self.approx_x_var.set(str(original_x))
        self.approx_y_var.set(str(original_y))
        self.log(f"클릭한 좌표 위치: ({original_x}, {original_y})", "info")

    def log(self, message, level='info'): self.logger.log(message, level)
    def load_calibration_data(self):
        self.mtx, self.dist, message, level = load_calib_data_logic()
        self.log(message, level)

    def _display_image(self, cv_img, label):
        if cv_img is None:
            label.config(image=''); label.image = None; return
        lbl_width, lbl_height = label.winfo_width(), label.winfo_height()
        if lbl_width < 2 or lbl_height < 2: lbl_width, lbl_height = 600, 600
        h, w = cv_img.shape[:2]
        ratio = min(lbl_width/w, lbl_height/h) if w > 0 and h > 0 else 1
        new_size = (int(w * ratio), int(h * ratio))
        resized_img = cv2.resize(cv_img, new_size, interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        label.config(image=img_tk); label.image = img_tk

    def load_folder_callback(self):
        folder_path = filedialog.askdirectory(title="Select a Folder")
        if not folder_path: return
        self.log(f"폴더 로드 중: {folder_path}", "info")
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        try:
            self.image_list = sorted([
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ], key=natural_sort_key)
        except Exception as e:
            self.log(f"폴더를 읽는 중 오류 발생: {e}", "error"); return
        if not self.image_list:
            self.log("폴더에 이미지 파일이 없습니다.", "warning"); self.current_image_index = -1
            self.image_info_label.config(text="No images in folder."); return
        self.current_image_index = 0
        self._load_image_by_index(self.current_image_index)

    def _load_image_by_index(self, index):
        if not (0 <= index < len(self.image_list)): return
        self.img_path = self.image_list[index]
        self.log(f"이미지 로드 중: {self.img_path}", "info")
        self.original_img, err_msg = load_and_undistort_image(self.img_path, self.mtx, self.dist)
        if err_msg:
            self.log(err_msg, "error"); self.original_img = None
        self.display_original_image(); self._display_image(None, self.processed_image_label)
        self.result_label.config(text="Result:")
        is_ready = self.original_img is not None
        self.start_button.config(state=NORMAL if is_ready else DISABLED)
        self.test_button.config(state=NORMAL if is_ready else DISABLED)
        self.prev_button.config(state=NORMAL if self.current_image_index > 0 else DISABLED)
        self.next_button.config(state=NORMAL if self.current_image_index < len(self.image_list) - 1 else DISABLED)
        filename = os.path.basename(self.img_path)
        info_text = f"Image {self.current_image_index + 1} / {len(self.image_list)}\n{filename}"
        self.image_info_label.config(text=info_text)
        if self.original_img is not None: self.log("이미지 로드 및 왜곡 보정 완료.", "success")
        else: self.log("이미지를 표시할 수 없습니다.", "error")

    def next_image_callback(self):
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1; self._load_image_by_index(self.current_image_index)

    def prev_image_callback(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1; self._load_image_by_index(self.current_image_index)

    def start_detection_callback(self): self._perform_detection()

    def _perform_detection(self):
        if self.original_img is None: self.log("이미지를 로드해주세요.", "error"); return False
        try:
            hough_params = {'dp': float(self.h_dp_var.get()), 'minDist': int(self.h_min_dist_var.get()), 'param1': int(self.h_param1_var.get()), 'param2': int(self.h_param2_var.get()), 'minRadius': int(self.h_min_radius_var.get()), 'maxRadius': int(self.h_max_radius_var.get())}
            approx_x, approx_y = int(self.approx_x_var.get()), int(self.approx_y_var.get())
            pixel_per_mm, search_window_size = float(self.pixel_per_mm_var.get()), int(self.window_size_var.get())
        except (ValueError, TypeError): self.log("유효하지 않은 파라미터 값입니다.", "error"); return False
        img_h, img_w = self.original_img.shape[:2]
        if not (0 <= approx_x < img_w and 0 <= approx_y < img_h):
            self.log(f"좌표({approx_x}, {approx_y})가 이미지 크기({img_w}x{img_h})를 벗어났습니다.", "error"); return False
        self.log("원 검출 시작...", "info")
        self.processed_img, result_text, result_data = detect_circle(self.original_img, pixel_per_mm, search_window_size, hough_params, (approx_x, approx_y))
        self.result_label.config(text=result_text); self.display_processed_image(); self.log("원 검출 완료.", "success")
        self.logger.save_result_to_file(result_data, self.img_path)
        return result_data is not None

    def test_detection_callback(self):
        if self.is_testing:
            self.is_testing = False
            self.test_button.config(text="Test Detection")
            self.log("자동 검사를 중지했습니다.", "info")
        else:
            self.is_testing = True
            self.test_button.config(text="Stop Test")
            self.log("자동 검사를 시작합니다...", "info")
            self._run_test_step()

    def _run_test_step(self):
        if not self.is_testing: return

        self.log(f"{self.current_image_index + 1}/{len(self.image_list)} 이미지 검사 중...", "info")
        
        if self._perform_detection():
            if self.current_image_index == len(self.image_list) - 1:
                self.is_testing = False
                self.test_button.config(text="Test Detection")
                self.log("모든 이미지 검사 완료.", "success")
                return
            
            self.after(10, self._advance_to_next_step)
        else:
            self.is_testing = False
            self.test_button.config(text="Test Detection")
            self.log(f"{os.path.basename(self.img_path)}에서 원 검출 실패. 검사를 중지합니다.", "warning")

    def _advance_to_next_step(self):
        if not self.is_testing: return
        self.current_image_index += 1
        self._load_image_by_index(self.current_image_index)
        self.after(10, self._run_test_step)

    def display_original_image(self, event=None):
        if self.original_img is not None: self._draw_approx_center_on_original() 
        else: self._display_image(None, self.original_image_label)

    def _draw_approx_center_on_original(self, *args):
        if self.original_img is None: self._display_image(None, self.original_image_label); return
        try:
            approx_x, approx_y = int(self.approx_x_var.get()), int(self.approx_y_var.get())
        except ValueError:
            self._display_image(self.original_img, self.original_image_label); return
        img_with_marker = self.original_img.copy()
        img_h, img_w = img_with_marker.shape[:2]
        if 0 <= approx_x < img_w and 0 <= approx_y < img_h:
            cv2.line(img_with_marker, (approx_x - 20, approx_y), (approx_x + 20, approx_y), (0, 0, 255), 2)
            cv2.line(img_with_marker, (approx_x, approx_y - 20), (approx_x, approx_y + 20), (0, 0, 255), 2)
            cv2.circle(img_with_marker, (approx_x, approx_y), 5, (0, 255, 255), -1)
        else:
            self.log(f"입력된 좌표({approx_x}, {approx_y})가 이미지 범위({img_w}x{img_h})를 벗어납니다.", "warning")
        self._display_image(img_with_marker, self.original_image_label)

    def display_processed_image(self, event=None): self._display_image(self.processed_img, self.processed_image_label)
    def open_calibration_window(self):
        if self.calibration_window and self.calibration_window.winfo_exists(): self.calibration_window.focus()
        else: self.calibration_window = CalibrationWindow(self, log_callback=self.log)
    
if __name__ == '__main__':
    app = CircleDetectorApp()
    app.mainloop()
