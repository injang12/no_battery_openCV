import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import cv2
from PIL import Image, ImageTk
from ..calibration.calibration import prepare_calibration, process_calibration_image, finalize_and_save_calibration

class CalibrationWindow(ttk.Toplevel):
    def __init__(self, master, log_callback=print):
        super().__init__(master)
        self.master = master
        self.log = log_callback
        self.title("Camera Calibration")
        self.geometry("800x800")

        # --- 변수 초기화 ---
        self.images = []
        self.image_index = 0
        self.objpoints = []
        self.imgpoints = []
        self.objp = None
        self.checkerboard_size = None
        self.image_size = None
        
        # --- UI 구성 ---
        self.image_label = ttk.Label(self)
        self.image_label.pack(fill=BOTH, expand=YES, padx=10, pady=10)

        self.status_label = ttk.Label(self, text="캘리브레이션을 시작하려면 'Start' 버튼을 누르세요.", anchor=W)
        self.status_label.pack(fill=X, padx=10, pady=5)

        self.start_button = ttk.Button(self, text="Start Calibration", command=self.run_calibration_logic, bootstyle="primary")
        self.start_button.pack(pady=10)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def run_calibration_logic(self):
        """캘리브레이션 프로세스를 시작합니다."""
        self.start_button.config(state=DISABLED)
        self.log("캘리브레이션 준비 중...", "info")

        images, objp, checkerboard_size, err_msg = prepare_calibration()
        
        if err_msg:
            self.log(err_msg, "error")
            self.start_button.config(state=NORMAL)
            return
        
        self.images = images
        self.objp = objp
        self.checkerboard_size = checkerboard_size
        self.image_index = 0
        self.objpoints = []
        self.imgpoints = []
        
        self.log(f"총 {len(self.images)}개의 이미지를 찾았습니다. 캘리브레이션을 시작합니다...", "info")
        self.status_label.config(text=f"총 {len(self.images)}개의 이미지를 찾았습니다. 캘리브레이션을 시작합니다...")
        self.after(100, self._process_next_image)

    # 다음 이미지를 처리하거나, 모든 이미지가 처리된 경우 캘리브레이션을 완료
    def _process_next_image(self):
        if self.image_index >= len(self.images):
            self._finish_calibration()
            return

        image_path = self.images[self.image_index]
        
        found, corners, drawn_img, img_size, message = process_calibration_image(image_path, self.objp, self.checkerboard_size)
        
        self.log(message, "info" if found else "warning")
        self.status_label.config(text=message)
        
        if found:
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners)

        if img_size:
            self.image_size = img_size

        self._display_calib_image(drawn_img)
        self.update_idletasks()
        
        self.image_index += 1
        self.after(200, self._process_next_image)

    # 모든 이미지를 처리한 후, 최종 캘리브레이션을 수행하고 결과를 저장
    def _finish_calibration(self):
        self.status_label.config(text="카메라 행렬 계산 중... 잠시 기다려주세요.")
        self.log("카메라 행렬 계산 중...", "info")
        self.update_idletasks()

        success, message, level = finalize_and_save_calibration(self.objpoints, self.imgpoints, self.image_size)
        
        self.log(message, level)
        
        if success:
            self.master.load_calibration_data()
        
        self.on_close()

    # 캘리브레이션 창에 이미지를 표시
    def _display_calib_image(self, cv_img):
        if cv_img is None: return

        lbl_width = self.image_label.winfo_width()
        lbl_height = self.image_label.winfo_height()
        
        if lbl_width < 2 or lbl_height < 2:
            lbl_width, lbl_height = 800, 600

        h, w = cv_img.shape[:2]
        ratio = min(lbl_width/w, lbl_height/h)
        new_size = (int(w * ratio), int(h * ratio))

        resized_img = cv2.resize(cv_img, new_size, interpolation=cv2.INTER_AREA)

        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def on_close(self):
        self.log("캘리브레이션 창이 닫혔습니다.", "info")
        self.master.calibration_window = None
        self.destroy()