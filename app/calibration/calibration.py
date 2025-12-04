import os
import glob

import cv2
import numpy as np


# 캘리브레이션 프로세스를 준비
def prepare_calibration(image_path_pattern='app/calibration/cali_images/*.jpg'):
    images = glob.glob(image_path_pattern)
    if not images:
        return None, None, f"'{image_path_pattern}' 경로에서 이미지를 찾을 수 없습니다."

    checkerboard_size = (14, 12)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    
    return images, objp, checkerboard_size, None

# 단일 캘리브레이션 이미지에서 코너를 찾고 결과 출력
def process_calibration_image(image_path, objp, checkerboard_size):
    try:
        with open(image_path, 'rb') as f:
            img_array = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    
    if img is None:
        return False, None, None, None, f"경고: {image_path} 파일을 읽을 수 없습니다. 건너뜀"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    
    display_img = img.copy()
    image_size = (img.shape[1], img.shape[0])

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(display_img, checkerboard_size, corners2, ret)
        return True, corners2, display_img, image_size, f"이미지 처리 중: {os.path.basename(image_path)}"
    
    return False, None, display_img, image_size, f"코너 검출 실패: {os.path.basename(image_path)}"

# 수집된 포인트로 캘리브레이션을 수행하고 결과를 저장
def finalize_and_save_calibration(objpoints, imgpoints, image_size, output_file='app/calibration/calibration_data.npz'):
    if not objpoints or not imgpoints:
        return False, "캘리브레이션을 수행하기에 충분한 코너를 찾지 못했습니다.", "error"

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    if not ret:
        return False, "cv2.calibrateCamera 실패.", "error"

    try:
        np.savez(output_file, mtx=mtx, dist=dist)
        return True, f"캘리브레이션 성공! 데이터가 {output_file}에 저장되었습니다.", "success"
    except Exception as e:
        return False, f"캘리브레이션 데이터를 저장할 수 없습니다: {e}", "error"

# 지정된 경로에서 캘리브레이션 데이터(.npz)를 로드
def load_calibration_data(filepath='app/calibration/calibration_data.npz'):
    try:
        with np.load(filepath) as data:
            mtx = data['mtx']
            dist = data['dist']
            return mtx, dist, "캘리브레이션 데이터가 성공적으로 로드되었습니다.", "success"
    except FileNotFoundError:
        return None, None, f"{filepath} 파일을 찾을 수 없습니다. 이미지는 왜곡 보정 없이 사용됩니다.", "warning"