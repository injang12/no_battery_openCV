import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.function.image_processing import detect_circle, load_and_undistort_image
from app.calibration.calibration import load_calibration_data
from app.function.log_manager import save_result_to_file # Import the standalone logging function
import os

# FastAPI 앱 생성
app = FastAPI()

# --- 템플릿 설정 ---
templates = Jinja2Templates(directory="templates")

# --- 정적 파일 마운트 ---
# 'log' 디렉토리가 없으면 생성
os.makedirs("log", exist_ok=True)
app.mount("/log", StaticFiles(directory="log"), name="log")


# --- 애플리케이션 시작 시 실행 ---
# 캘리브레이션 데이터 로드
mtx, dist, calib_message, _ = load_calibration_data()
print(calib_message) # 서버 시작 시 캘리브레이션 데이터 로드 상태 출력


# --- 엔드포인트 정의 ---
@app.get("/")
async def read_root(request: Request):
    """루트 엔드포인트, index.html 템플릿을 렌더링합니다."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-image/")
async def process_image(
    # --- 이미지 파일 ---
    file: UploadFile = File(..., description="처리할 이미지 파일"),
    # --- 일반 파라미터 ---
    pixel_per_mm: float = Form(0.117, description="픽셀 당 밀리미터"),
    window_size: int = Form(250, description="원을 탐색할 윈도우 크기"),
    approx_x: int = Form(1820, description="예상 원 중심의 X좌표"),
    approx_y: int = Form(760, description="예상 원 중심의 Y좌표"),
    # --- HoughCircles 파라미터 ---
    h_dp: float = Form(1.0, description="HoughCircles: dp"),
    h_min_dist: int = Form(75, description="HoughCircles: minDist"),
    h_param1: int = Form(110, description="HoughCircles: param1"),
    h_param2: int = Form(20, description="HoughCircles: param2"),
    h_min_radius: int = Form(125, description="HoughCircles: minRadius"),
    h_max_radius: int = Form(135, description="HoughCircles: maxRadius")
):
    """
    이미지를 업로드하여 원을 검출하고, 처리된 이미지를 Base64로 인코딩하여 반환합니다.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="잘못된 이미지 파일 형식입니다.")

        # 이미지 왜곡 보정
        if mtx is not None and dist is not None:
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
            x, y, w, h = roi
            image_to_process = undistorted_img[y:y+h, x:x+w]
        else:
            image_to_process = img
        
        hough_params = {
            'dp': h_dp, 'minDist': h_min_dist, 'param1': h_param1,
            'param2': h_param2, 'minRadius': h_min_radius, 'maxRadius': h_max_radius
        }
        
        processed_img, result_text, result_data = detect_circle(
            image=image_to_process,
            pixel_per_mm=pixel_per_mm,
            search_window_size=window_size,
            hough_params=hough_params,
            approx_center=(approx_x, approx_y)
        )

        if processed_img is None:
            raise HTTPException(status_code=500, detail="이미지 처리 중 오류가 발생했습니다.")

        # 원 검출 성공 시 로그 파일에 결과 저장
        if result_data: # result_data가 None이 아닌 경우에만 저장
            save_result_to_file(result_data, file.filename)


        # 결과 이미지를 메모리에서 JPEG 포맷으로 인코딩 후 Base64로 변환
        _, img_encoded = cv2.imencode(".jpg", processed_img)
        img_b64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        # 클라이언트에게 반환할 결과 데이터 구성
        response_data = {
            "result_text": result_text,
            "result_data": result_data,
            "processed_image_b64": img_b64
        }
        
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
