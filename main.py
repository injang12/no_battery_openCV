import cv2
import numpy as np
import base64
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.function.image_processing import detect_circle
from app.calibration.calibration import load_calibration_data, prepare_calibration, process_calibration_image, finalize_and_save_calibration
from app.function.log_manager import save_result_to_file
import os

# FastAPI 앱 생성
app = FastAPI()

# --- 템플릿 설정 ---
templates = Jinja2Templates(directory="templates")

# --- 정적 파일 마운트 ---
os.makedirs("log", exist_ok=True)
app.mount("/log", StaticFiles(directory="log"), name="log")


# --- 애플리케이션 시작 시 실행 ---
mtx, dist = None, None
def load_global_calibration_data():
    """전역 캘리브레이션 변수를 로드하고 갱신하는 함수"""
    global mtx, dist
    loaded_mtx, loaded_dist, message, _ = load_calibration_data()
    mtx = loaded_mtx
    dist = loaded_dist
    print(message)

load_global_calibration_data()


# --- 엔드포인트 정의 ---
@app.get("/")
async def read_root(request: Request):
    """루트 엔드포인트, index.html 템플릿을 렌더링합니다."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/calibration-data")
async def get_calibration_data():
    """현재 로드된 캘리브레이션 데이터를 반환합니다."""
    if mtx is not None and dist is not None:
        return { "matrix": mtx.tolist(), "distortion": dist.tolist() }
    else:
        raise HTTPException(status_code=404, detail="캘리브레이션 데이터가 로드되지 않았습니다.")

@app.websocket("/ws/run-calibration")
async def websocket_run_calibration(websocket: WebSocket):
    """
    웹소켓을 통해 실시간으로 캘리브레이션을 실행하고 진행 상황을 클라이언트에 전송합니다.
    """
    await websocket.accept()
    objpoints = []
    imgpoints = []
    image_size = None

    try:
        images, objp, checkerboard_size, prep_message = prepare_calibration()
        if not images:
            await websocket.send_json({"type": "error", "message": prep_message})
            return

        total_images = len(images)
        await websocket.send_json({"type": "info", "message": f"캘리브레이션을 시작합니다. 총 {total_images}개의 이미지를 처리합니다."})
        
        for i, fname in enumerate(images):
            ret, corners, display_img, size, msg = process_calibration_image(fname, objp, checkerboard_size)
            
            if image_size is None and size is not None:
                image_size = size

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

            # 클라이언트에 진행 상황 전송
            _, img_encoded = cv2.imencode(".jpg", display_img)
            img_b64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
            
            progress_data = {
                "type": "progress",
                "message": msg,
                "image": img_b64,
                "current": i + 1,
                "total": total_images,
                "found": ret
            }
            await websocket.send_json(progress_data)
            await asyncio.sleep(0.1) # UI가 업데이트될 시간을 줍니다.

        # 최종 캘리브레이션 및 저장
        success, final_message, level = finalize_and_save_calibration(objpoints, imgpoints, image_size)
        
        if success:
            # 캘리브레이션 성공 시 전역 변수 갱신
            load_global_calibration_data()
            await websocket.send_json({"type": "success", "message": final_message})
        else:
            await websocket.send_json({"type": "error", "message": final_message})

    except Exception as e:
        await websocket.send_json({"type": "error", "message": f"캘리브레이션 중 예외 발생: {e}"})
    finally:
        await websocket.close()


@app.post("/process-image/")
async def process_image(
    file: UploadFile = File(..., description="처리할 이미지 파일"),
    pixel_per_mm: float = Form(0.117, description="픽셀 당 밀리미터"),
    window_size: int = Form(250, description="원을 탐색할 윈도우 크기"),
    approx_x: int = Form(1820, description="예상 원 중심의 X좌표"),
    approx_y: int = Form(760, description="예상 원 중심의 Y좌표"),
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

        if result_data:
            save_result_to_file(result_data, file.filename)

        _, img_encoded = cv2.imencode(".jpg", processed_img)
        img_b64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        response_data = {
            "result_text": result_text,
            "result_data": result_data,
            "processed_image_b64": img_b64
        }
        
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류가 발생했습니다: {str(e)}")
