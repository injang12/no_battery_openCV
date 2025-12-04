import cv2
import numpy as np


# 픽셀 당 mm
PIXEL_PER_MM = 0.117

# 고정 4:3 비율로 디스플레이 크기 설정
DISPLAY_WIDTH = 1400
DISPLAY_HEIGHT = 1050
display_size = (DISPLAY_WIDTH, DISPLAY_HEIGHT)

# --- 디버그 모드 설정 ---
DEBUG = False 

# --- 1. 렌즈 왜곡 보정 적용 ---

# 캘리브레이션 데이터 로드
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# 원본 이미지 로드
original_img = cv2.imread("image.jpg")

# 왜곡 보정
h, w = original_img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
undistorted_img = cv2.undistort(original_img, mtx, dist, None, newcameramtx)

# ROI를 사용하여 검은색 테두리 자르기
x, y, w, h = roi
undistorted_img = undistorted_img[y:y+h, x:x+w]


# 왜곡 보정된 이미지 표시
# if DEBUG:
#     cv2.imshow("Original Image", cv2.resize(original_img, display_size))
#     cv2.imshow("Undistorted Image", cv2.resize(undistorted_img, display_size))


# --- 이후 모든 처리는 보정된 이미지를 사용 ---
img = undistorted_img 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if DEBUG:
    cv2.imshow("Grayscale", cv2.resize(gray, display_size))

# 노이즈 제거를 위한 Median Blur (HoughCircles에 유리)
median_blurred = cv2.medianBlur(gray, 5)

if DEBUG:
    cv2.imshow("Median Blurred", cv2.resize(median_blurred, display_size))


# 결과 이미지에 그릴 복사본 생성
result_img = img.copy()

# 이미지의 중심 좌표 계산
img_h, img_w = img.shape[:2]
img_center_x, img_center_y = int(img_w / 2), int(img_h / 2)
if DEBUG:
    print(f"이미지 크기: W={img_w}, H={img_h}")
    print(f"이미지 중심 (pixel): ({img_center_x}, {img_center_y})")
cv2.circle(result_img, (img_center_x, img_center_y), 5, (0, 0, 255), -1) # 이미지 중심을 빨간색으로 표시

# Hough Circle 변환을 사용하여 원 찾기
circles = cv2.HoughCircles(median_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                           param1=110, param2=30, minRadius=20, maxRadius=40)

if circles is not None:
    circles = np.uint16(np.around(circles))
    if DEBUG:
        print(f"총 {len(circles[0])}개의 원을 찾았습니다. 분석 시작...")

    verified_count = 0
    for i, circle in enumerate(circles[0]):
        obj_center_x, obj_center_y, radius = circle[0], circle[1], circle[2]

        verified_count += 1 
        
        # --- 객체 정보 출력 및 표시 ---
        cv2.circle(result_img, (obj_center_x, obj_center_y), radius, (0, 255, 0), 2) # 원 외곽 초록색
        cv2.circle(result_img, (obj_center_x, obj_center_y), 5, (255, 0, 0), -1) # 원 중심 파란색
        cv2.circle(result_img, (obj_center_x, obj_center_y), 15, (0, 255, 0), 2) # 객체 주위에 초록색 원 마커 표시

        # 이미지 중심과 객체 중심을 잇는 선 그리기
        cv2.line(result_img, (img_center_x, img_center_y), (obj_center_x, obj_center_y), (0, 255, 255), 1)
        
        # 픽셀 오프셋 계산
        dx_pixels = int(obj_center_x) - int(img_center_x)
        dy_pixels = int(obj_center_y) - int(img_center_y)

        # mm 오프셋 계산
        dx_mm = dx_pixels * PIXEL_PER_MM
        dy_mm = dy_pixels * PIXEL_PER_MM * (-1.0) # Y축 방향 보정

        print(f"\n--- 로봇 이동 정보 (객체 #{verified_count}) ---")
        print(f"객체 중심 (pixel): ({obj_center_x}, {obj_center_y}), 반지름: {radius}")
        print(f"이동해야 할 거리 (mm): (dx: {dx_mm:.3f}, dy: {dy_mm:.3f})")
        print("--------------------")
else:
    if DEBUG:
        print("원을 찾지 못했습니다.")

cv2.imshow("result", cv2.resize(result_img, display_size))
cv2.waitKey(0)
cv2.destroyAllWindows()