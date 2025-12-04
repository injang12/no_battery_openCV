import os, cv2, sys
import numpy as np


# --- 설정값 ---
# 픽셀 당 mm (조정 필요)
PIXEL_PER_MM = 0.117
# 딥러닝이 알려준 좌표 주변을 탐색할 영역의 크기 (픽셀)
SEARCH_WINDOW_SIZE = 150
# 결과 이미지를 화면에 표시할지 여부
SHOW_VISUAL_DEBUG = True
# 결과 이미지 디스플레이 크기
DISPLAY_WIDTH = 1400
DISPLAY_HEIGHT = 1050


def find_precise_center(current_img_path, approx_center_x, approx_center_y):
    # --- 1. 이미지 로드 ---
    if not os.path.exists(current_img_path):
        print(f"오류: 이미지 파일을 찾을 수 없습니다. '{current_img_path}'")
        return
        
    img = cv2.imread(current_img_path)
    if img is None:
        print(f"오류: '{current_img_path}' 파일을 로드할 수 없습니다.")
        return

    img_h, img_w = img.shape[:2]

    # --- 2. 탐색 영역(ROI) 설정 ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median_blurred = cv2.medianBlur(gray, 5) # 노이즈 제거

    half_window = SEARCH_WINDOW_SIZE // 2
    roi_x1 = int(max(0, approx_center_x - half_window))
    roi_y1 = int(max(0, approx_center_y - half_window))
    roi_x2 = int(min(img_w, approx_center_x + half_window))
    roi_y2 = int(min(img_h, approx_center_y + half_window))

    # 전체 이미지에서 탐색할 부분만 잘라냄
    search_area = median_blurred[roi_y1:roi_y2, roi_x1:roi_x2]

    if search_area.size == 0:
        print(f"오류: 근사 좌표 ({approx_center_x}, {approx_center_y}) 주변에 유효한 탐색 영역을 설정할 수 없습니다.")
        return

    # --- 3. 탐색 영역 내에서 HoughCircles 실행 ---
    circles_in_roi = cv2.HoughCircles(search_area, cv2.HOUGH_GRADIENT, dp=1, minDist=SEARCH_WINDOW_SIZE//2,
                                      param1=110, param2=25, minRadius=15, maxRadius=50)

    # --- 4. 가장 적합한 원 선택 및 정밀 좌표 계산 ---
    if circles_in_roi is not None:
        circles_in_roi = np.uint16(np.around(circles_in_roi))
        
        best_circle = None
        # ROI의 중심 좌표 (상대 좌표)
        roi_center_x_rel = approx_center_x - roi_x1
        roi_center_y_rel = approx_center_y - roi_y1
        min_dist_to_approx = float('inf')

        # 찾은 원 중에서 ROI 중심(근사 좌표)과 가장 가까운 원을 선택
        for circle in circles_in_roi[0, :]:
            c_x_rel, c_y_rel = circle[0], circle[1]
            # 오버플로우 방지를 위해 int로 캐스팅
            dist = np.sqrt((int(c_x_rel) - int(roi_center_x_rel))**2 + (int(c_y_rel) - int(roi_center_y_rel))**2)
            
            if dist < min_dist_to_approx:
                min_dist_to_approx = dist
                best_circle = circle
        
        if best_circle is not None:
            # 정밀 좌표 계산 (전체 이미지 기준)
            precise_center_x = best_circle[0] + roi_x1
            precise_center_y = best_circle[1] + roi_y1
            precise_radius = best_circle[2]
            
            # --- 5. 결과 출력 ---
            print(f"입력된 근사 좌표: ({approx_center_x}, {approx_center_y})")
            print(f"찾아낸 정밀 좌표: ({precise_center_x}, {precise_center_y}), 반지름: {precise_radius}")

            # 로봇 이동을 위한 mm 오프셋 계산
            img_center_x, img_center_y = int(img_w / 2), int(img_h / 2)
            # 오버플로우 방지를 위해 int로 캐스팅
            dx_pixels = int(precise_center_x) - int(img_center_x)
            dy_pixels = int(precise_center_y) - int(img_center_y)
            dx_mm = dx_pixels * PIXEL_PER_MM
            dy_mm = dy_pixels * PIXEL_PER_MM * (-1.0) # Y축 방향 보정
            print(f"이미지 중심 기준 이동 거리(mm): (dx: {dx_mm:.3f}, dy: {dy_mm:.3f})")

            # --- 6. (선택) 시각적 디버깅 ---
            if SHOW_VISUAL_DEBUG:
                result_img = img.copy()
                # 근사 좌표와 탐색 영역 표시
                cv2.circle(result_img, (approx_center_x, approx_center_y), 5, (0, 0, 255), -1) # 빨간 원 (중심점)
                cv2.rectangle(result_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2) # 빨간 사각형
                # 찾은 정밀 원 표시
                cv2.circle(result_img, (precise_center_x, precise_center_y), precise_radius, (0, 255, 0), 2) # 초록 원
                cv2.circle(result_img, (precise_center_x, precise_center_y), 5, (255, 0, 0), -1) # 파란 중심점

                cv2.imshow("Circle Search Result", cv2.resize(result_img, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return
            
    # 원을 찾지 못한 경우
    print(f"오류: 근사 좌표 ({approx_center_x}, {approx_center_y}) 주변에서 원을 찾지 못했습니다.")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("사용법: python circleToSearch.py <approx_x> <approx_y>")
        print("예시: python circleToSearch.py 1820 760")
        sys.exit(1)

    try:
        # 커맨드 라인 인자로부터 근사 좌표를 정수로 변환하여 받음
        approx_x = int(sys.argv[1])
        approx_y = int(sys.argv[2])
    except ValueError:
        print("오류: x와 y 좌표는 반드시 정수여야 합니다.")
        sys.exit(1)

    # 함수 호출
    find_precise_center(
        current_img_path='image.jpg', 
        approx_center_x=approx_x,
        approx_center_y=approx_y
    )