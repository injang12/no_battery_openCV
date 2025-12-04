import cv2
import numpy as np

# 이미지 파일을 로드하고, 캘리브레이션 데이터로 왜곡을 보정
def load_and_undistort_image(path, mtx, dist):
    try:
        with open(path, 'rb') as f:
            img_array = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return None, f"파일 읽기 실패: {e}"

    if img is None:
        return None, f"{path} 이미지 디코딩 실패"

    if mtx is not None and dist is not None:
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        return undistorted[y:y+h, x:x+w], None
    else:
        return img, None


# 주어진 이미지의 중앙 영역에서 원을 검출하고, 결과 이미지와 텍스트를 반환
def detect_circle(image, pixel_per_mm, search_window_size, hough_params, approx_center):
    if image is None:
        return None, "처리할 이미지가 없습니다."

    processed_img = image.copy()
    img_h, img_w = processed_img.shape[:2]
    result_data = None

    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.medianBlur(gray, 5)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)

    approx_center_x, approx_center_y = approx_center

    # HoughCircles를 전체 이미지에 대해 실행
    all_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                                   dp=hough_params['dp'],
                                   minDist=hough_params['minDist'],
                                   param1=hough_params['param1'],
                                   param2=hough_params['param2'],
                                   minRadius=hough_params['minRadius'],
                                   maxRadius=hough_params['maxRadius'])

    # 근사치 좌표에 마커 표시
    cv2.line(processed_img, (int(img_w / 2), 0), (int(img_w / 2), img_h), (0, 255, 255), 2)
    cv2.line(processed_img, (0, int(img_h / 2)), (img_w, int(img_h / 2)), (0, 255, 255), 2)
    cv2.circle(processed_img, (approx_center_x, approx_center_y), 5, (0, 0, 255), -1)

    result_text = "결과: 이미지에서 원을 찾을 수 없습니다."

    if all_circles is not None:
        all_circles = np.uint16(np.around(all_circles))
        
        candidate_circles = []
        half_window_size = search_window_size / 2

        # 검색 창 내의 원들을 후보로 추가
        for circle in all_circles[0, :]:
            center_x, center_y = int(circle[0]), int(circle[1])
            if (abs(center_x - approx_center_x) < half_window_size and
                abs(center_y - approx_center_y) < half_window_size):
                candidate_circles.append(circle)
        
        if candidate_circles:
            best_circle = None
            min_dist_to_approx = float('inf')

            # 후보 원들 중에서 가장 가까운 원을 찾음
            for circle in candidate_circles:
                center_x, center_y = int(circle[0]), int(circle[1])
                dist = np.sqrt((center_x - approx_center_x)**2 + (center_y - approx_center_y)**2)
                
                if dist < min_dist_to_approx:
                    min_dist_to_approx = dist
                    best_circle = circle
            
            if best_circle is not None:
                precise_center_x, precise_center_y, precise_radius = best_circle
                
                cv2.circle(processed_img, (precise_center_x, precise_center_y), precise_radius, (0, 255, 0), 2)
                cv2.circle(processed_img, (precise_center_x, precise_center_y), 5, (255, 0, 0), -1)

                img_center_x, img_center_y = img_w // 2, img_h // 2
                dx_pixels = int(precise_center_x) - int(img_center_x)
                dy_pixels = int(precise_center_y) - int(img_center_y)
                dx_mm = dx_pixels * pixel_per_mm
                dy_mm = dy_pixels * pixel_per_mm * (-1.0)

                result_text = (
                    f"Result:\n"
                    f"  Precise Center: ({precise_center_x}, {precise_center_y})\n"
                    f"  Radius: {precise_radius} pixels\n"
                    f"  Offset (mm): dx={dx_mm:.3f}, dy={dy_mm:.3f}"
                )
                result_data = {
                    'center': (precise_center_x, precise_center_y),
                    'offset': (dx_mm, dy_mm),
                    'radius': precise_radius
                }
        else:
            result_text = "결과: 검색 창 내에서 원을 찾을 수 없습니다."
            
    return processed_img, result_text, result_data