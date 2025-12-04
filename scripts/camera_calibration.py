import cv2, os
import numpy as np
import glob

# ===========================================================================
# 캘리브레이션 설정
# ===========================================================================
# 1. 체커보드의 내부 코너 개수 (가로, 세로)
CHECKERBOARD = (14, 12)

# 2. 캘리브레이션 이미지가 저장된 경로
IMAGE_PATH = 'cali_images/*.jpg'
# ===========================================================================

print("캘리브레이션을 시작합니다...")
print(f"체커보드 내부 코너 개수: {CHECKERBOARD}")
print(f"이미지 경로: {IMAGE_PATH}")

# 체커보드 코너의 3D 월드 좌표 생성 (z=0으로 가정)
# (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 모든 이미지에서 찾은 3D 월드 좌표와 2D 픽셀 좌표를 저장할 배열
objpoints = []  # 3D 점
imgpoints = []  # 2D 점

# 캘리브레이션 이미지 목록 가져오기
images = glob.glob(IMAGE_PATH)

if not images:
    print(f"오류: '{IMAGE_PATH}' 경로에서 이미지를 찾을 수 없습니다.")
    print("'calibration_images' 폴더에 체커보드 이미지를 넣었는지 확인해주세요.")
else:
    print(f"총 {len(images)}개의 이미지를 찾았습니다.")

    # 각 이미지를 순회하며 코너 검출
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 체커보드 코너 찾기
        # ret: 코너를 찾았으면 True, 못 찾았으면 False
        # corners: 검출된 코너의 픽셀 좌표
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        # 코너를 성공적으로 찾은 경우
        if ret == True:
            print(f"  - {fname}: 코너 검출 성공")
            objpoints.append(objp)

            # 코너의 정확도를 높이기 위한 서브픽셀 연산
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # (선택 사항) 검출된 코너를 이미지에 그려서 보여주기
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(500) # 0.5초 대기
        else:
            print(f"  - {fname}: 코너 검출 실패")

    cv2.destroyAllWindows()

    if objpoints and imgpoints:
        print("\n카메라 캘리브레이션 계산 중...")
        # 카메라 캘리브레이션 수행
        # mtx: 카메라 행렬
        # dist: 왜곡 계수
        # rvecs: 회전 벡터
        # tvecs: 이동 벡터
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        if ret:
            print("\n캘리브레이션 성공!")
            print("\n[카메라 행렬 (Camera Matrix)]")
            print(mtx)
            print("\n[왜곡 계수 (Distortion Coefficients)]")
            print(dist)

            # 캘리브레이션 결과 저장
            output_file = '../calibration_data.npz'
            np.savez(output_file, mtx=mtx, dist=dist)
            print(f"\n캘리브레이션 결과를 '{output_file}' 파일로 저장했습니다.")

            # --- 왜곡 및 원근 보정 테스트 ---
            print("\n왜곡 및 원근 보정 테스트를 시작합니다...")
            test_img_path = images[3]
            test_img = cv2.imread(test_img_path)
            h, w = test_img.shape[:2]
            
            # 1. 렌즈 왜곡 보정
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            undistorted_img = cv2.undistort(test_img, mtx, dist, None, newcameramtx)
            x, y, w, h = roi
            undistorted_img = undistorted_img[y:y+h, x:x+w]

            os.mkdir("cali_result_images")
            cv2.imwrite('cali_result_images/Calibration_image.jpg', undistorted_img)
            print(f"  - 1/2: 렌즈 왜곡 보정 완료 ('Calibration_image.jpg' 저장). 테스트 이미지: {test_img_path}")

            # 2. 원근 보정을 위한 코너 찾기
            gray_undistorted = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
            ret_corners, corners = cv2.findChessboardCorners(gray_undistorted, CHECKERBOARD, None)

            if ret_corners:
                # 소스 포인트 (체커보드의 네 꼭짓점)
                src_pts = np.float32([
                    corners[0][0],
                    corners[CHECKERBOARD[0] - 1][0],
                    corners[(CHECKERBOARD[1] - 1) * CHECKERBOARD[0]][0],
                    corners[CHECKERBOARD[0] * CHECKERBOARD[1] - 1][0]
                ])

                # 목적지 포인트 (결과 이미지 크기 설정)
                # 참고: 체커보드 한 칸의 실제 크기가 다른 경우 아래 값을 수정하세요.
                SQUARE_SIZE = 13.0 # 체커보드 한 칸 실제 크기 (mm)
                width_mm = (CHECKERBOARD[0] - 1) * SQUARE_SIZE
                height_mm = (CHECKERBOARD[1] - 1) * SQUARE_SIZE
                pixel_per_mm = 10 # 1mm당 10픽셀로 결과 해상도 설정
                dst_width = int(width_mm * pixel_per_mm)
                dst_height = int(height_mm * pixel_per_mm)
                
                dst_pts = np.float32([
                    [0, 0], [dst_width - 1, 0],
                    [0, dst_height - 1], [dst_width - 1, dst_height - 1]
                ])

                # 3. 원근 변환 적용
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                perspective_corrected_img = cv2.warpPerspective(undistorted_img, matrix, (dst_width, dst_height))
                cv2.imwrite('result_images/perspective_image.jpg', perspective_corrected_img)
                print("  - 2/2: 원근 변환 적용 완료. ('perspective_image.jpg' 저장)")

            else:
                print("  - 원근 보정 실패: 테스트 이미지에서 코너를 찾을 수 없습니다.")
                cv2.imshow('Original Image', test_img)
                cv2.imshow('1. Undistorted (Lens Corrected)', undistorted_img)

            cv2.destroyAllWindows()

        else:
            print("\n캘리브레이션 계산에 실패했습니다.")

    else:
        print("\n캘리브레이션을 수행하기에 충분한 코너 포인트를 찾지 못했습니다.")
        print("이미지 품질이나 CHECKERBOARD 설정을 확인해주세요.")