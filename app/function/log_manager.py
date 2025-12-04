import os
from datetime import datetime

def save_result_to_file(result_data, image_path):
    """
    검사 결과를 매일의 로그 파일에 추가합니다.
    :param result_data: 저장할 결과 데이터 딕셔너리
    :param image_path: 원본 이미지 경로 (파일 이름 포함)
    """
    if not result_data:
        return # 저장할 데이터가 없으면 아무것도 하지 않음

    try:
        # 결과 폴더 생성 (FastAPI main.py에서 이미 생성하므로 중복 방지)
        results_dir = "log"

        # 파일 이름 생성 (년_월_일.txt)
        date_str = datetime.now().strftime("%Y_%m_%d")
        result_filename = f"{date_str}.txt"
        result_filepath = os.path.join(results_dir, result_filename)

        # 저장할 내용 포맷팅
        time_str = datetime.now().strftime("%H:%M:%S")
        # image_path는 업로드된 파일의 임시 이름이므로, 원본 파일 이름 정보를 별도로 받아야 함.
        # 일단은 image_path를 파일 이름으로 사용
        base_name = os.path.basename(image_path)
        
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

        print(f"[{time_str}] 결과가 '{result_filepath}'에 추가되었습니다.") # 콘솔에 출력

    except Exception as e:
        print(f"[{time_str}] 결과 파일 저장 중 오류 발생: {e}") # 콘솔에 오류 출력