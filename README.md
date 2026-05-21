# Camera Calibration

최민규 개인용 C# WPF 기반 데스크톱 카메라 캘리브레이션 애플리케이션입니다. 기존 Python/FastAPI 웹 UI를 대체하며, OpenCvSharp로 카메라 캘리브레이션과 검증용 원 검출을 수행합니다.

## 주요 기능

- 카메라 캘리브레이션 데이터 로드 및 체커보드 이미지 기반 재생성
- 캘리브레이션 결과를 NumPy `.npz` 파일로 원하는 위치에 저장
- 체커보드 내부 코너 수 입력, 기본값 `8 x 8`
- 캘리브레이션 적용 상태 확인을 위한 원 검출
- UI에서 체커보드 이미지 폴더 선택
- 원본 이미지 클릭으로 검증 좌표 입력
- HoughCircles 검증 파라미터 조정
- 결과 로그 저장
- .NET 런타임 포함 self-contained Windows 설치 파일 생성

## 프로젝트 구조

- `NoBatteryOpenCV.sln`: C# 솔루션
- `src/NoBatteryOpenCV.Wpf`: WPF 데스크톱 앱
- `installer`: 설치 패키지에 포함되는 설치 스크립트
- `build-installer.ps1`: 게시 및 설치 EXE 생성 스크립트
- `artifacts/publish/win-x64/build-*`: self-contained 게시 폴더
- `artifacts/installer/CameraCalibrationSetup-win-x64.exe`: 설치 파일

## 빌드

```powershell
.\.tools\dotnet\dotnet.exe build .\NoBatteryOpenCV.sln -c Release
```

로컬 `.tools\dotnet` SDK가 없다면 PATH에 등록된 `dotnet` SDK를 사용할 수 있습니다.

## 설치 파일 생성

```powershell
powershell -ExecutionPolicy Bypass -File .\build-installer.ps1
```

생성 결과:

```text
artifacts\installer\CameraCalibrationSetup-win-x64.exe
```

설치 파일은 관리자 권한으로 승격된 뒤 앱을 `C:\Program Files\Camera Calibration`에 설치하고 공용 시작 메뉴/바탕화면 바로가기를 생성합니다. 앱 실행에 별도 Python, 웹 서버, .NET 런타임 설치가 필요하지 않습니다.

기본 체커보드 이미지는 설치 폴더의 `C:\Program Files\Camera Calibration\CalibrationImages`에 포함됩니다. 앱의 `캘리브레이션` 탭에서 `체커보드 이미지 폴더 선택` 버튼으로 다른 이미지 폴더를 지정할 수 있습니다.

## 앱 데이터 저장 위치

- 캘리브레이션 데이터: `%PROGRAMDATA%\CameraCalibration\calibration_data.json`
- 설정: `%PROGRAMDATA%\CameraCalibration\settings.json`
- 로그: `%PROGRAMDATA%\CameraCalibration\log`
