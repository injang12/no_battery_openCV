using NoBatteryOpenCV.Wpf.Models;
using OpenCvSharp;

namespace NoBatteryOpenCV.Wpf.Services;

public sealed class CalibrationService
{
    private readonly CalibrationStore _calibrationStore;

    public CalibrationService(CalibrationStore calibrationStore)
    {
        _calibrationStore = calibrationStore;
    }

    public async Task<CalibrationRunResult> RunAsync(
        IProgress<CalibrationProgress>? progress,
        CancellationToken cancellationToken = default)
    {
        var images = _calibrationStore.GetCalibrationImages();
        if (images.Count == 0)
        {
            return new CalibrationRunResult(
                false,
                $"캘리브레이션 이미지를 찾을 수 없습니다.{Environment.NewLine}폴더: {_calibrationStore.CalibrationImagesDirectory}",
                null);
        }

        var checkerboardSize = new Size(14, 12);
        var objectPointTemplate = CreateObjectPoints(checkerboardSize);
        var objectPoints = new List<Point3f[]>();
        var imagePoints = new List<Point2f[]>();
        Size? imageSize = null;

        for (var index = 0; index < images.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var imagePath = images[index];
            using var image = _calibrationStore.LoadCalibrationImage(imagePath);
            imageSize ??= new Size(image.Width, image.Height);
            using var gray = new Mat();
            Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);

            var found = Cv2.FindChessboardCorners(gray, checkerboardSize, out Point2f[] corners);
            var displayImage = image.Clone();
            var message = found
                ? $"이미지 처리 중: {CalibrationStore.ToDisplayName(imagePath)}"
                : $"코너 검출 실패: {CalibrationStore.ToDisplayName(imagePath)}";

            if (found)
            {
                var criteria = new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.MaxIter, 30, 0.001);
                Cv2.CornerSubPix(gray, corners, new Size(11, 11), new Size(-1, -1), criteria);
                Cv2.DrawChessboardCorners(displayImage, checkerboardSize, corners, found);
                objectPoints.Add(objectPointTemplate.ToArray());
                imagePoints.Add(corners);
            }

            progress?.Report(new CalibrationProgress(message, index + 1, images.Count, found, displayImage));
            await Task.Delay(60, cancellationToken);
        }

        if (objectPoints.Count == 0 || imagePoints.Count == 0 || imageSize is null)
        {
            return new CalibrationRunResult(false, "캘리브레이션을 수행하기에 충분한 코너를 찾지 못했습니다.", null);
        }

        var cameraMatrix = new double[3, 3];
        var distortion = new double[5];
        Cv2.CalibrateCamera(objectPoints, imagePoints, imageSize.Value, cameraMatrix, distortion, out Vec3d[] _, out Vec3d[] _);

        var data = new CalibrationData
        {
            Matrix =
            [
                [cameraMatrix[0, 0], cameraMatrix[0, 1], cameraMatrix[0, 2]],
                [cameraMatrix[1, 0], cameraMatrix[1, 1], cameraMatrix[1, 2]],
                [cameraMatrix[2, 0], cameraMatrix[2, 1], cameraMatrix[2, 2]]
            ],
            Distortion = distortion
        };
        await _calibrationStore.SaveAsync(data, cancellationToken);

        return new CalibrationRunResult(
            true,
            $"캘리브레이션 성공!{Environment.NewLine}데이터가 {AppPaths.CalibrationDataPath}에 저장되었습니다.",
            data);
    }

    private static Point3f[] CreateObjectPoints(Size checkerboardSize)
    {
        var points = new List<Point3f>(checkerboardSize.Width * checkerboardSize.Height);
        for (var y = 0; y < checkerboardSize.Height; y++)
        {
            for (var x = 0; x < checkerboardSize.Width; x++)
            {
                points.Add(new Point3f(x, y, 0));
            }
        }

        return points.ToArray();
    }
}
