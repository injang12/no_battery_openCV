using System.IO;
using NoBatteryOpenCV.Wpf.Models;
using OpenCvSharp;

namespace NoBatteryOpenCV.Wpf.Services;

public sealed class ImageProcessingService
{
    public Mat LoadImage(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var image = Cv2.ImDecode(bytes, ImreadModes.Color);
        if (image.Empty())
        {
            throw new InvalidOperationException("잘못된 이미지 파일 형식입니다.");
        }

        return image;
    }

    public Mat UndistortIfAvailable(Mat image, CalibrationData? calibrationData)
    {
        if (calibrationData is null)
        {
            return image.Clone();
        }

        var (matrix, distortion) = calibrationData.ToMats();
        using (matrix)
        using (distortion)
        {
            var imageSize = new Size(image.Width, image.Height);
            var newCameraMatrix = Cv2.GetOptimalNewCameraMatrix(matrix, distortion, imageSize, 1, imageSize, out var roi);
            using (newCameraMatrix)
            {
                var undistorted = new Mat();
                Cv2.Undistort(image, undistorted, matrix, distortion, newCameraMatrix);

                if (roi.Width <= 0 || roi.Height <= 0)
                {
                    return undistorted;
                }

                var cropped = new Mat(undistorted, roi).Clone();
                undistorted.Dispose();
                return cropped;
            }
        }
    }

    public CircleDetectionResult DetectCircle(Mat image, DetectionSettings settings)
    {
        if (image.Empty())
        {
            throw new InvalidOperationException("처리할 이미지가 없습니다.");
        }

        var processedImage = image.Clone();
        using var gray = new Mat();
        using var blurred = new Mat();

        Cv2.CvtColor(processedImage, gray, ColorConversionCodes.BGR2GRAY);
        Cv2.GaussianBlur(gray, blurred, new Size(5, 5), 1.2);

        var imageCenterX = processedImage.Width / 2;
        var imageCenterY = processedImage.Height / 2;
        Cv2.Line(processedImage, new Point(imageCenterX, 0), new Point(imageCenterX, processedImage.Height), Scalar.Yellow, 2);
        Cv2.Line(processedImage, new Point(0, imageCenterY), new Point(processedImage.Width, imageCenterY), Scalar.Yellow, 2);
        Cv2.Circle(processedImage, new Point(settings.ApproximateX, settings.ApproximateY), 5, Scalar.Red, -1);

        var circles = Cv2.HoughCircles(
            blurred,
            HoughModes.Gradient,
            settings.Hough.Dp,
            settings.Hough.MinDist,
            settings.Hough.Param1,
            settings.Hough.Param2,
            settings.Hough.MinRadius,
            settings.Hough.MaxRadius);

        var resultText = "결과: 이미지에서 원을 찾을 수 없습니다.";
        CircleMeasurement? measurement = null;

        var halfWindowSize = settings.SearchWindowSize / 2.0;
        var bestCircle = circles
            .Where(circle =>
                Math.Abs(circle.Center.X - settings.ApproximateX) < halfWindowSize &&
                Math.Abs(circle.Center.Y - settings.ApproximateY) < halfWindowSize)
            .OrderBy(circle => Distance(circle.Center.X, circle.Center.Y, settings.ApproximateX, settings.ApproximateY))
            .FirstOrDefault();

        if (circles.Length > 0 && bestCircle.Radius <= 0)
        {
            resultText = "결과: 검색 창 내에서 원을 찾을 수\n없습니다.";
        }

        if (bestCircle.Radius > 0)
        {
            var refinedCenter = RefineCircleWithEdges(gray, bestCircle.Center, 250);

            Cv2.Circle(
                processedImage,
                new Point((int)Math.Round(bestCircle.Center.X), (int)Math.Round(bestCircle.Center.Y)),
                (int)Math.Round(bestCircle.Radius),
                Scalar.LimeGreen,
                2);
            Cv2.Circle(
                processedImage,
                new Point((int)Math.Round(bestCircle.Center.X), (int)Math.Round(bestCircle.Center.Y)),
                5,
                Scalar.Blue,
                -1);

            var dxPixels = refinedCenter.X - processedImage.Width / 2.0;
            var dyPixels = refinedCenter.Y - processedImage.Height / 2.0;
            var dxMillimeters = dxPixels * settings.PixelPerMillimeter;
            var dyMillimeters = -dyPixels * settings.PixelPerMillimeter;

            measurement = new CircleMeasurement(
                refinedCenter.X,
                refinedCenter.Y,
                dxMillimeters,
                dyMillimeters,
                bestCircle.Radius);

            resultText =
                $" 중심점: ({refinedCenter.X:F3}, {refinedCenter.Y:F3}){Environment.NewLine}" +
                $" 반지름: {bestCircle.Radius:F1}px{Environment.NewLine}" +
                $" dx={dxMillimeters:F3}mm{Environment.NewLine}" +
                $" dy={dyMillimeters:F3}mm";
        }

        return new CircleDetectionResult(processedImage, resultText, measurement);
    }

    public Mat DrawApproximationMarker(Mat image, int x, int y)
    {
        var preview = image.Clone();
        var lineLength = Math.Max(20, Math.Min(preview.Width, preview.Height) / 60);
        var lineThickness = Math.Max(2, Math.Min(preview.Width, preview.Height) / 900);
        var point = new Point(x, y);

        Cv2.Line(preview, new Point(point.X - lineLength, point.Y), new Point(point.X + lineLength, point.Y), Scalar.Red, lineThickness);
        Cv2.Line(preview, new Point(point.X, point.Y - lineLength), new Point(point.X, point.Y + lineLength), Scalar.Red, lineThickness);
        return preview;
    }

    private static Point2f RefineCircleWithEdges(Mat gray, Point2f initialCenter, int roiSize)
    {
        var x0 = Math.Max((int)(initialCenter.X - roiSize), 0);
        var y0 = Math.Max((int)(initialCenter.Y - roiSize), 0);
        var x1 = Math.Min((int)(initialCenter.X + roiSize), gray.Width);
        var y1 = Math.Min((int)(initialCenter.Y + roiSize), gray.Height);

        if (x1 <= x0 || y1 <= y0)
        {
            return initialCenter;
        }

        using var roi = new Mat(gray, new Rect(x0, y0, x1 - x0, y1 - y0));
        using var edges = new Mat();
        Cv2.Canny(roi, edges, 50, 150);
        Cv2.FindContours(edges, out Point[][] contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxNone);

        if (contours.Length == 0)
        {
            return initialCenter;
        }

        var largestContour = contours.OrderByDescending(contour => Cv2.ContourArea(contour)).First();
        Cv2.MinEnclosingCircle(largestContour, out var center, out _);
        return new Point2f(x0 + center.X, y0 + center.Y);
    }

    private static double Distance(double x1, double y1, double x2, double y2)
    {
        var dx = x1 - x2;
        var dy = y1 - y2;
        return Math.Sqrt(dx * dx + dy * dy);
    }
}
