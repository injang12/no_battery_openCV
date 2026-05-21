using OpenCvSharp;

namespace NoBatteryOpenCV.Wpf.Models;

public sealed record CircleMeasurement(
    double CenterX,
    double CenterY,
    double OffsetXMillimeters,
    double OffsetYMillimeters,
    double Radius);

public sealed record CircleDetectionResult(
    Mat ProcessedImage,
    string ResultText,
    CircleMeasurement? Measurement);
