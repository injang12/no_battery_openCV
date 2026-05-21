using OpenCvSharp;

namespace NoBatteryOpenCV.Wpf.Models;

public sealed record CalibrationProgress(
    string Message,
    int Current,
    int Total,
    bool Found,
    Mat? DisplayImage);

public sealed record CalibrationRunResult(
    bool Success,
    string Message,
    CalibrationData? Data);
