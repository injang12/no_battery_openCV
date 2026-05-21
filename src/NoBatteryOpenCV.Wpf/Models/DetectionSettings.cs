namespace NoBatteryOpenCV.Wpf.Models;

public sealed record DetectionSettings(
    double PixelPerMillimeter,
    int SearchWindowSize,
    int ApproximateX,
    int ApproximateY,
    HoughParameters Hough);
