namespace NoBatteryOpenCV.Wpf.Models;

public sealed record HoughParameters(
    double Dp,
    double MinDist,
    double Param1,
    double Param2,
    int MinRadius,
    int MaxRadius);
