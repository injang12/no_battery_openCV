using System.IO;
using NoBatteryOpenCV.Wpf.Models;

namespace NoBatteryOpenCV.Wpf.Services;

public static class LogManager
{
    public static void SaveResult(CircleMeasurement measurement, string sourceImagePath)
    {
        AppPaths.EnsureCreated();

        var now = DateTime.Now;
        var resultPath = Path.Combine(AppPaths.LogDirectory, $"{now:yyyy_MM_dd}.txt");
        var fileName = Path.GetFileName(sourceImagePath);
        var line =
            $"[{now:HH:mm:ss}] [{fileName}] Radius: {measurement.Radius:F3}, " +
            $"좌표값: ({measurement.CenterX:F3}, {measurement.CenterY:F3}), " +
            $"offset(mm): dx={measurement.OffsetXMillimeters:F3}, dy={measurement.OffsetYMillimeters:F3}{Environment.NewLine}";

        File.AppendAllText(resultPath, line);
    }
}
