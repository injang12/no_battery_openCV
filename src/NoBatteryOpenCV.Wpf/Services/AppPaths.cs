using System.IO;

namespace NoBatteryOpenCV.Wpf.Services;

public static class AppPaths
{
    public static string AppDataRoot { get; } =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), "CameraCalibration");

    public static string LogDirectory => Path.Combine(AppDataRoot, "log");

    public static string CalibrationDataPath => Path.Combine(AppDataRoot, "calibration_data.json");

    public static string SettingsPath => Path.Combine(AppDataRoot, "settings.json");

    public static string InstalledCalibrationImagesDirectory => Path.Combine(AppContext.BaseDirectory, "CalibrationImages");

    public static string InstalledDefaultCalibrationDataPath => Path.Combine(AppContext.BaseDirectory, "calibration_data.json");

    public static void EnsureCreated()
    {
        Directory.CreateDirectory(AppDataRoot);
        Directory.CreateDirectory(LogDirectory);
    }
}
