using System.IO;
using System.Text.Json;
using System.Text.RegularExpressions;
using NoBatteryOpenCV.Wpf.Models;
using OpenCvSharp;

namespace NoBatteryOpenCV.Wpf.Services;

public sealed class CalibrationStore
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };
    private static readonly string[] SupportedImagePatterns =
    [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.tif",
        "*.tiff"
    ];

    public CalibrationStore()
    {
        CalibrationImagesDirectory = LoadSavedImageDirectory() ?? AppPaths.InstalledCalibrationImagesDirectory;
    }

    public CalibrationData? Current { get; private set; }

    public string CalibrationImagesDirectory { get; private set; }

    public async Task<CalibrationData?> LoadAsync(CancellationToken cancellationToken = default)
    {
        AppPaths.EnsureCreated();

        if (!File.Exists(AppPaths.CalibrationDataPath))
        {
            await SeedDefaultCalibrationAsync(cancellationToken);
        }

        if (!File.Exists(AppPaths.CalibrationDataPath))
        {
            Current = null;
            return null;
        }

        await using var stream = File.OpenRead(AppPaths.CalibrationDataPath);
        Current = await JsonSerializer.DeserializeAsync<CalibrationData>(stream, JsonOptions, cancellationToken);
        return Current;
    }

    public async Task SaveAsync(CalibrationData data, CancellationToken cancellationToken = default)
    {
        AppPaths.EnsureCreated();
        await using var stream = File.Create(AppPaths.CalibrationDataPath);
        await JsonSerializer.SerializeAsync(stream, data, JsonOptions, cancellationToken);
        Current = data;
    }

    public async Task SetCalibrationImagesDirectoryAsync(string directory, CancellationToken cancellationToken = default)
    {
        if (!Directory.Exists(directory))
        {
            throw new DirectoryNotFoundException($"체커보드 이미지 폴더를 찾을 수 없습니다: {directory}");
        }

        CalibrationImagesDirectory = directory;
        AppPaths.EnsureCreated();

        var settings = new AppSettings { CalibrationImagesDirectory = directory };
        await using var stream = File.Create(AppPaths.SettingsPath);
        await JsonSerializer.SerializeAsync(stream, settings, JsonOptions, cancellationToken);
    }

    public IReadOnlyList<string> GetCalibrationImages()
    {
        if (!Directory.Exists(CalibrationImagesDirectory))
        {
            return [];
        }

        return SupportedImagePatterns
            .SelectMany(pattern => Directory.EnumerateFiles(CalibrationImagesDirectory, pattern, SearchOption.TopDirectoryOnly))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(NaturalSortKey)
            .ToList();
    }

    public Mat LoadCalibrationImage(string imagePath)
    {
        var bytes = File.ReadAllBytes(imagePath);
        var image = Cv2.ImDecode(bytes, ImreadModes.Color);
        if (image.Empty())
        {
            throw new InvalidOperationException($"캘리브레이션 이미지를 디코딩할 수 없습니다: {imagePath}");
        }

        return image;
    }

    public static string ToDisplayName(string imagePath)
    {
        return Path.GetFileName(imagePath);
    }

    private string? LoadSavedImageDirectory()
    {
        try
        {
            if (!File.Exists(AppPaths.SettingsPath))
            {
                return null;
            }

            var settings = JsonSerializer.Deserialize<AppSettings>(File.ReadAllText(AppPaths.SettingsPath), JsonOptions);
            return !string.IsNullOrWhiteSpace(settings?.CalibrationImagesDirectory)
                   && Directory.Exists(settings.CalibrationImagesDirectory)
                ? settings.CalibrationImagesDirectory
                : null;
        }
        catch
        {
            return null;
        }
    }

    private static async Task SeedDefaultCalibrationAsync(CancellationToken cancellationToken)
    {
        if (!File.Exists(AppPaths.InstalledDefaultCalibrationDataPath))
        {
            return;
        }

        await using var source = File.OpenRead(AppPaths.InstalledDefaultCalibrationDataPath);
        await using var destination = File.Create(AppPaths.CalibrationDataPath);
        await source.CopyToAsync(destination, cancellationToken);
    }

    private static string NaturalSortKey(string value)
    {
        return Regex.Replace(value, @"\d+", match => match.Value.PadLeft(10, '0'));
    }

    private sealed class AppSettings
    {
        public string? CalibrationImagesDirectory { get; init; }
    }
}
