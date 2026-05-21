using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using NoBatteryOpenCV.Wpf.Models;
using NoBatteryOpenCV.Wpf.Services;
using OpenCvSharp.WpfExtensions;
using Mat = OpenCvSharp.Mat;
using WpfPoint = System.Windows.Point;

namespace NoBatteryOpenCV.Wpf;

public partial class MainWindow : Window
{
    private readonly CalibrationStore _calibrationStore = new();
    private readonly ImageProcessingService _imageProcessingService = new();
    private readonly CalibrationService _calibrationService;

    private Mat? _originalImage;
    private string? _currentImagePath;

    public MainWindow()
    {
        InitializeComponent();
        _calibrationService = new CalibrationService(_calibrationStore);

        Loaded += MainWindow_Loaded;
        Closed += MainWindow_Closed;
        RefreshCalibrationPaths();
    }

    private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        try
        {
            var calibrationData = await _calibrationStore.LoadAsync();
            SetCalibrationData(calibrationData);
            SetStatus(calibrationData is null
                ? "캘리브레이션 데이터 없음"
                : "캘리브레이션 데이터 로드됨");
        }
        catch (Exception ex)
        {
            SetStatus("캘리브레이션 로드 실패");
            MessageBox.Show(this, ex.Message, "캘리브레이션 로드 오류", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void MainWindow_Closed(object? sender, EventArgs e)
    {
        _originalImage?.Dispose();
    }

    private void LoadImageButton_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Title = "처리할 이미지 선택",
            Filter = "Image files|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff|All files|*.*"
        };

        if (dialog.ShowDialog(this) != true)
        {
            return;
        }

        try
        {
            var loadedImage = _imageProcessingService.LoadImage(dialog.FileName);
            _originalImage?.Dispose();
            _originalImage = loadedImage;
            _currentImagePath = dialog.FileName;

            SelectedFileTextBlock.Text = dialog.FileName;
            ProcessedImage.Source = null;
            ResultTextBox.Text = string.Empty;
            RenderOriginalPreview();
            SetStatus("이미지 로드됨");
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "이미지 로드 오류", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private async void RunDetectionButton_Click(object sender, RoutedEventArgs e)
    {
        if (_originalImage is null || _currentImagePath is null)
        {
            MessageBox.Show(this, "먼저 이미지를 선택해 주세요.", "이미지 없음", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        DetectionSettings settings;
        try
        {
            settings = ReadDetectionSettings();
        }
        catch (FormatException ex)
        {
            MessageBox.Show(this, ex.Message, "파라미터 오류", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        SetBusy(true, "원 검출 중...");
        ResultTextBox.Text = "처리 중...";

        try
        {
            using var imageForProcessing = _originalImage.Clone();
            var calibrationData = _calibrationStore.Current;
            var result = await Task.Run(() =>
            {
                using var undistortedImage = _imageProcessingService.UndistortIfAvailable(imageForProcessing, calibrationData);
                return _imageProcessingService.DetectCircle(undistortedImage, settings);
            });

            using (result.ProcessedImage)
            {
                ProcessedImage.Source = ToBitmapSource(result.ProcessedImage);
            }

            ResultTextBox.Text = result.ResultText;

            if (result.Measurement is not null)
            {
                LogManager.SaveResult(result.Measurement, _currentImagePath);
            }

            SetStatus(result.Measurement is null ? "검출 결과 없음" : "검출 완료");
        }
        catch (Exception ex)
        {
            ResultTextBox.Text = $"오류: {ex.Message}";
            SetStatus("검출 실패");
        }
        finally
        {
            SetBusy(false);
        }
    }

    private void OpenLogButton_Click(object sender, RoutedEventArgs e)
    {
        AppPaths.EnsureCreated();
        Process.Start(new ProcessStartInfo
        {
            FileName = AppPaths.LogDirectory,
            UseShellExecute = true
        });
    }

    private async void LoadCalibrationButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            var calibrationData = await _calibrationStore.LoadAsync();
            SetCalibrationData(calibrationData);
            SetStatus(calibrationData is null ? "캘리브레이션 데이터 없음" : "캘리브레이션 데이터 로드됨");
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "캘리브레이션 로드 오류", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private async void SaveCalibrationNpzButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            var calibrationData = _calibrationStore.Current ?? await _calibrationStore.LoadAsync();
            if (calibrationData is null)
            {
                MessageBox.Show(this, "저장할 캘리브레이션 데이터가 없습니다.", "데이터 없음", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var dialog = new SaveFileDialog
            {
                Title = "캘리브레이션 .npz 저장",
                FileName = "calibration_data.npz",
                Filter = "NumPy archive (*.npz)|*.npz|All files (*.*)|*.*",
                AddExtension = true,
                DefaultExt = ".npz",
                OverwritePrompt = true
            };

            if (dialog.ShowDialog(this) != true)
            {
                return;
            }

            NpzExportService.SaveCalibrationData(calibrationData, dialog.FileName);
            SetStatus(".npz 저장 완료");
            AppendCalibrationLog($".npz 저장 완료: {dialog.FileName}");
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, ".npz 저장 오류", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private async void SelectCalibrationFolderButton_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog
        {
            Title = "체커보드 이미지 폴더 선택",
            InitialDirectory = Directory.Exists(_calibrationStore.CalibrationImagesDirectory)
                ? _calibrationStore.CalibrationImagesDirectory
                : AppPaths.InstalledCalibrationImagesDirectory
        };

        if (dialog.ShowDialog(this) != true)
        {
            return;
        }

        try
        {
            await _calibrationStore.SetCalibrationImagesDirectoryAsync(dialog.FolderName);
            RefreshCalibrationPaths();
            SetStatus("체커보드 이미지 폴더 선택됨");
            AppendCalibrationLog($"체커보드 이미지 폴더: {dialog.FolderName}");
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "폴더 선택 오류", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private async void RunCalibrationButton_Click(object sender, RoutedEventArgs e)
    {
        SetBusy(true, "캘리브레이션 중...");
        CalibrationLogBox.Clear();
        CalibrationImage.Source = null;
        CalibrationProgressBar.Value = 0;
        CalibrationProgressText.Text = "0%";

        var progress = new Progress<CalibrationProgress>(UpdateCalibrationProgress);

        try
        {
            var cornerColumns = ReadInt(CheckerboardColumnsBox, "체커보드 가로 내부 코너 수");
            var cornerRows = ReadInt(CheckerboardRowsBox, "체커보드 세로 내부 코너 수");
            var result = await Task.Run(() => _calibrationService.RunAsync(progress, cornerColumns, cornerRows));
            AppendCalibrationLog(result.Message);

            if (result.Success)
            {
                SetCalibrationData(result.Data);
                CalibrationProgressBar.Value = 100;
                CalibrationProgressText.Text = "완료";
                SetStatus("캘리브레이션 완료");
            }
            else
            {
                SetStatus("캘리브레이션 실패");
                MessageBox.Show(this, result.Message, "캘리브레이션 실패", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }
        catch (Exception ex)
        {
            AppendCalibrationLog($"캘리브레이션 중 예외 발생: {ex.Message}");
            SetStatus("캘리브레이션 실패");
            MessageBox.Show(this, ex.Message, "캘리브레이션 오류", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            SetBusy(false);
        }
    }

    private void ApproxCoordinateBox_TextChanged(object sender, TextChangedEventArgs e)
    {
        RenderOriginalPreview();
    }

    private void OriginalImage_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (_originalImage is null)
        {
            return;
        }

        var point = GetImageCoordinateFromClick(OriginalImage, e.GetPosition(OriginalImage), _originalImage.Width, _originalImage.Height);
        if (point is null)
        {
            return;
        }

        ApproxXBox.Text = point.Value.X.ToString(CultureInfo.InvariantCulture);
        ApproxYBox.Text = point.Value.Y.ToString(CultureInfo.InvariantCulture);
        RenderOriginalPreview();
    }

    private void RenderOriginalPreview()
    {
        if (_originalImage is null || OriginalImage is null)
        {
            return;
        }

        if (!TryReadInt(ApproxXBox.Text, out var x) || !TryReadInt(ApproxYBox.Text, out var y))
        {
            OriginalImage.Source = ToBitmapSource(_originalImage);
            return;
        }

        using var preview = _imageProcessingService.DrawApproximationMarker(_originalImage, x, y);
        OriginalImage.Source = ToBitmapSource(preview);
    }

    private void UpdateCalibrationProgress(CalibrationProgress progress)
    {
        AppendCalibrationLog(progress.Message);

        if (progress.Total > 0)
        {
            var percent = progress.Current * 100.0 / progress.Total;
            CalibrationProgressBar.Value = percent;
            CalibrationProgressText.Text = $"{Math.Round(percent)}% ({progress.Current}/{progress.Total})";
        }

        if (progress.DisplayImage is not null)
        {
            using (progress.DisplayImage)
            {
                CalibrationImage.Source = ToBitmapSource(progress.DisplayImage);
            }
        }
    }

    private void AppendCalibrationLog(string message)
    {
        CalibrationLogBox.AppendText(message + Environment.NewLine);
        CalibrationLogBox.ScrollToEnd();
    }

    private DetectionSettings ReadDetectionSettings()
    {
        return new DetectionSettings(
            ReadDouble(PixelPerMmBox, "픽셀 당 mm 수"),
            ReadInt(WindowSizeBox, "탐색 윈도우 크기"),
            ReadInt(ApproxXBox, "선택 좌표 X"),
            ReadInt(ApproxYBox, "선택 좌표 Y"),
            new HoughParameters(
                ReadDouble(HDpBox, "해상도 반전율 dp"),
                ReadDouble(HMinDistBox, "중심점 최소 거리 minDist"),
                ReadDouble(HParam1Box, "캐니 엣지 상위 임계값 param1"),
                ReadDouble(HParam2Box, "축적기 임계값 param2"),
                ReadInt(HMinRadiusBox, "최소 반지름 minRadius"),
                ReadInt(HMaxRadiusBox, "최대 반지름 maxRadius")));
    }

    private static double ReadDouble(TextBox textBox, string label)
    {
        if (double.TryParse(textBox.Text, NumberStyles.Float, CultureInfo.CurrentCulture, out var currentValue))
        {
            return currentValue;
        }

        if (double.TryParse(textBox.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out var invariantValue))
        {
            return invariantValue;
        }

        throw new FormatException($"{label} 값이 올바른 숫자가 아닙니다.");
    }

    private static int ReadInt(TextBox textBox, string label)
    {
        if (TryReadInt(textBox.Text, out var value))
        {
            return value;
        }

        throw new FormatException($"{label} 값이 올바른 정수가 아닙니다.");
    }

    private static bool TryReadInt(string text, out int value)
    {
        return int.TryParse(text, NumberStyles.Integer, CultureInfo.CurrentCulture, out value)
               || int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
    }

    private void SetCalibrationData(CalibrationData? calibrationData)
    {
        if (calibrationData is null)
        {
            MatrixTextBox.Text = "사용 가능한 데이터 없음.";
            DistortionTextBox.Text = "사용 가능한 데이터 없음.";
            return;
        }

        MatrixTextBox.Text = FormatMatrix(calibrationData.Matrix);
        DistortionTextBox.Text = FormatDistortion(calibrationData.Distortion);
    }

    private void RefreshCalibrationPaths()
    {
        CalibrationImageFolderText.Text = $"이미지 폴더: {_calibrationStore.CalibrationImagesDirectory}";
        CalibrationPathText.Text =
            $"데이터 저장 위치: {AppPaths.CalibrationDataPath}{Environment.NewLine}" +
            $"기본 이미지 폴더: {AppPaths.InstalledCalibrationImagesDirectory}";
    }

    private static string FormatMatrix(double[][] matrix)
    {
        if (matrix.Length == 0)
        {
            return "사용 가능한 데이터 없음.";
        }

        var builder = new StringBuilder();
        builder.AppendLine("[");
        foreach (var row in matrix)
        {
            builder.Append("  [ ");
            builder.Append(string.Join(", ", row.Select(value => value.ToString("F5", CultureInfo.InvariantCulture).PadLeft(12))));
            builder.AppendLine(" ]");
        }

        builder.Append(']');
        return builder.ToString();
    }

    private static string FormatDistortion(double[] distortion)
    {
        if (distortion.Length == 0)
        {
            return "사용 가능한 데이터 없음.";
        }

        return "[ " + string.Join(", ", distortion.Select(value => value.ToString("F5", CultureInfo.InvariantCulture).PadLeft(12))) + " ]";
    }

    private static BitmapSource ToBitmapSource(Mat image)
    {
        var source = BitmapSourceConverter.ToBitmapSource(image);
        source.Freeze();
        return source;
    }

    private static (int X, int Y)? GetImageCoordinateFromClick(Image image, WpfPoint clickPosition, int pixelWidth, int pixelHeight)
    {
        if (pixelWidth <= 0 || pixelHeight <= 0 || image.ActualWidth <= 0 || image.ActualHeight <= 0)
        {
            return null;
        }

        var scale = Math.Min(image.ActualWidth / pixelWidth, image.ActualHeight / pixelHeight);
        if (scale <= 0)
        {
            return null;
        }

        var displayedWidth = pixelWidth * scale;
        var displayedHeight = pixelHeight * scale;
        var offsetX = (image.ActualWidth - displayedWidth) / 2.0;
        var offsetY = (image.ActualHeight - displayedHeight) / 2.0;

        if (clickPosition.X < offsetX ||
            clickPosition.Y < offsetY ||
            clickPosition.X > offsetX + displayedWidth ||
            clickPosition.Y > offsetY + displayedHeight)
        {
            return null;
        }

        var x = (int)Math.Round((clickPosition.X - offsetX) / scale);
        var y = (int)Math.Round((clickPosition.Y - offsetY) / scale);
        return (Math.Clamp(x, 0, pixelWidth - 1), Math.Clamp(y, 0, pixelHeight - 1));
    }

    private void SetBusy(bool isBusy, string? message = null)
    {
        LoadImageButton.IsEnabled = !isBusy;
        RunDetectionButton.IsEnabled = !isBusy;
        RunCalibrationButton.IsEnabled = !isBusy;
        LoadCalibrationButton.IsEnabled = !isBusy;
        SelectCalibrationFolderButton.IsEnabled = !isBusy;
        SaveCalibrationNpzButton.IsEnabled = !isBusy;
        CheckerboardColumnsBox.IsEnabled = !isBusy;
        CheckerboardRowsBox.IsEnabled = !isBusy;

        if (message is not null)
        {
            SetStatus(message);
        }
    }

    private void SetStatus(string message)
    {
        StatusTextBlock.Text = message;
    }
}
