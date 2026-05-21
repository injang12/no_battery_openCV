using System.Text.Json.Serialization;
using OpenCvSharp;

namespace NoBatteryOpenCV.Wpf.Models;

public sealed class CalibrationData
{
    [JsonPropertyName("matrix")]
    public double[][] Matrix { get; init; } = [];

    [JsonPropertyName("distortion")]
    public double[] Distortion { get; init; } = [];

    public static CalibrationData FromMats(Mat matrix, Mat distortion)
    {
        var matrixValues = new double[matrix.Rows][];
        for (var row = 0; row < matrix.Rows; row++)
        {
            matrixValues[row] = new double[matrix.Cols];
            for (var col = 0; col < matrix.Cols; col++)
            {
                matrixValues[row][col] = matrix.Get<double>(row, col);
            }
        }

        var distortionValues = new double[distortion.Total()];
        for (var i = 0; i < distortionValues.Length; i++)
        {
            distortionValues[i] = distortion.Get<double>(i);
        }

        return new CalibrationData
        {
            Matrix = matrixValues,
            Distortion = distortionValues
        };
    }

    public (Mat Matrix, Mat Distortion) ToMats()
    {
        if (Matrix.Length == 0 || Matrix.Any(row => row.Length == 0) || Distortion.Length == 0)
        {
            throw new InvalidOperationException("Calibration data is empty.");
        }

        var matrix = new Mat(Matrix.Length, Matrix[0].Length, MatType.CV_64FC1);
        for (var row = 0; row < Matrix.Length; row++)
        {
            for (var col = 0; col < Matrix[row].Length; col++)
            {
                matrix.Set(row, col, Matrix[row][col]);
            }
        }

        var distortion = new Mat(1, Distortion.Length, MatType.CV_64FC1);
        for (var i = 0; i < Distortion.Length; i++)
        {
            distortion.Set(0, i, Distortion[i]);
        }

        return (matrix, distortion);
    }
}
