using System.IO;
using System.IO.Compression;
using System.Text;
using NoBatteryOpenCV.Wpf.Models;

namespace NoBatteryOpenCV.Wpf.Services;

public static class NpzExportService
{
    public static void SaveCalibrationData(CalibrationData data, string outputPath)
    {
        using var archive = ZipFile.Open(outputPath, ZipArchiveMode.Create);

        WriteArray(
            archive,
            "mtx.npy",
            FlattenMatrix(data.Matrix),
            data.Matrix.Length,
            data.Matrix.Length > 0 ? data.Matrix[0].Length : 0);

        WriteArray(
            archive,
            "dist.npy",
            data.Distortion,
            1,
            data.Distortion.Length);
    }

    private static double[] FlattenMatrix(double[][] matrix)
    {
        if (matrix.Length == 0 || matrix.Any(row => row.Length != matrix[0].Length))
        {
            throw new InvalidOperationException("카메라 행렬 데이터가 올바르지 않습니다.");
        }

        return matrix.SelectMany(row => row).ToArray();
    }

    private static void WriteArray(ZipArchive archive, string entryName, double[] values, int rows, int columns)
    {
        if (rows <= 0 || columns <= 0)
        {
            throw new InvalidOperationException($"{entryName} 배열 크기가 올바르지 않습니다.");
        }

        var entry = archive.CreateEntry(entryName, CompressionLevel.Optimal);
        using var stream = entry.Open();
        WriteNpy(stream, values, rows, columns);
    }

    private static void WriteNpy(Stream stream, double[] values, int rows, int columns)
    {
        var header = $"{{'descr': '<f8', 'fortran_order': False, 'shape': ({rows}, {columns}), }}";
        var headerBytes = BuildPaddedHeader(header);

        stream.Write([0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y']);
        stream.WriteByte(1);
        stream.WriteByte(0);
        stream.Write(BitConverter.GetBytes((ushort)headerBytes.Length));
        stream.Write(headerBytes);

        Span<byte> buffer = stackalloc byte[sizeof(double)];
        foreach (var value in values)
        {
            BitConverter.TryWriteBytes(buffer, value);
            stream.Write(buffer);
        }
    }

    private static byte[] BuildPaddedHeader(string header)
    {
        var headerLength = Encoding.ASCII.GetByteCount(header);
        var padding = 16 - ((10 + headerLength + 1) % 16);
        if (padding == 16)
        {
            padding = 0;
        }

        return Encoding.ASCII.GetBytes(header + new string(' ', padding) + "\n");
    }
}
