using System;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;

const int rows = 32;
const int cols = 64;
const int ggmlTypeF32 = 0;
long sourceBytes = rows * cols * sizeof(float);

static float ExpectedValue(int row, int col) => (row * 1000) + col;

static GgmlBackendType ParseBackend(string[] args)
{
    string backend = args.Length == 0 ? "cpu" : args[0].ToLowerInvariant();
    return backend switch
    {
        "cpu" or "ggml_cpu" => GgmlBackendType.Cpu,
        "cuda" or "ggml_cuda" => GgmlBackendType.Cuda,
        _ => throw new ArgumentException("Unknown backend. Use cpu or cuda.")
    };
}

float[] source = new float[rows * cols];
for (int row = 0; row < rows; ++row)
{
    for (int col = 0; col < cols; ++col)
    {
        source[(row * cols) + col] = ExpectedValue(row, col);
    }
}

IntPtr weightData = GgmlBasicOps.AlignedAlloc(sourceBytes);
Marshal.Copy(source, 0, weightData, source.Length);

try
{
    GgmlBackendType backendType = ParseBackend(args);
    var context = new GgmlContext(new[] { 0 }, backendType);
    var allocator = new GgmlAllocator(context, 0);

    using var result = new Tensor(allocator, DType.Float32, 2, cols);
    using var indices = new Tensor(allocator, DType.Int32, 2);
    indices.SetElementsAsInt(new[] { 5, 1 });

    GgmlBasicOps.GetRowsQuant(result, weightData, ggmlTypeF32, cols, rows, sourceBytes, indices);

    float[] actual = result.GetElementsAsFloat(2 * cols);
    int[] expectedRows = { 5, 1 };
    for (int row = 0; row < expectedRows.Length; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            float expected = ExpectedValue(expectedRows[row], col);
            float observed = actual[(row * cols) + col];
            if (Math.Abs(observed - expected) > 1e-5f)
            {
                throw new InvalidOperationException(
                    $"Mismatch at row {row}, col {col}: got {observed}, expected {expected}.");
            }
        }
    }

    Console.WriteLine($"GgmlManagedSmoke passed ({backendType})");
}
finally
{
    GgmlBasicOps.AlignedFree(weightData);
}
