using System;
using System.IO;
using System.Linq;
using TensorSharp.Cuda;
using TensorSharp.Models.Direct;
using TensorSharp.MLX;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>Bounded dense safetensors projection; no whole-encoder F32 expansion.</summary>
internal sealed class Ltx25MappedLinear : IDisposable
{
    private readonly SafetensorsFile _file;
    private readonly DirectContext _context;
    private readonly SafetensorTensorInfo _info;
    private readonly Tensor _bias;
    private readonly IntPtr _pointer;
    private readonly int _type;
    private readonly bool _cudaBf16Supported;
    private readonly Ltx25CudaFilePrefault _prefault;
    internal int Input { get; }
    internal int Output { get; }
    internal bool UsedCudaMapping { get; private set; }
    internal bool UsedCudaBf16 { get; private set; }

    internal Ltx25MappedLinear(SafetensorsFile file, DirectContext context, string name, bool cudaBf16Supported,
        Ltx25CudaFilePrefault prefault)
    {
        _file = file; _context = context; _cudaBf16Supported = cudaBf16Supported; _prefault = prefault; _info = file.GetInfo(name + ".weight");
        if (_info.Shape.Length != 2 || _info.Shape.Any(s => s is < 1 or > int.MaxValue))
            throw new InvalidDataException("Invalid LTX encoder matrix: " + name);
        Output = (int)_info.Shape[0]; Input = (int)_info.Shape[1];
        _type = _info.Dtype switch { SafetensorDtype.BF16 => 30, SafetensorDtype.F16 => 1, SafetensorDtype.F32 => 0,
            _ => throw new InvalidDataException("Unsupported LTX encoder weight type: " + _info.Dtype) };
        if (!file.TryGetTensorDataPointer(_info, out _pointer)) throw new IOException("Could not map LTX encoder weights.");
        if (file.HasTensor(name + ".bias")) _bias = context.FromFloats(file.ReadFloat32(name + ".bias"), 1, Output);
    }

    internal Tensor Forward(Tensor input, bool bf16, System.Threading.CancellationToken cancellationToken = default)
    {
        var result = _context.NewF32(input.Sizes[0], Output);
        try
        {
            if (_context.IsCuda)
            {
                cancellationToken.ThrowIfCancellationRequested();
                _prefault?.Prepare(_pointer, checked(_file.DataOffset + _info.Begin), _info.ByteCount);
                bool nativeBf16 = _cudaBf16Supported && bf16 && _type == 30;
                bool completed = nativeBf16
                    ? CudaQuantizedOps.TryAddmmQuantizedToFloat32(result, input, _pointer, _pointer, _type, Input, Output, _info.ByteCount)
                    : CudaMiniMaxH3Ops.TryMappedDense(result, input, _pointer, _type, Input, Output, _info.ByteCount);
                if (!completed)
                    throw new NotSupportedException("CUDA mapped LTX dense projection is unavailable.");
                UsedCudaMapping = true;
                UsedCudaBf16 = nativeBf16;
            }
            else if (_context.IsMlx)
            {
                // MLX BF16 addmm can round the product before adding its bias: every projection accumulates in F32.
                if (!MlxLtx25PromptOps.TryMappedDense(result, input, _pointer, _type, Input, Output, false))
                    throw new NotSupportedException("MLX mapped LTX dense projection is unavailable.");
            }
            else
            {
                int band = Math.Max(1, Math.Min(Output, (64 << 20) / (sizeof(float) * Input)));
                for (int start = 0; start < Output; start += band)
                {
                    int count = Math.Min(band, Output - start);
                    long begin = _info.Begin + (long)start * Input * (_type == 0 ? 4 : 2);
                    var slice = new SafetensorTensorInfo { Name = _info.Name, Shape = [count, Input], Dtype = _info.Dtype,
                        Begin = begin, End = begin + (long)count * Input * (_type == 0 ? 4 : 2) };
                    var values = new float[count * Input];
                    _file.ReadTensorDataToFloat32(slice, values, values.Length);
                    using var weight = _context.FromFloats(values, count, Input);
                    using var transposed = weight.Transpose();
                    using var output = _context.NewF32(input.Sizes[0], count);
                    Ops.Addmm(output, 0, output, 1, input, transposed);
                    using var destination = result.Narrow(1, start, count);
                    Ops.Copy(destination, output);
                }
            }
            if (_bias != null) Ops.Add(result, result, _bias);
            if (bf16) Ltx25PromptMath.RoundBf16(result);
            return result;
        }
        catch { result.Dispose(); throw; }
    }

    public void Dispose()
    {
        if (_context.IsCuda) CudaQuantizedOps.ReleaseQuantizedWeight(_context.CudaAllocator, _pointer);
        _bias?.Dispose();
    }
}
