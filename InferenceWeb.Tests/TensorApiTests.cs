using AdvUtils;
using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceWeb.Tests;

public class TensorApiTests
{
    private readonly IAllocator _allocator = new CpuAllocator(BlasEnum.DotNet);

    [Fact]
    public void ConstructorCopiesShapeMetadata()
    {
        long[] sizes = { 2, 3 };

        using Tensor tensor = new Tensor(_allocator, DType.Float32, sizes);
        sizes[0] = 99;

        Assert.Equal(2, tensor.Sizes[0]);
        Assert.Equal(3, tensor.Sizes[1]);
    }

    [Fact]
    public void ShapeMetadataCanOnlyBeMutatedThroughACopy()
    {
        using Tensor tensor = new Tensor(_allocator, DType.Float32, 2, 3);

        long[] sizes = tensor.Sizes.ToArray();
        sizes[0] = 99;

        Assert.Equal(2, tensor.Sizes[0]);
    }

    [Fact]
    public void ViewCopiesRequestedShapeMetadata()
    {
        using Tensor tensor = new Tensor(_allocator, DType.Float32, 2, 3);
        long[] viewShape = { 3, 2 };

        using Tensor view = tensor.View(viewShape);
        viewShape[0] = 99;

        Assert.Equal(3, view.Sizes[0]);
        Assert.Equal(2, view.Sizes[1]);
    }

    [Fact]
    public void DisposeIsIdempotentForTensorAndAlias()
    {
        Tensor tensor = new Tensor(_allocator, DType.Float32, 2, 2);
        Tensor alias = tensor.CopyRef();

        tensor.Dispose();
        tensor.Dispose();

        alias.SetElementAsFloat(42, 0, 0);
        Assert.Equal(42, alias.GetElementAsFloat(0, 0));

        alias.Dispose();
        alias.Dispose();
    }
}
