using System.Reflection;
using System.Runtime.ExceptionServices;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public partial class Qwen4ExpMtpValidationTests
{
    [Fact]
    public void Published32TensorShapesAndSharedTargetMetadata_AreAcceptedWithoutPayloadReads()
    {
        using var f = new HeadMetadataFixture();
        Assert.Equal(48, Validate(f.Target, f.Draft));
        Assert.Equal(32, f.Draft.Tensors.Count);
        Assert.Equal(32, new FileInfo(f.Draft.FilePaths[0]).Length);
        Assert.Equal(32, new FileInfo(f.Target.FilePaths[0]).Length);
    }

    public static IEnumerable<object[]> TensorNames() => PublishedTensors.Select(t => new object[] { t.Name });

    [Theory]
    [MemberData(nameof(TensorNames))]
    public void EveryPublishedTensor_IsRequiredIncludingInactiveIndexer(string name)
    {
        using var f = new HeadMetadataFixture();
        f.Draft.Tensors.Remove(name);
        Assert.Throws<InvalidOperationException>(() => Validate(f.Target, f.Draft));
    }

    [Theory]
    [MemberData(nameof(TensorNames))]
    public void EveryPublishedTensor_ShapeIsValidated(string name)
    {
        using var f = new HeadMetadataFixture();
        f.Draft.Tensors[name].Shape[0]++;
        Assert.Throws<InvalidOperationException>(() => Validate(f.Target, f.Draft));
    }

    [Theory]
    [InlineData("token_embd.weight", false)]
    [InlineData("token_embd.weight", true)]
    [InlineData("output.weight", false)]
    [InlineData("output.weight", true)]
    public void BorrowedTargetTensor_MustExistWithExactShape(string name, bool wrongShape)
    {
        using var f = new HeadMetadataFixture();
        if (wrongShape) f.Target.Tensors[name].Shape[1]++;
        else f.Target.Tensors.Remove(name);
        Assert.Throws<InvalidOperationException>(() => Validate(f.Target, f.Draft));
    }

    [Theory]
    [InlineData("tokenizer.ggml.tokens")]
    [InlineData("tokenizer.ggml.merges")]
    public void SameCountPermutedTokenizerMapping_IsRejected(string key)
    {
        using var f = new HeadMetadataFixture();
        var items = (string[])f.Draft.Metadata[key];
        (items[0], items[1]) = (items[1], items[0]);
        Assert.Throws<InvalidOperationException>(() => Validate(f.Target, f.Draft));
        Assert.NotEqual(items, f.Target.GetStringArray(key));
    }

    [Theory]
    [InlineData("tokenizer.ggml.bos_token_id")]
    [InlineData("tokenizer.ggml.eos_token_id")]
    [InlineData("tokenizer.ggml.padding_token_id")]
    public void SpecialTokenMappingMismatch_IsRejected(string key)
    {
        using var f = new HeadMetadataFixture();
        f.Draft.Metadata[key] = 3u;
        Assert.Throws<InvalidOperationException>(() => Validate(f.Target, f.Draft));
    }

    [Fact]
    public void DifferentTokenTypeAtSameId_IsRejected()
    {
        using var f = new HeadMetadataFixture();
        ((int[])f.Draft.Metadata["tokenizer.ggml.token_type"])[2] = 3;
        Assert.Throws<InvalidOperationException>(() => Validate(f.Target, f.Draft));
    }

    [Fact]
    public void ActiveMtpIndexer_IsRejectedEvenWhenIndexerWeightsExist()
    {
        using var f = new HeadMetadataFixture();
        ((int[])f.Draft.Metadata["qwen4exp.attention.compress_ratios"])[48] = 4;
        Assert.Throws<NotSupportedException>(() => Validate(f.Target, f.Draft));
    }

    [Theory]
    [InlineData("token_embd.weight")]
    [InlineData("output.weight")]
    [InlineData("blk.48.nextn.eh_proj.weight")]
    public void IntegerMatrixStorage_IsRejectedBeforeBorrowOrUpload(string name)
    {
        using var f = new HeadMetadataFixture();
        var file = name.StartsWith("blk.", StringComparison.Ordinal) ? f.Draft : f.Target;
        file.Tensors[name].Type = GgmlTensorType.I8;
        AssertRejected(f);
    }

    [Theory]
    [InlineData("attention.layer_norm_rms_epsilon", 0f)]
    [InlineData("attention.layer_norm_rms_epsilon", float.PositiveInfinity)]
    [InlineData("rope.freq_base", 0f)]
    [InlineData("rope.freq_base", float.PositiveInfinity)]
    public void MatchingButInvalidArithmeticMetadata_IsRejected(string suffix, float value)
    {
        using var f = new HeadMetadataFixture();
        f.Target.Metadata["qwen4exp." + suffix] = f.Draft.Metadata["qwen4exp." + suffix] = value;
        AssertRejected(f);
    }

    [Fact]
    public void UnequalKeyAndValueHeadWidths_AreRejectedBeforeBuildingTheGraph()
    {
        using var f = new HeadMetadataFixture();
        f.Target.Metadata["qwen4exp.attention.value_length"] = f.Draft.Metadata["qwen4exp.attention.value_length"] = 128u;
        AssertRejected(f);
    }

    [Fact]
    public void MatchingOddRotaryWidth_IsRejectedBeforeNativeRopeAssertion()
    {
        using var f = new HeadMetadataFixture();
        f.Target.Metadata["qwen4exp.rope.dimension_count"] = f.Draft.Metadata["qwen4exp.rope.dimension_count"] = 63u;
        AssertRejected(f);
    }

    [Fact]
    public void MatchingEmptyRotarySections_AreRejectedBeforeMediaRopeAssertion()
    {
        using var f = new HeadMetadataFixture();
        f.Target.Metadata["qwen4exp.rope.dimension_sections"] = new int[4];
        f.Draft.Metadata["qwen4exp.rope.dimension_sections"] = new int[4];
        AssertRejected(f);
    }

    private static void AssertRejected(HeadMetadataFixture f)
    {
        Exception error = Assert.ThrowsAny<Exception>(() => Validate(f.Target, f.Draft));
        Assert.True(error is InvalidOperationException or NotSupportedException, error.ToString());
    }

    private static int Validate(GgufFile target, GgufFile draft)
    {
        try
        {
            return (int)typeof(Qwen4ExpModel).GetMethod("ValidateMtpHead", BindingFlags.Static | BindingFlags.NonPublic)!
                .Invoke(null, new object[] { target, draft })!;
        }
        catch (TargetInvocationException e) when (e.InnerException != null)
        { ExceptionDispatchInfo.Capture(e.InnerException).Throw(); throw; }
    }

    private sealed class HeadMetadataFixture : IDisposable
    {
        internal readonly GgufFile Target = EmptyGguf();
        internal readonly GgufFile Draft = EmptyGguf();
        internal HeadMetadataFixture()
        {
            foreach (GgufFile file in new[] { Target, Draft })
            {
                foreach (var pair in PublishedScalarMetadata) file.Metadata[pair.Key] = pair.Value;
                file.Metadata["qwen4exp.block_count"] = ReferenceEquals(file, Target) ? 48u : 49u;
                file.Metadata["qwen4exp.nextn_predict_layers"] = ReferenceEquals(file, Target) ? 0u : 1u;
                file.Metadata["qwen4exp.nextn_shared_target_tensors"] = true;
                file.Metadata["qwen4exp.rope.dimension_sections"] = new[] { 16, 24, 24, 0 };
                file.Metadata["qwen4exp.attention.compress_ratios"] = Enumerable.Range(0, ReferenceEquals(file, Target) ? 48 : 49)
                    .Select(i => (i + 1) % 4 == 0 ? 4 : 0).ToArray();
                // Tiny synthetic vocabulary; only identity/shape validation is
                // tested. Actual publisher matrix shapes stay unchanged below.
                file.Metadata["tokenizer.ggml.tokens"] = new[] { "<pad>", "<eos>", "alpha", "beta" };
                file.Metadata["tokenizer.ggml.merges"] = new[] { "a l", "al pha", "b eta" };
                file.Metadata["tokenizer.ggml.token_type"] = new[] { 3, 3, 1, 1 };
                file.Metadata["tokenizer.ggml.bos_token_id"] = 0u;
                file.Metadata["tokenizer.ggml.eos_token_id"] = 1u;
                file.Metadata["tokenizer.ggml.padding_token_id"] = 0u;
                file.Metadata["tokenizer.ggml.add_bos_token"] = false;
            }
            foreach (var t in PublishedTensors)
                Draft.Tensors.Add(t.Name, new GgufTensorInfo { Name = t.Name, Shape = (ulong[])t.Shape.Clone(), Type = t.Type });
            foreach (string name in new[] { "token_embd.weight", "output.weight" })
                Target.Tensors.Add(name, new GgufTensorInfo { Name = name, Shape = new ulong[] { 2560, 4 }, Type = GgmlTensorType.Q8_0 });
        }
        private static GgufFile EmptyGguf()
        {
            string path = Path.Combine(Path.GetTempPath(), $"ts-q4e-head-metadata-{Guid.NewGuid():N}.gguf");
            using (var writer = new BinaryWriter(new FileStream(path, FileMode.CreateNew, FileAccess.Write)))
            {
                writer.Write(0x46554747u); writer.Write(3u); writer.Write(0UL); writer.Write(0UL); writer.Write(new byte[8]);
            }
            return new GgufFile(path);
        }
        public void Dispose()
        {
            foreach (GgufFile file in new[] { Target, Draft })
            {
                string path = file.FilePaths[0]; file.Dispose(); File.Delete(path);
            }
        }
    }
}
