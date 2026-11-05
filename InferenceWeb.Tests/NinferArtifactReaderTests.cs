using System.Text;
using System.Text.Json.Nodes;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

// Generated sparse NInfer v3 files only. No model, CUDA or native loader dependency.
public sealed class NinferArtifactReaderTests
{
    [Fact]
    public void HeaderIdentityAndPayloadAlignmentFollowTheV3Entry()
    {
        var b = new NinferV3Builder();
        b.Bind("w", b.Tensor("bf16", new ulong[] { 4, 128 }));
        using var f = b.Write();
        using var reader = new NinferArtifactReader(f.Path);
        Assert.Equal(f.PayloadStart, reader.PayloadStart);
        Assert.Equal(0, reader.PayloadStart % 4096);
        Assert.Equal("08090a0b0c0d0e0f1011121314151617", reader.ArtifactId);
        Assert.Equal(new NinferArtifactIdentity("qwen3.8-27b", "qwen3_8_27b_nvfp4"), reader.Identity);
        Assert.Equal(32 + f.JsonBytes, reader.InitializationBytesRead);
    }

    [Theory]
    [InlineData("bf16", new ulong[] { 3, 128 }, 768UL)]
    [InlineData("fp32", new ulong[] { 48 }, 192UL)]
    [InlineData("int32", new ulong[] { 7 }, 28UL)]
    [InlineData("fp8_e4m3fn_row_bf16", new ulong[] { 16, 256 }, 4096UL + 32)]
    [InlineData("q8_g32_fp16", new ulong[] { 3, 128 }, 512UL + 24)]
    [InlineData("q4_g64_fp16", new ulong[] { 4, 128 }, 256UL + 16)]
    [InlineData("nvfp4", new ulong[] { 128, 64 }, 4096UL + 512 + 4)]
    public void FormatsHaveTheirEncodedGeometry(string format, ulong[] shape, ulong bytes)
    {
        var b = new NinferV3Builder();
        b.Bind("w", b.Tensor(format, shape, nvfp4Divisor: 2.5f));
        using var f = b.Write();
        using var reader = new NinferArtifactReader(f.Path);
        var t = Assert.Single(reader.Tensors);
        Assert.Equal(bytes, t.ByteLength);
        Assert.Equal(shape, t.Shape);
        if (format == "nvfp4") Assert.Equal(2.5f, t.WeightDivisor);
        else Assert.Null(t.WeightDivisor);
        if (format == "fp8_e4m3fn_row_bf16")
        {
            Assert.Equal(NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale, t.Format);
            Assert.Equal(4096UL, t.Geometry.ScalePlaneOffset);
            Assert.Equal(32UL, t.Geometry.ScalePlaneBytes);
        }
    }

    [Fact]
    public void BindingsResolveWholeObjectsAndRowParts()
    {
        var b = new NinferV3Builder();
        b.Fused("fp8_e4m3fn_row_bf16", 256, new[] { ("q", 32UL), ("k", 16UL) }, "AllowA8");
        string whole = b.Tensor("bf16", new ulong[] { 8 });
        b.Bind("norm", whole);
        using var f = b.Write();
        using var reader = new NinferArtifactReader(f.Path);
        var k = reader.GetBinding("k");
        Assert.False(k.IsWholeObject);
        Assert.Equal((32, 16), k.RowRange);
        Assert.Same(reader.GetBinding("q").Object, k.Object);
        Assert.True(reader.GetBinding("norm").IsWholeObject);
        Assert.Equal(NinferActivationPolicy.AllowA8, Assert.Single(reader.UsesOf("q")).Policy);
        Assert.Throws<InvalidDataException>(() => reader.GetBinding("missing"));
    }

    [Fact]
    public void A4UsesCarryTheirActivationDivisorScalar()
    {
        var b = new NinferV3Builder();
        b.Fused("nvfp4", 64, new[] { ("gate", 128UL), ("up", 128UL) }, "AllowA4", divisors: new[] { 836f, 836f });
        using var f = b.Write();
        using var reader = new NinferArtifactReader(f.Path);
        var use = Assert.Single(reader.UsesOf("up"));
        Assert.Equal(NinferActivationPolicy.AllowA4, use.Policy);
        Assert.Equal(836f, reader.ReadScalar(use.ActivationInputDivisor!));
    }

    [Fact]
    public void ComponentResourcesAreMappedByFileName()
    {
        var b = new NinferV3Builder();
        b.Resource("text", "tokenizer.json", Encoding.UTF8.GetBytes("{}"));
        b.Bind("w", b.Tensor("bf16", new ulong[] { 2 }));
        using var f = b.Write();
        using var reader = new NinferArtifactReader(f.Path);
        var resource = reader.ComponentResources["text"]["tokenizer.json"];
        Assert.Equal("{}", Encoding.UTF8.GetString(reader.ReadResource(resource)));
        Assert.True(reader.ComponentConfigs.ContainsKey("text"));
    }

    [Fact]
    public void V2ArtifactsAreRefusedWithAnUpgradeMessage()
    {
        var b = new NinferV3Builder();
        b.Bind("w", b.Tensor("bf16", new ulong[] { 2 }));
        using var f = b.Write(magic: "NINFER\0\x02");
        var error = Assert.Throws<NotSupportedException>(() => new NinferArtifactReader(f.Path));
        Assert.Contains("v3", error.Message);
        Assert.True(NinferArtifactReader.IsNinferArtifact(f.Path));
    }

    [Fact]
    public void UnknownFormatsOverlapsAndMultiPartBindingsAreRefused()
    {
        var b = new NinferV3Builder();
        b.Bind("w", b.Tensor("bf16", new ulong[] { 2 }));
        using (var f = b.Write(d => { d["objects"]![0]!["format"] = "fp4_e9"; return d; }))
            Assert.Throws<NotSupportedException>(() => new NinferArtifactReader(f.Path));
        var c = new NinferV3Builder();
        c.Bind("a", c.Tensor("bf16", new ulong[] { 256 }));
        c.Bind("b", c.Tensor("bf16", new ulong[] { 256 }));
        using (var f = c.Write(d => { d["objects"]![1]!["offset"] = 0; return d; }))
            Assert.Throws<InvalidDataException>(() => new NinferArtifactReader(f.Path));
        using (var f = c.Write(d =>
        {
            var part = new JsonObject { ["object"] = "weight/000000", ["range"] = new JsonArray(0, 128) };
            d["bindings"]!["a"] = new JsonObject { ["parts"] = new JsonArray(part, part.DeepClone()) };
            return d;
        }))
            Assert.Throws<InvalidDataException>(() => new NinferArtifactReader(f.Path));
    }

    [Fact]
    public void GgufMagicIsNotAnArtifact()
    {
        string path = Path.Combine(Path.GetTempPath(), "ts-gguf-" + Guid.NewGuid().ToString("N") + ".ninfer");
        File.WriteAllBytes(path, "GGUF\x03\0\0\0"u8.ToArray());
        try { Assert.False(NinferArtifactReader.IsNinferArtifact(path)); }
        finally { File.Delete(path); }
    }
}
