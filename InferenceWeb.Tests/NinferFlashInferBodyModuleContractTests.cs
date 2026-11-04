// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Cuda;
using Xunit;

public sealed class NinferFlashInferBodyModuleContractTests
{
    private static readonly (int N, int K)[] Shapes = { (34816, 5120), (5120, 17408), (5120, 6144), (14336, 5120) };
    private static JsonObject Metadata(string family)
    {
        string policy = family == "n128" ? CudaNinferFlashInferBodyModule.N128Policy : CudaNinferFlashInferBodyModule.N64Policy;
        var values = new List<object>(); var images = new List<CudaNinferFlashInferBodyModule.ImageSpec>();
        foreach (var (n, k) in Shapes) for (int t = 1; t <= 8; t++)
        {
            var layout = CudaNinferFlashInferBodyModule.Plan(n, k, t); string file = "test" + layout.Index + ".cubin", image = layout.Index.ToString("x64"), kernel = "kernel" + layout.Index;
            images.Add(new(layout, file, image, kernel));
            values.Add(new { outputRows = n, columns = k, rows = t, cubin = file, cubinSha256 = image, kernel,
                codeBytes = layout.CodeBytes, scaleBytes = layout.ScaleBytes, inputBytes = layout.InputBytes, outputBytes = layout.OutputBytes,
                alphaBytes = 4, dynamicSmemBytes = CudaNinferFlashInferBodyModule.SharedBytes(family, t), grid = new[] { n / (family == "n128" ? 128 : 64), 1, 1 }, block = new[] { 128, 1, 1 },
                argumentOrder = new[] { "inputBf16", "originalCodes", "originalScales", "alphaF32", "outputF32" },
                parameters = Enumerable.Range(0, 5).Select(i => new { index = i, offset = i * 8, size = 8 }).ToArray() });
        }
        return JsonSerializer.SerializeToNode(new { schema = "tensorsharp-ninfer-flashinfer-body-aot/1", family, arithmeticPolicy = policy, status = "complete", arch = "sm_120a",
            outputDtype = "Float32", splits = 1, warps = 4, tileK = 128, stages = 2, enablePdl = false,
            sourceVerification = new { revision = CudaNinferFlashInferBodyModule.Revision, manifestSha256 = CudaNinferFlashInferBodyModule.OriginalSourceManifestSha256,
                originalFilesVerified = 4, licenseFilesVerified = 1, modifiedUpstreamFiles = Array.Empty<string>() },
            specializations = values, imageSetSha256 = CudaNinferFlashInferBodyModule.ComputeImageSetSha256(policy, images.ToArray()) })!.AsObject();
    }
    private static CudaNinferFlashInferBodyModule.ManifestContract Parse(JsonObject metadata)
        => CudaNinferFlashInferBodyModule.ValidateManifest(JsonSerializer.SerializeToUtf8Bytes(metadata));

    [Theory]
    [InlineData("n128", 128)]
    [InlineData("n64", 64)]
    public void CompleteStaticFamilyContainsExactlyThe32BodyCases(string family, int tile)
    {
        var contract = Parse(Metadata(family));
        Assert.Equal(family, contract.Family); Assert.Equal(tile, contract.TileN); Assert.Equal(32, contract.Images.Length);
        Assert.Equal(Enumerable.Range(0, 32), contract.Images.Select(x => x.Layout.Index));
    }
    [Fact]
    public void GeometrySpansAreExactForEveryShapeAndPhysicalBatch()
    {
        foreach (var (n, k) in Shapes) for (int t = 1; t <= 8; t++)
        {
            Assert.True(CudaNinferFlashInferBodyModule.Supports(n, k, t)); var plan = CudaNinferFlashInferBodyModule.Plan(n, k, t);
            Assert.Equal((long)n * k / 2, plan.CodeBytes); Assert.Equal((long)n * k / 16, plan.ScaleBytes);
            Assert.Equal((long)t * k * 2, plan.InputBytes); Assert.Equal((long)t * n * 4, plan.OutputBytes);
            foreach (string family in new[] { "n128", "n64" })
            {
                int live = family == "n128" ? 16 : Math.Min(t + 1, 16), tileN = family == "n128" ? 128 : 64;
                Assert.Equal(2 * (live * 68 * 4 + tileN * 20 * 4 + 1024), CudaNinferFlashInferBodyModule.SharedBytes(family, t));
            }
        }
        Assert.Equal(100270080L, Shapes.Max(s => (long)s.N * s.K * 9 / 16));
    }
    [Theory]
    [InlineData(34816, 5120, 0)]
    [InlineData(34816, 5120, 9)]
    [InlineData(16384, 5120, 1)]
    [InlineData(96, 5120, 8)]
    [InlineData(5120, 5120, 1)]
    public void GdnInputBf16AndUnsupportedPhysicalShapesAreNotAdmitted(int n, int k, int t)
    {
        Assert.False(CudaNinferFlashInferBodyModule.Supports(n, k, t));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferFlashInferBodyModule.Plan(n, k, t));
    }
    [Fact]
    public void MissingDuplicateAndAliasedSpecializationsAreRejected()
    {
        var missing = Metadata("n128"); missing["specializations"]!.AsArray().RemoveAt(31); Assert.Throws<ArgumentException>(() => Parse(missing));
        var duplicate = Metadata("n128"); duplicate["specializations"]![31] = duplicate["specializations"]![0]!.DeepClone(); Assert.Throws<ArgumentException>(() => Parse(duplicate));
        var aliased = Metadata("n128"); aliased["specializations"]![31]!["cubin"] = aliased["specializations"]![0]!["cubin"]!.GetValue<string>(); Assert.Throws<ArgumentException>(() => Parse(aliased));
    }
    [Theory]
    [InlineData("status", "building")]
    [InlineData("arch", "sm_121a")]
    [InlineData("outputDtype", "BFloat16")]
    [InlineData("family", "unknown")]
    [InlineData("arithmeticPolicy", CudaNinferFlashInferBodyModule.N64Policy)]
    public void IncompleteOrMislabeledArithmeticMetadataFails(string field, string value)
    {
        var metadata = Metadata("n128"); metadata[field] = value;
        Assert.Throws<ArgumentException>(() => Parse(metadata));
    }
    [Theory]
    [InlineData("splits", 2)]
    [InlineData("warps", 8)]
    [InlineData("stages", 3)]
    [InlineData("tileK", 64)]
    public void UnqualifiedTacticsRequireTheirOwnExplicitImplementation(string field, int value)
    {
        var metadata = Metadata("n128"); metadata[field] = value;
        Assert.Throws<ArgumentException>(() => Parse(metadata));
    }

    [Fact]
    public void KernelAbiLaunchPayloadAndProvenanceChangesFailBeforeCudaAccess()
    {
        foreach (string field in new[] { "codeBytes", "scaleBytes", "inputBytes", "outputBytes", "dynamicSmemBytes" })
        {
            var metadata = Metadata("n64"); metadata["specializations"]![0]![field] = 1;
            Assert.Throws<ArgumentException>(() => Parse(metadata));
        }
        var abi = Metadata("n128"); abi["specializations"]![0]!["parameters"]![4]!["offset"] = 24; Assert.Throws<ArgumentException>(() => Parse(abi));
        var path = Metadata("n128"); path["specializations"]![0]!["cubin"] = "../other.cubin"; Assert.Throws<ArgumentException>(() => Parse(path));
        var source = Metadata("n128"); source["sourceVerification"]!["modifiedUpstreamFiles"] = new JsonArray("kernel.py"); Assert.Throws<ArgumentException>(() => Parse(source));
    }
    [Fact]
    public void EveryImageAndThePolicyContributeToImmutableIdentity()
    {
        var metadata = Metadata("n128"); var contract = Parse(metadata);
        string before = metadata["imageSetSha256"]!.GetValue<string>();
        var copy = contract.Images.ToArray(); copy[31] = copy[31] with { CubinSha256 = new string('f', 64) };
        Assert.NotEqual(before, CudaNinferFlashInferBodyModule.ComputeImageSetSha256(contract.Policy, copy));
        Assert.NotEqual(before, CudaNinferFlashInferBodyModule.ComputeImageSetSha256(CudaNinferFlashInferBodyModule.N64Policy, contract.Images));
        metadata["specializations"]![31]!["cubinSha256"] = new string('f', 64); Assert.Throws<ArgumentException>(() => Parse(metadata));
    }
}
