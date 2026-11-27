// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Ltx25;

namespace InferenceWeb.Tests;

/// <summary>
/// LTX 2.5's denoiser on the diffusion runtime against the retired direct denoiser's steps, frozen before it was deleted:
/// Fixtures/Ltx25/LegacyGoldens/&lt;lane&gt;.json holds every output bit of the four oracle cases on the synthetic checkpoint,
/// each also with a synthetic rank-2 Ingredients adapter over all ten adapted projections of both blocks (written by
/// eng/validation/Ltx25DenoiserAB's fixture mode, whose adapter <see cref="FixtureAdapter"/> writes byte for byte). On the lane
/// a file was frozen on, over the same checkpoint and adapter bytes, the runtime gives the legacy bits; against every other
/// frozen lane (another backend or device, whose kernels round otherwise: the retired CPU and CUDA lanes differed by up to
/// 1.8e-7) its steps stay within 2e-5, the bound the unchanged official oracle holds both implementations to.
/// </summary>
public sealed partial class Ltx25DenoiserTests
{
    private const string GoldenPrefix = "InferenceWeb.Tests.Fixtures.Ltx25.LegacyGoldens.";

    private sealed record FrozenStep(string Lane, string CheckpointSha256, string AdapterSha256, bool RuntimeBitExact,
        float[] Video, float[] Audio, int Blocks);

    private static readonly Lazy<List<(string Case, FrozenStep Step)>> s_frozen = new(() =>
    {
        var steps = new List<(string, FrozenStep)>();
        var asm = typeof(Ltx25DenoiserTests).Assembly;
        foreach (string name in asm.GetManifestResourceNames().Where(n => n.StartsWith(GoldenPrefix, StringComparison.Ordinal)))
        {
            using Stream stream = asm.GetManifestResourceStream(name)!;
            using JsonDocument doc = JsonDocument.Parse(stream);
            JsonElement root = doc.RootElement;
            static float[] Bits(JsonElement output)
            {
                string hex = output.GetProperty("bits").GetString()!;
                var values = new float[hex.Length / 8];
                for (int i = 0; i < values.Length; i++)
                    values[i] = BitConverter.Int32BitsToSingle(Convert.ToInt32(hex.Substring(i * 8, 8), 16));
                Assert.Equal(output.GetProperty("count").GetInt32(), values.Length);
                Assert.Equal(output.GetProperty("sha256").GetString(),
                    Convert.ToHexStringLower(SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()))));
                return values;
            }
            foreach (JsonProperty c in root.GetProperty("cases").EnumerateObject())
                steps.Add((c.Name, new FrozenStep(root.GetProperty("lane").GetString()!, root.GetProperty("checkpointSha256").GetString()!,
                    root.GetProperty("adapterSha256").GetString()!, root.GetProperty("runtimeBitExact").GetBoolean(),
                    Bits(c.Value.GetProperty("video")), Bits(c.Value.GetProperty("audio")), c.Value.GetProperty("blocks").GetInt32())));
        }
        return steps;
    });

    public static TheoryData<int, bool> GoldenCases()
    {
        var data = new TheoryData<int, bool>();
        foreach (object[] c in Cases) { data.Add((int)c[0], false); data.Add((int)c[0], true); }
        return data;
    }

    /// <summary>The lane a backend runs on here, as the A/B tool names it: backend, runtime identifier, device.</summary>
    private static string LaneOf(IAllocator allocator) => allocator switch
    {
        CudaAllocator cuda => "cuda|" + System.Runtime.InteropServices.RuntimeInformation.RuntimeIdentifier + "|" + CudaDevice.GetDevice(cuda.DeviceId).Name,
        MlxAllocator => "mlx|" + System.Runtime.InteropServices.RuntimeInformation.RuntimeIdentifier + "|" + HostDevice(),
        _ => "cpu|" + System.Runtime.InteropServices.RuntimeInformation.RuntimeIdentifier + "|" + HostDevice(),
    };

    private static string HostDevice() => MiniMaxH3ImageGoldens.LaneKey(TensorSharp.Runtime.BackendType.GgmlCpu).Split('|')[2];

    [Theory]
    [MemberData(nameof(GoldenCases))]
    public void StepsGiveTheRetiredDirectDenoisersBitsOnItsLane(int index, bool adapted)
    {
        using var fixture = new Checkpoint();
        using var backend = new Backend();
        using var adapter = new FixtureAdapter(fixture.Path);
        string name = $"case{index}{(adapted ? "-ingredients" : "")}", lane = LaneOf(backend.Allocator);
        string checkpointSha = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(fixture.Path)));
        var frozen = s_frozen.Value.Where(f => f.Case == name).Select(f => f.Step).ToList();
        Assert.NotEmpty(frozen);
        using var model = new Ltx25Denoiser(fixture.Path, backend.Allocator, adapter.Path);
        var output = model.Forward(Input(Oracle.GetProperty("cases")[index], adapted));
        bool checkedBits = false;
        foreach (FrozenStep f in frozen)
        {
            Assert.Equal(checkpointSha, f.CheckpointSha256);
            Assert.Equal(f.Blocks, output.ExecutedBlocks);
            if (f.Lane == lane && f.AdapterSha256 == adapter.Sha256 && f.RuntimeBitExact)
            {
                Assert.Equal(f.Video.Select(BitConverter.SingleToInt32Bits), output.VideoVelocity.Select(BitConverter.SingleToInt32Bits));
                Assert.Equal(f.Audio.Select(BitConverter.SingleToInt32Bits), output.AudioVelocity.Select(BitConverter.SingleToInt32Bits));
                checkedBits = true;
            }
            else
            {
                Equal(f.Video, output.VideoVelocity, 2e-5, $"{name} video against the legacy {f.Lane}");
                Equal(f.Audio, output.AudioVelocity, 2e-5, $"{name} audio against the legacy {f.Lane}");
            }
        }
        // On a lane a file was frozen on, the bits are the check: an adapter whose bytes differ from the tool's is a broken fixture.
        if (frozen.Any(f => f.Lane == lane && f.RuntimeBitExact))
            Assert.True(checkedBits, $"{name}: frozen on {lane}, but the fixture adapter's bytes ({adapter.Sha256}) differ from the frozen ones");
    }

    /// <summary>The A/B tool's synthetic Ingredients adapter over the fixture checkpoint: rank 2, F32, every adapted projection
    /// of every block, each pair seeded in order (A = sin(.37 i + seed) / 4, B = cos(.23 i + seed) / 8).</summary>
    private sealed class FixtureAdapter : IDisposable
    {
        private static readonly string[] Targets =
            ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0", "ff.net.0.proj", "ff.net.2"];

        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "tensorsharp-ltx25-golden-" + Guid.NewGuid().ToString("N") + ".safetensors");
        public string Sha256 { get; }

        public FixtureAdapter(string checkpoint)
        {
            using var file = new TensorSharp.Runtime.GgufFile(checkpoint);
            var config = Ltx25Config.FromGguf(file);
            var tensors = new List<(string Name, int[] Shape, float[] Values)>();
            int seed = 1;
            for (int b = 0; b < config.Layers; b++)
                foreach (string t in Targets)
                {
                    var shape = file.Tensors[config.Prefix + $"transformer_blocks.{b}.{t}.weight"].Shape;
                    int inner = (int)shape[0], outer = (int)shape[1], rank = 2;
                    tensors.Add(($"diffusion_model.transformer_blocks.{b}.{t}.lora_A.weight", [rank, inner],
                        Enumerable.Range(0, rank * inner).Select(i => MathF.Sin(i * .37f + seed) * .25f).ToArray()));
                    tensors.Add(($"diffusion_model.transformer_blocks.{b}.{t}.lora_B.weight", [outer, rank],
                        Enumerable.Range(0, outer * rank).Select(i => MathF.Cos(i * .23f + seed) * .125f).ToArray()));
                    seed++;
                }
            var header = new Dictionary<string, object> { ["__metadata__"] = new Dictionary<string, string> { ["model_version"] = "2.5", ["reference_downscale_factor"] = "1" } };
            long offset = 0;
            foreach (var t in tensors)
            {
                long bytes = t.Values.LongLength * 4;
                header[t.Name] = new { dtype = "F32", shape = t.Shape, data_offsets = new[] { offset, offset + bytes } };
                offset += bytes;
            }
            byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
            int padded = (json.Length + 7) / 8 * 8;
            using (var output = new BinaryWriter(File.Create(Path)))
            {
                output.Write((ulong)padded); output.Write(json);
                for (int i = json.Length; i < padded; i++) output.Write((byte)' ');
                foreach (var t in tensors) foreach (float v in t.Values) output.Write(v);
            }
            Sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(Path)));
        }

        public void Dispose() => File.Delete(Path);
    }
}
