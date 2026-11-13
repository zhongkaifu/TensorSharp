// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

// TestAssemblyConfig disables parallel execution: the native registry and
// backend are process global. These tests require the built native bridge for
// TestGates.PinnedGgmlBackend (CPU by default);
// a missing bridge is a failed prerequisite, never an early-return pass.
public sealed class Bonsai2ConstructionFailureTests
{
    [Fact]
    public void FailedConstructorReleasesTranscodedWeightsAndPartiallyRegisteredKeys()
    {
        using var file = new MalformedBonsai();
        CapturingModel.Last = null;
        var error = Assert.Throws<InvalidDataException>(() => new CapturingModel(file.Path));
        Assert.Contains("grouped GDN dimensions", error.Message);
        Snapshot snapshot = Assert.IsType<Snapshot>(CapturingModel.Last);
        Assert.Equal(2, snapshot.Weights.Length);
        Assert.Equal(0, snapshot.Model.VirtualDisposeCalls);
        Assert.Equal(0, snapshot.Model.LoadedWeightCount);
        AssertReleasedAndReusable(snapshot.Weights, snapshot.Keys);
        // Repeat in the same process to expose stale registrations and pointers
        // being reused by the allocator after a failed load.
        Assert.Throws<InvalidDataException>(() => new CapturingModel(file.Path));
        snapshot = Assert.IsType<Snapshot>(CapturingModel.Last);
        AssertReleasedAndReusable(snapshot.Weights, snapshot.Keys);
    }

    [Fact]
    public void ThrowingNativeCleanupStillReleasesOwnedWeightAndPreservesLoadException()
    {
        using var file = new MalformedBonsai();
        Assert.Throws<InvalidDataException>(() => new CapturingModel(file.Path));
        var model = CapturingModel.Last!.Model;
        var original = new InvalidDataException("original load failure");
        QuantizedWeight? weight = null;
        IntPtr key = IntPtr.Zero;
        int cleanupAttempts = 0;
        var actual = Assert.Throws<InvalidDataException>(() =>
            model.FailWithUnavailableNativeCleanup(original, out weight, out key, out cleanupAttempts));
        Assert.Same(original, actual);
        Assert.Equal(2, cleanupAttempts);
        AssertReleasedAndReusable(new[] { weight! }, new[] { key });
        Assert.Equal(0, model.LoadedWeightCount);
    }

    private static void AssertReleasedAndReusable(QuantizedWeight[] weights, IntPtr[] keys)
    {
        foreach (var weight in weights)
        {
            Assert.False(weight.HasHostData);
            Assert.Equal(IntPtr.Zero, weight.Data);
            Assert.Equal(IntPtr.Zero, weight.CacheKey);
        }
        foreach (IntPtr key in keys)
        {
            Assert.NotEqual(IntPtr.Zero, key);
            // Registration does not dereference the identity. Reusing the exact
            // former key proves the real native registry no longer owns it.
            GgmlBonsai.RegisterWeight(key, Enumerable.Repeat(1f, 128).ToArray(), 128, false);
            GgmlBonsai.UnregisterWeight(key);
        }
    }

    private sealed record Snapshot(CapturingModel Model, QuantizedWeight[] Weights, IntPtr[] Keys);

    private sealed class CapturingModel : Qwen35RuntimeModel
    {
        internal static Snapshot? Last;
        internal int VirtualDisposeCalls;
        internal int LoadedWeightCount => _quantWeights.Count;

        internal CapturingModel(string path) : base(path, TestGates.PinnedGgmlBackend) { }

        protected override bool IsQuantizedLinearWeight(GgufTensorInfo info)
        {
            // The final F32 norm is classified after both custom matrices have
            // been transcoded but before native transform registration starts.
            var weights = _quantWeights.Values.ToArray();
            Last = new Snapshot(this, weights, weights.Select(weight => weight.CacheKey).ToArray());
            return base.IsQuantizedLinearWeight(info);
        }

        public override void Dispose()
        {
            VirtualDisposeCalls++;
            throw new InvalidOperationException("A subclass is not initialized during a base constructor failure.");
        }

        internal void FailWithUnavailableNativeCleanup(Exception original, out QuantizedWeight weight,
            out IntPtr key, out int cleanupAttempts)
        {
            weight = new QuantizedWeight(new byte[36], (int)GgmlTensorType.Q2_0, 128, 1);
            weight.SetBonsaiTransform(Enumerable.Repeat(1f, 128).ToArray(), 128, false);
            key = weight.CacheKey;
            _quantWeights.Add("output.weight", weight);
            int attempts = 0;
            cleanupAttempts = 0;
            try { throw original; }
            catch
            {
                CleanUpFailedBonsaiConstruction(
                    () => { attempts++; throw new EntryPointNotFoundException("missing graph reset export"); },
                    () => { attempts++; throw new EntryPointNotFoundException("missing base reset export"); });
                cleanupAttempts = attempts;
                throw;
            }
        }
    }

    private sealed class MalformedBonsai : IDisposable
    {
        public string Path { get; } = System.IO.Path.GetTempFileName();

        public MalformedBonsai()
        {
            var metadata = new Dictionary<string, object>
            {
                ["general.architecture"] = "qwen35",
                ["qwen35.block_count"] = 1u,
                ["qwen35.context_length"] = 128u,
                ["qwen35.embedding_length"] = 128u,
                ["qwen35.feed_forward_length"] = 128u,
                ["qwen35.attention.head_count"] = 2u,
                ["qwen35.attention.head_count_kv"] = 2u,
                ["qwen35.attention.key_length"] = 64u,
                ["qwen35.attention.value_length"] = 64u,
                ["qwen35.attention.layer_norm_rms_epsilon"] = 1e-6f,
                ["qwen35.rope.freq_base"] = 10000f,
                ["qwen35.ssm.inner_size"] = 256u,
                ["qwen35.ssm.state_size"] = 64u,
                ["qwen35.ssm.group_count"] = 2u,
                ["qwen35.ssm.time_step_rank"] = 4u,
                ["qwen35.ssm.conv_kernel"] = 4u,
                ["tokenizer.ggml.model"] = "gpt2",
                ["tokenizer.ggml.pre"] = "qwen35",
                ["tokenizer.ggml.tokens"] = new[] { "a", "b" },
                ["tokenizer.ggml.merges"] = Array.Empty<string>(),
                ["tokenizer.ggml.bos_token_id"] = 0u,
                ["tokenizer.ggml.eos_token_id"] = 1u,
                ["tokenizer.ggml.add_bos_token"] = false,
                ["prism.hadamard.version"] = 1u,
                ["prism.hadamard.block_size"] = 128u,
                ["prism.hadamard.transform"] = "normalized-sylvester-walsh-hadamard",
                ["prism.hadamard.axis"] = "input-last-dimension",
                ["prism.hadamard.sign_mode"] = "explicit",
                ["prism.hadamard.sign_widths"] = new[] { 128 },
                ["prism.hadamard.sign_values"] = Enumerable.Repeat(1, 128).ToArray(),
                ["prism.hadamard.weight_names"] = new[] { "output.weight", "blk.0.ssm_out.weight" },
                ["prism.hadamard.inverse_weight_names"] = Array.Empty<string>(),
                ["prism.hadamard.gdn_v_grouped"] = true,
            };
            // ssm_out's input width is deliberately 128 instead of 256. The
            // first output.weight registration succeeds before this is refused.
            var tensors = new[]
            {
                ("output.weight", new ulong[] { 128, 2 }, 142u, new byte[68]),
                ("blk.0.ssm_out.weight", new ulong[] { 128, 2 }, 142u, new byte[68]),
                ("output_norm.weight", new ulong[] { 128 }, 0u, new byte[512]),
            };
            using var writer = new BinaryWriter(File.Create(Path));
            writer.Write(0x46554747u); writer.Write(3u);
            writer.Write((ulong)tensors.Length); writer.Write((ulong)metadata.Count);
            foreach (var (name, value) in metadata)
            {
                WriteString(writer, name);
                switch (value)
                {
                    case string text: writer.Write(8u); WriteString(writer, text); break;
                    case uint number: writer.Write(4u); writer.Write(number); break;
                    case float number: writer.Write(6u); writer.Write(number); break;
                    case bool flag: writer.Write(7u); writer.Write(flag); break;
                    case int[] values:
                        writer.Write(9u); writer.Write(5u); writer.Write((ulong)values.Length);
                        foreach (int number in values) writer.Write(number);
                        break;
                    case string[] values:
                        writer.Write(9u); writer.Write(8u); writer.Write((ulong)values.Length);
                        foreach (string text in values) WriteString(writer, text);
                        break;
                }
            }
            ulong offset = 0;
            foreach (var (name, shape, type, bytes) in tensors)
            {
                WriteString(writer, name); writer.Write((uint)shape.Length);
                foreach (ulong dimension in shape) writer.Write(dimension);
                writer.Write(type); writer.Write(offset);
                offset += ((ulong)bytes.Length + 31) / 32 * 32;
            }
            Pad(writer);
            foreach (var (_, _, _, bytes) in tensors) { writer.Write(bytes); Pad(writer); }
        }

        private static void WriteString(BinaryWriter writer, string value)
        {
            byte[] encoded = Encoding.UTF8.GetBytes(value);
            writer.Write((ulong)encoded.Length); writer.Write(encoded);
        }

        private static void Pad(BinaryWriter writer)
        {
            while (writer.BaseStream.Position % 32 != 0) writer.Write((byte)0);
        }

        public void Dispose() => File.Delete(Path);
    }
}
