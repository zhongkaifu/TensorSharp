// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.CompilerServices;
using ImageMagick;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// Temporal-pair encoding through the shared Qwen-VL tower and the injector's
/// end-to-end video expansion, on a synthetic one-block projector (see
/// <see cref="QwenVLSyntheticMmprojBuilder"/>). Wiring and ordering, not encoder
/// quality: the weights are random.
/// </summary>
public sealed class QwenVLVideoEncoderTests : IDisposable
{
    private const int Side = 64;   // frame side; 2 frames * 64 * 64 sits above the video pixel floor, so no upscale
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "qwenvl-video-" + Guid.NewGuid().ToString("N"));
    private readonly string _mmproj;
    private readonly IAllocator _allocator = new CpuAllocator(BlasEnum.DotNet);

    public QwenVLVideoEncoderTests()
    {
        Directory.CreateDirectory(_directory);
        _mmproj = QwenVLSyntheticMmprojBuilder.Write(Path.Combine(_directory, "mmproj.gguf"));
    }

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, true);
    }

    [Fact]
    public void TemporalPair_MergesBothFramesInOrderAndADuplicatedFrameEqualsTheStillPath()
    {
        using var encoder = new Qwen35VisionEncoder(_mmproj, _allocator);
        Assert.Equal(2, encoder.TemporalPatchSize);
        float[] a = Pixels(seed: 1), b = Pixels(seed: 2);

        using var still = encoder.Encode(a, Side, Side);
        using var duplicated = encoder.Encode(a, a, Side, Side);
        using var forward = encoder.Encode(a, b, Side, Side);
        using var reversed = encoder.Encode(b, a, Side, Side);

        int tokens = (Side / QwenVLSyntheticMmprojBuilder.PatchSize / QwenVLSyntheticMmprojBuilder.MergeSize);
        tokens *= tokens;
        Assert.Equal(new long[] { tokens, QwenVLSyntheticMmprojBuilder.ProjectionDim }, still.Sizes.ToArray());
        Assert.Equal(still.Sizes.ToArray(), forward.Sizes.ToArray());

        // A still image fills both temporal slices with itself (the summed weight), so
        // a pair of the same frame must reproduce it: that pins the temporal im2col
        // layout against the combined-weight path.
        float[] s = Values(still), d = Values(duplicated), f = Values(forward), r = Values(reversed);
        AssertFinite(f); AssertFinite(r);
        for (int i = 0; i < s.Length; i++)
            Assert.True(Math.Abs(s[i] - d[i]) <= 1e-4f * (1 + Math.Abs(s[i])), $"still/duplicated differ at {i}: {s[i]} vs {d[i]}");
        Assert.True(MaxAbsDiff(s, f) > 1e-3f, "a different second frame must change the embedding");
        Assert.True(MaxAbsDiff(f, r) > 1e-3f, "the two temporal slices are distinct weights, so frame order must matter");
    }

    [Fact]
    public void Injector_ExpandsVideoPairsWithIncreasingTemporalIdsAndInjectsEachPair()
    {
        using var encoder = new Qwen35VisionEncoder(_mmproj, _allocator);
        var (model, injector) = Host(encoder);
        try
        {
            string[] frames = { Frame("f0", 255, 0, 0), Frame("f1", 0, 255, 0), Frame("f2", 0, 0, 255) };
            var message = new ChatMessage
            {
                Role = "user", Content = "order?", IsVideo = true,
                ImagePaths = frames.ToList(),
                ImageTimestamps = new() { 0, 1, 2 },
            };
            const int imagePad = 100, videoPad = 101;
            // vision_start <0.5 seconds> vision_start PAIR vision_end <2.0 seconds> vision_start PAIR vision_end vision_end text
            var prompt = new List<int> { 7, 8, 7, videoPad, 9, 8, 7, videoPad, 9, 9, 5 };
            var expanded = injector.ProcessPromptTokens(new() { message }, prompt);

            // 64x64 frames, 3 frames padded to 4: no rescale (16384 px >= the 4096 floor), so one
            // pair is a 4x4 patch grid = 2x2 merged tokens.
            Assert.Equal(prompt.Count - 2 + 2 * 4, expanded.Count);
            Assert.Equal(4, expanded.Count(t => t == videoPad) / 2);
            Assert.Equal(0, expanded.Count(t => t == imagePad));
            int[] positions = injector.TryGetMRoPEPositionsForSlice(null, 0, expanded.Count)!;
            Assert.Equal(3 * expanded.Count, positions.Length);
            int first = 3, second = 3 + 4 + 3;
            Assert.Equal(videoPad, expanded[first]); Assert.Equal(videoPad, expanded[second]);
            int t0 = positions[3 * first], t1 = positions[3 * second];
            Assert.True(t1 > t0, $"second pair must be later in time: {t0} -> {t1}");
            Assert.Equal(new[] { t0, t0, t0, t0 }, Enumerable.Range(first, 4).Select(t => positions[3 * t]));
            Assert.Equal(new[] { t0, t0, t0 + 1, t0 + 1 }, Enumerable.Range(first, 4).Select(t => positions[3 * t + 1]));
            Assert.Equal(new[] { t0, t0 + 1, t0, t0 + 1 }, Enumerable.Range(first, 4).Select(t => positions[3 * t + 2]));
            // The last text token continues the scalar stream past the second pair.
            Assert.Equal(positions[3 * (expanded.Count - 1)], positions[3 * (expanded.Count - 1) + 1]);
            Assert.True(positions[3 * (expanded.Count - 1)] > t1);

            Assert.True(injector.HasPendingEmbeddings(null));
            Assert.True(injector.QueuePromptEmbeddingsForSlice(0, expanded.Count));
            var queued = Field<List<(Tensor Embeddings, int StartPosition)>>(model, "_visionEmbeddingsList");
            Assert.Equal(new[] { first, second }, queued.Select(q => q.StartPosition));
            Assert.All(queued, q => Assert.Equal(new long[] { 4, QwenVLSyntheticMmprojBuilder.ProjectionDim }, q.Embeddings.Sizes.ToArray()));
            float[] pairA = Values(queued[0].Embeddings), pairB = Values(queued[1].Embeddings);
            AssertFinite(pairA); AssertFinite(pairB);

            // Reversing the clip changes which pixels each pair carries, not the coordinates.
            foreach (var q in queued) q.Embeddings.Dispose();
            queued.Clear();
            injector.ClearPreparedPromptState(null);
            var reversed = new ChatMessage
            {
                Role = "user", Content = "order?", IsVideo = true,
                ImagePaths = frames.Reverse().ToList(),
                ImageTimestamps = new() { 0, 1, 2 },
            };
            var expandedReversed = injector.ProcessPromptTokens(new() { reversed }, prompt);
            Assert.Equal(expanded, expandedReversed);
            Assert.Equal(positions, injector.TryGetMRoPEPositionsForSlice(null, 0, expandedReversed.Count));
            Assert.True(injector.QueuePromptEmbeddingsForSlice(0, expandedReversed.Count));
            float[] pairR = Values(queued[0].Embeddings);
            Assert.True(MaxAbsDiff(pairA, pairR) > 1e-3f, "the reversed clip's first pair carries different frames");
            Assert.True(MaxAbsDiff(pairA, pairB) > 1e-3f, "the two pairs of the clip carry different frames");
            foreach (var q in queued) q.Embeddings.Dispose();
        }
        finally
        {
            injector.Dispose();
            GC.SuppressFinalize(model);
        }
    }

    [Fact]
    public void Injector_RefusesVideoWhenTheTokenizerHasNoVideoPad()
    {
        using var encoder = new Qwen35VisionEncoder(_mmproj, _allocator);
        var (model, injector) = Host(encoder, videoPad: -1);
        try
        {
            var message = new ChatMessage
            {
                Role = "user", IsVideo = true,
                ImagePaths = new() { Frame("a", 255, 0, 0), Frame("b", 0, 0, 255) },
                ImageTimestamps = new() { 0, 1 },
            };
            var ex = Assert.Throws<InvalidOperationException>(() => injector.ProcessPromptTokens(new() { message }, new() { 1, 101, 2 }));
            Assert.Contains("<|video_pad|>", ex.Message);
        }
        finally
        {
            injector.Dispose();
            GC.SuppressFinalize(model);
        }
    }

    [Fact]
    public void TemporalPair_RejectsMalformedFrameBuffersBeforeUnsafePatchReads()
    {
        using var encoder = new Qwen35VisionEncoder(_mmproj, _allocator);
        var valid = Pixels(1);
        Assert.Throws<ArgumentException>(() => encoder.Encode(new float[1], new float[1], Side, Side));
        Assert.Throws<ArgumentException>(() => encoder.Encode(valid, valid, Side + 1, Side));
        Assert.Throws<ArgumentException>(() => encoder.Encode(valid, valid, -Side, Side));
        Assert.Throws<ArgumentException>(() => encoder.Encode(valid, valid, int.MaxValue, Side));
        using var recovered = encoder.Encode(valid, valid, Side, Side);
        AssertFinite(Values(recovered));
    }

    [Fact]
    public void VideoCache_ReencodesWhenOlderFrameChangesWithoutChangingPairSizeOrLatestTimestamp()
    {
        using var encoder = new Qwen35VisionEncoder(_mmproj, _allocator);
        var (model, injector) = Host(encoder);
        try
        {
            string first = Path.Combine(_directory, "first.bmp");
            string second = Path.Combine(_directory, "second.bmp");
            void WriteFrame(string path, byte color)
            {
                // Uncompressed 24-bit BMP: changing pixels preserves the exact file
                // length. Side * 3 is already a multiple of the four-byte row stride.
                using var writer = new BinaryWriter(File.Create(path));
                writer.Write((ushort)0x4d42); writer.Write(54 + Side * Side * 3);
                writer.Write(0); writer.Write(54); writer.Write(40);
                writer.Write(Side); writer.Write(Side); writer.Write((ushort)1); writer.Write((ushort)24);
                writer.Write(0); writer.Write(Side * Side * 3);
                writer.Write(0); writer.Write(0); writer.Write(0); writer.Write(0);
                for (int y = 0; y < Side; y++)
                    for (int x = 0; x < Side; x++)
                    { writer.Write(color); writer.Write((byte)(x * 3)); writer.Write((byte)(y * 3)); }
            }
            WriteFrame(first, 16); WriteFrame(second, 200);
            var epoch = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc);
            File.SetLastWriteTimeUtc(first, epoch);
            File.SetLastWriteTimeUtc(second, epoch.AddHours(2));
            var message = new ChatMessage { Role = "user", IsVideo = true,
                ImagePaths = new() { first, second }, ImageTimestamps = new() { 0, 1 } };
            float[] Encode()
            {
                var tokens = injector.ProcessPromptTokens(new() { message }, new() { 7, 101, 9 });
                Assert.True(injector.QueuePromptEmbeddingsForSlice(0, tokens.Count));
                var queued = Field<List<(Tensor Embeddings, int StartPosition)>>(model, "_visionEmbeddingsList");
                float[] values = Values(Assert.Single(queued).Embeddings);
                foreach (var q in queued) q.Embeddings.Dispose();
                queued.Clear();
                injector.ClearPreparedPromptState(null);
                return values;
            }
            float[] before = Encode();
            Assert.Equal(before, Encode());
            long size = new FileInfo(first).Length;
            WriteFrame(first, 120);
            File.SetLastWriteTimeUtc(first, epoch.AddHours(1));
            Assert.Equal(size, new FileInfo(first).Length);
            Assert.True(MaxAbsDiff(before, Encode()) > 1e-3f);
        }
        finally { injector.Dispose(); GC.SuppressFinalize(model); }
    }

    [Fact]
    public void VideoResize_FitsTheWholeClipAgainstTheSharedBudget()
    {
        var processor = new Qwen35ImageProcessor(QwenVLSyntheticMmprojBuilder.PatchSize, QwenVLSyntheticMmprojBuilder.MergeSize);
        // Within budget: rounded to the 32-px factor, frame count irrelevant.
        Assert.Equal((64, 64), processor.SmartResizeVideo(2, 64, 64));
        Assert.Equal((64, 96), processor.SmartResizeVideo(3, 70, 90));
        // Below the floor over the padded clip: scaled up as a whole.
        Assert.Equal((64, 64), processor.SmartResizeVideo(1, 32, 32));
        // Over the budget: 64 frames of 1080p (133 M px) shrink so t*h*w fits 25.2 M.
        var (h, w) = processor.SmartResizeVideo(64, 1080, 1920);
        Assert.True((long)64 * h * w <= Qwen35ImageProcessor.VideoMaxPixels);
        Assert.True(h % 32 == 0 && w % 32 == 0 && h < 1080 && w < 1920);
        Assert.Throws<ArgumentException>(() => processor.SmartResizeVideo(2, 16, 64));
        Assert.Throws<ArgumentOutOfRangeException>(() => processor.SmartResizeVideo(2, 64, 64, temporalPatchSize: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => processor.SmartResizeVideo(2, 64, 64, minPixels: 0));
        Assert.Throws<ArgumentException>(() => processor.SmartResizeVideo(int.MaxValue, 64, 64));
    }

    // ---- helpers -------------------------------------------------------------

    private (Qwen4ExpModel Model, ModelMultimodalInjector Injector) Host(Qwen35VisionEncoder encoder, int videoPad = 101)
    {
        var model = (Qwen4ExpModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen4ExpModel));
        Qwen4ExpMtpMathTests.Set(typeof(ModelBase), model, "<Tokenizer>k__BackingField", new PadTokenizer(imagePad: 100, videoPad: videoPad));
        Qwen4ExpMtpMathTests.Set(typeof(Qwen4ExpModel), model, "<VisionEncoder>k__BackingField", encoder);
        Qwen4ExpMtpMathTests.Set(typeof(Qwen4ExpModel), model, "_visionEmbeddingsList", new List<(Tensor Embeddings, int StartPosition)>());
        return (model, new ModelMultimodalInjector(model));
    }

    private string Frame(string name, byte r, byte g, byte b)
    {
        string path = Path.Combine(_directory, name + ".png");
        // A flat colour with a gradient patch, so the frame is not uniform and the two
        // temporal slices see structure that differs between frames. Explicit bytes
        // rather than a MagickColor: the Q8 and Q16 builds disagree on channel width.
        var rgb = new byte[Side * Side * 3];
        for (int y = 0; y < Side; y++)
            for (int x = 0; x < Side; x++)
            {
                int i = (y * Side + x) * 3;
                bool patch = x < Side / 2 && y < Side / 2;
                rgb[i] = patch ? (byte)(x * 8) : r;
                rgb[i + 1] = patch ? (byte)(y * 8) : g;
                rgb[i + 2] = patch ? (byte)((x + y) * 4) : b;
            }
        using var image = new MagickImage(rgb, new PixelReadSettings((uint)Side, (uint)Side, StorageType.Char, PixelMapping.RGB));
        image.Format = MagickFormat.Png;
        image.Write(path);
        return path;
    }

    private static float[] Pixels(int seed)
    {
        var data = new float[3 * Side * Side];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)Math.Sin((i + 1) * 0.013 * seed + seed);
        return data;
    }

    private static float[] Values(Tensor t)
    {
        using var contiguous = t.IsContiguous() ? null : Ops.NewContiguous(t);
        return (contiguous ?? t).GetElementsAsFloat((int)t.ElementCount());
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float max = 0;
        for (int i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    private static void AssertFinite(float[] values) => Assert.All(values, v => Assert.True(float.IsFinite(v)));

    private static T Field<T>(object target, string name) => (T)target.GetType()
        .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(target)!;

    /// <summary>Only the two placeholder lookups the injector needs.</summary>
    private sealed class PadTokenizer(int imagePad, int videoPad) : ITokenizer
    {
        public string[] Vocab => Array.Empty<string>();
        public int BosTokenId => 0;
        public int[] EosTokenIds => Array.Empty<int>();
        public int VocabSize => 260;
        public List<int> Encode(string text, bool addSpecial = true) => throw new NotSupportedException();
        public string Decode(List<int> ids) => throw new NotSupportedException();
        public void AppendTokenBytes(int tokenId, List<byte> buffer) => throw new NotSupportedException();
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string tokenStr) => tokenStr switch
        {
            QwenVideoFrames.ImagePad => imagePad,
            QwenVideoFrames.VideoPad => videoPad,
            _ => -1,
        };
    }
}
