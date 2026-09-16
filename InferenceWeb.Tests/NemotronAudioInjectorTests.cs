using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class NemotronAudioInjectorTests
{
    [Fact]
    public void AudioOnlyTwoClipHistory_QueuesExactRowsAndKeepsOtherRequestAlive()
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-nemotron-injector-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        var pending = new List<(Tensor embeddings, int position)>();
        try
        {
            using var reference = NemotronAudioEncoderTests.ReadReference(128);
            string companion = Path.Combine(directory, "audio.gguf");
            NemotronAudioEncoderTests.WriteCompanion(companion, reference.RootElement.GetProperty("fixtures")[0], false, 128);
            using var encoder = new NemotronAudioEncoder(companion, new CpuAllocator(BlasEnum.DotNet));
            float[] a = Enumerable.Range(0, 1280).Select(i => .2f * MathF.Sin(i * .13f)).ToArray();
            float[] b = Enumerable.Range(0, 2560).Select(i => .3f * MathF.Cos(i * .09f)).ToArray();
            string aPath = Path.Combine(directory, "a.wav"), bPath = Path.Combine(directory, "b.wav");
            WriteWav(aPath, a); WriteWav(bPath, b);
            var aMel = NemotronAudioPreprocessor.ComputeParakeetMelSpectrogram(a);
            var bMel = NemotronAudioPreprocessor.ComputeParakeetMelSpectrogram(b);
            using var aExpected = encoder.Encode(aMel.mel, aMel.frames, aMel.validFrames);
            using var bExpected = encoder.Encode(bMel.mel, bMel.frames, bMel.validFrames);
            var model = (NemotronModel)RuntimeHelpers.GetUninitializedObject(typeof(NemotronModel));
            typeof(ModelBase).GetProperty(nameof(ModelBase.Tokenizer))!.SetValue(model, new SentinelTokenizer());
            typeof(NemotronModel).GetField("_audioEncoder", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, encoder);
            typeof(NemotronModel).GetField("_pendingAudioEmbeddings", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, pending);
            using var injector = new ModelMultimodalInjector(model);
            var history = new List<ChatMessage> { new() { Role = "user", AudioPaths = new() { aPath, bPath } } };
            var expanded = injector.ProcessPromptTokens(history, new() { 1, 27, 2, 27, 3 }, "a");
            Assert.Equal(new[] { 1, 28, 27, 27, 29, 2, 28, 27, 27, 27, 29, 3 }, expanded);
            Assert.Null(model.VisionEncoder);
            Assert.Equal(2, injector.ClampReusablePrefix(3, "a"));
            Assert.Equal(5, injector.ClampTrimStart(3, "a"));
            // A second holder survives A's sliced queue and subsequent release.
            Assert.Equal(new[] { 28, 27, 27, 27, 29 }, injector.ProcessPromptTokens(
                new() { new() { Role = "user", AudioPaths = new() { bPath } } }, new() { 27 }, "b"));
            Assert.True(injector.QueuePromptEmbeddingsForSlice(3, 6, "a"));
            Assert.Equal(2, pending.Count);
            Assert.Equal(0, pending[0].position); Assert.Equal(4, pending[1].position);
            Assert.Equal(aExpected.GetElementsAsFloat(12).Skip(6), pending[0].embeddings.GetElementsAsFloat(6));
            Assert.Equal(bExpected.GetElementsAsFloat(18).Take(12), pending[1].embeddings.GetElementsAsFloat(12));
            DisposePending(pending);
            injector.ClearPreparedPromptState("a");
            Assert.False(injector.HasPendingEmbeddings("a")); Assert.True(injector.HasPendingEmbeddings("b"));
            Assert.True(injector.QueuePromptEmbeddings(0, "b"));
            Assert.Single(pending); Assert.Equal(1, pending[0].position);
            Assert.Equal(bExpected.GetElementsAsFloat(18), pending[0].embeddings.GetElementsAsFloat(18));
            DisposePending(pending);
            Assert.False(injector.QueuePromptEmbeddingsForSlice(0, 1, "b"));
        }
        finally { DisposePending(pending); Directory.Delete(directory, true); }
    }

    private static void DisposePending(List<(Tensor embeddings, int position)> pending)
    { foreach (var item in pending) item.embeddings.Dispose(); pending.Clear(); }

    private static void WriteWav(string path, float[] samples)
    {
        using var writer = new BinaryWriter(File.Create(path), Encoding.ASCII);
        writer.Write(Encoding.ASCII.GetBytes("RIFF")); writer.Write(36 + samples.Length * 4); writer.Write(Encoding.ASCII.GetBytes("WAVEfmt "));
        writer.Write(16); writer.Write((ushort)3); writer.Write((ushort)1); writer.Write(16000); writer.Write(64000);
        writer.Write((ushort)4); writer.Write((ushort)32); writer.Write(Encoding.ASCII.GetBytes("data")); writer.Write(samples.Length * 4);
        foreach (float sample in samples) writer.Write(sample);
    }

    private sealed class SentinelTokenizer : ITokenizer
    {
        public string[] Vocab => Array.Empty<string>();
        public int BosTokenId => 0;
        public int[] EosTokenIds => Array.Empty<int>();
        public int VocabSize => 30;
        public List<int> Encode(string text, bool addSpecial = true) => throw new NotSupportedException();
        public string Decode(List<int> ids) => throw new NotSupportedException();
        public void AppendTokenBytes(int tokenId, List<byte> buffer) => throw new NotSupportedException();
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string tokenStr) => tokenStr switch
        { "<image>" => 18, "<img>" => 19, "</img>" => 20, "<so_embedding>" => 27, "<so_start>" => 28, "<so_end>" => 29, _ => -1 };
    }
}
