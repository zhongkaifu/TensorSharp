using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models.Architecture;

namespace InferenceWeb.Tests;

public class MultimodalPreparationIsolationTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task InterleavedEncoders_DeliverEachRequestsOwnEmbeddingRows(bool vision)
    {
        // No checkpoint is needed: exercise the real injector's preparation and
        // delivery around a deterministic pause at an encoder's yield boundary.
        var model = (ProbeModel)RuntimeHelpers.GetUninitializedObject(typeof(ProbeModel));
        using var injector = new ModelMultimodalInjector(model);
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        using var left = new Tensor(allocator, DType.Float32, 2, 1);
        using var right = new Tensor(allocator, DType.Float32, 2, 1);
        left.SetElementsAsFloat(new[] { 11f, 12f });
        right.SetElementsAsFloat(new[] { 21f, 22f });
        using var rendezvous = new Barrier(2);
        var appendLock = new object();
        model.Expand = (active, history, tokens) =>
        {
            Assert.True(rendezvous.SignalAndWait(TimeSpan.FromSeconds(10)));
            lock (appendLock)
                AppendPrepared(active, tokens[0] == 1 ? left : right, vision);
            return tokens;
        };
        var history = new List<ChatMessage> { new() { Role = "user", Content = "audio" } };
        await Task.WhenAll(
            Task.Run(() => injector.ProcessPromptTokens(history, new List<int> { 1 }, "left")),
            Task.Run(() => injector.ProcessPromptTokens(history, new List<int> { 2 }, "right")));

        model.Delivered = new List<float[]>();
        Assert.True(injector.QueuePromptEmbeddingsForSlice(0, 3, "left"));
        Assert.Equal(new[] { 11f, 12f }, Assert.Single(model.Delivered));
        model.Delivered.Clear();
        Assert.True(injector.QueuePromptEmbeddingsForSlice(0, 3, "right"));
        Assert.Equal(new[] { 21f, 22f }, Assert.Single(model.Delivered));
        injector.ClearPreparedPromptState("left");
        Assert.False(injector.HasPendingEmbeddings("left"));
        Assert.True(injector.HasPendingEmbeddings("right"));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NestedPreparation_RestoresOuterRequestAfterCompletionOrFailure(bool fail)
    {
        var model = (ProbeModel)RuntimeHelpers.GetUninitializedObject(typeof(ProbeModel));
        using var injector = new ModelMultimodalInjector(model);
        using var rows = new Tensor(new CpuAllocator(BlasEnum.DotNet), DType.Float32, 2, 1);
        rows.SetElementsAsFloat(new[] { 31f, 32f });
        var history = new List<ChatMessage> { new() { Role = "user", Content = "audio" } };
        model.Expand = (active, messages, tokens) =>
        {
            if (tokens[0] == 1)
            {
                if (fail)
                    Assert.Throws<InvalidOperationException>(() =>
                        active.ProcessPromptTokens(messages, new List<int> { 2 }, "inner"));
                else
                    active.ProcessPromptTokens(messages, new List<int> { 2 }, "inner");
                AppendPrepared(active, rows, false);
            }
            else if (fail)
                throw new InvalidOperationException("encoder failure");
            return tokens;
        };
        injector.ProcessPromptTokens(history, new List<int> { 1 }, "outer");
        Assert.True(injector.HasPendingEmbeddings("outer"));
        Assert.False(injector.HasPendingEmbeddings("inner"));
        Assert.False(injector.HasPendingEmbeddings(""));
    }

    private static void AppendPrepared(ModelMultimodalInjector injector, Tensor tensor, bool vision)
    {
        // Supply a tiny encoder result at the existing private hand-off, then
        // verify it through public per-request delivery. The regression is in
        // request ownership, independent of the encoder's numerical calculation.
        const BindingFlags flags = BindingFlags.NonPublic | BindingFlags.Instance;
        var type = typeof(ModelMultimodalInjector);
        var cacheType = type.GetNestedType("CachedEmbedding", BindingFlags.NonPublic)!;
        var spanType = type.GetNestedType("PreparedEmbeddingSpan", BindingFlags.NonPublic)!;
        object cached = Activator.CreateInstance(cacheType, "fixture", 0L, 0L, tensor, 2, 0, 0)!;
        object span = Activator.CreateInstance(spanType, cached, 1, 0, 3)!;
        string member = vision ? "_preparedVisionEmbeddings" : "_preparedAudioEmbeddings";
        object bucket = type.GetField(member, flags)?.GetValue(injector)
            ?? type.GetProperty(member, flags)!.GetValue(injector)!;
        ((IList)bucket).Add(span);
    }

    private sealed class ProbeModel : ModelBase, IMultimodalPromptExpander, IAudioCapableModel, IVisionCapableModel
    {
        private ProbeModel() : base("unused", BackendType.Cpu) { }
        public Func<ModelMultimodalInjector, List<ChatMessage>, List<int>, List<int>> Expand = null!;
        public List<float[]> Delivered = null!;
        public bool IsVisionEncoderLoaded => true;
        public void LoadVisionEncoder(string path) => throw new NotSupportedException();
        public void SetVisionEmbeddings(Tensor embeddings, int position) => SetAudioEmbeddings(embeddings, position);
        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> tokens)
            => Expand(injector, history, tokens);
        public void SetAudioEmbeddings(Tensor embeddings, int position)
        {
            using (embeddings)
                Delivered.Add(embeddings.GetElementsAsFloat((int)embeddings.ElementCount()));
        }
        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }
}
