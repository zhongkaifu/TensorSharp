// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections;
using System.Linq.Expressions;
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models.Architecture;

namespace InferenceWeb.Tests;

/// <summary>
/// The embedding cache is keyed by media content, so concurrent requests carrying the
/// same picture share one key. Every encoder yields the GPU compute lock between
/// blocks, which lets a second preparation encode and cache that picture while the
/// first is still encoding it. The first must then take the cached entry rather than
/// overwrite it: an overwrite dropped the other entry from the dictionary without
/// disposing it or subtracting its bytes, so the byte budget drifted up until every
/// insert evicted the whole cache.
/// </summary>
public sealed class MultimodalEmbeddingCacheRaceTests : IDisposable
{
    private const BindingFlags Private = BindingFlags.NonPublic | BindingFlags.Instance;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-mm-race-" + Guid.NewGuid().ToString("N"));

    public MultimodalEmbeddingCacheRaceTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void TheSameMediaEncodedByTwoInterleavedPreparations_IsCachedAndCountedOnce()
    {
        string image = Path.Combine(_dir, "photo.png");
        File.WriteAllBytes(image, new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 });

        var model = (ProbeModel)RuntimeHelpers.GetUninitializedObject(typeof(ProbeModel));
        using var injector = new ModelMultimodalInjector(model);
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        var type = typeof(ModelMultimodalInjector);
        var cacheType = type.GetNestedType("CachedEmbedding", BindingFlags.NonPublic)!;
        object visionCache = type.GetField("_visionCache", Private)!.GetValue(injector)!;
        MethodInfo getOrCreate = type.GetMethod("GetOrCreateCachedEmbedding", Private)!;
        var history = new List<ChatMessage> { new() { Role = "user", Content = "describe" } };
        var obtained = new Dictionary<int, object>();

        Delegate Encoder(Action whileEncoding)
        {
            Func<string, object> encode = path =>
            {
                whileEncoding();   // the encoder's yield point
                var rows = new Tensor(allocator, DType.Float32, 3, 2);
                return Activator.CreateInstance(cacheType, path, 0L, 0L, rows, 3, 0, 0)!;
            };
            var input = Expression.Parameter(typeof(string));
            return Expression.Lambda(
                typeof(Func<,>).MakeGenericType(typeof(string), cacheType),
                Expression.Convert(Expression.Invoke(Expression.Constant(encode), input), cacheType),
                input).Compile();
        }

        model.Expand = (active, messages, tokens) =>
        {
            Action whileEncoding = tokens[0] == 1
                ? () => active.ProcessPromptTokens(messages, new List<int> { 2 }, "second")
                : () => { };
            obtained[tokens[0]] = getOrCreate.Invoke(active, new[] { visionCache, image, Encoder(whileEncoding) })!;
            return tokens;
        };

        injector.ProcessPromptTokens(history, new List<int> { 1 }, "first");

        var entries = ((IDictionary)visionCache).Values.Cast<object>().ToList();
        long cachedBytes = entries.Sum(e => (long)cacheType.GetProperty("Bytes")!.GetValue(e)!);
        long countedBytes = (long)type.GetField("_embeddingCacheBytes", Private)!.GetValue(injector)!;
        Assert.Single(entries);
        Assert.Equal(cachedBytes, countedBytes);
        Assert.Same(obtained[2], obtained[1]);
    }

    private sealed class ProbeModel : ModelBase, IMultimodalPromptExpander, IVisionCapableModel
    {
        private ProbeModel() : base("unused", BackendType.Cpu) { }
        public Func<ModelMultimodalInjector, List<ChatMessage>, List<int>, List<int>> Expand = null!;
        public bool IsVisionEncoderLoaded => true;
        public void LoadVisionEncoder(string path) => throw new NotSupportedException();
        public void SetVisionEmbeddings(Tensor embeddings, int position) => embeddings.Dispose();
        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> tokens)
            => Expand(injector, history, tokens);
        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }
}
