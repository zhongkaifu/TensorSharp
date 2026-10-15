// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed class GlmNativeBatchEvidenceTests
{
    [GgmlFact(BackendType.GgmlCpu)]
    public void SuccessfulNativeGraphsPublishExactN3ThenN2EvidenceAndRefusalClearsIt()
    {
        using var environment = new EnvScope();
        environment.ClearSpeculationVars(); environment.Set("MAX_CONTEXT", "128");
        environment.Set("TS_GLM_NATIVE", null);
        environment.Set("TS_GLM_THREADS", "2"); environment.Set("TS_GLM_UBATCH", "16");
        environment.Set("TS_CPU_MOE_THREADS", "2");
        string directory = Path.Combine(Path.GetTempPath(), "ts-glm-batch-proof-" + Guid.NewGuid());
        Directory.CreateDirectory(directory);
        try
        {
            string path = GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(Path.Combine(directory, "tiny.gguf"),
                4, true, numLayers: 2, mixedAttention: true, routedExperts: true,
                quantizeExperts: true, contextLength: 128, indexerTopK: 8, architecture: "glm5-next");
            using var model = (GlmDsaModel)ModelBase.Create(path, BackendType.GgmlCpu);
            IBatchedPagedModel common = model;
            Assert.Contains("exec=native", model.KVStateFingerprint);
            string[] ids = ["a", "b", "c"];
            int[] positions = [17, 22, 27];
            for (int lane = 0; lane < ids.Length; ++lane)
            {
                Assert.True(model.BindSequenceCache(ids[lane]));
                model.ForwardRefill(Enumerable.Range(0, positions[lane]).Select(i => 3 + (i * 11 + lane) % 90).ToArray());
            }
            Assert.Null(common.LastAcceptedFusedBatch);
            ulong accepted = 0;
            for (int step = 0; step < 3; ++step)
            {
                var output = new float[3][];
                Assert.True(model.TryForwardBatchedFusedDecode(ids, [17 + step, 31 + step, 43 + step], positions, output));
                var native = Assert.IsType<TensorSharp.GGML.GgmlGlmBatchExecution>(model.LastNativeBatchedExecution);
                var evidence = Assert.IsType<FusedBatchExecutionEvidence>(common.LastAcceptedFusedBatch);
                Assert.Equal(accepted + 1, native.ExecutionId); accepted = native.ExecutionId;
                Assert.Equal(native.ExecutionId, evidence.NativeAcceptedExecutionId);
                Assert.Equal(3, evidence.ActiveSequenceCount);
                Assert.True(evidence.NativeNodeCount > 0 && native.BackendSplits > 0);
                Assert.Equal(0, evidence.NativeHostSegmentCount); // scheduler spans are separate diagnostics
                Assert.Equal(positions, native.PositionsBefore);
                Assert.Equal(positions.Select(p => p + 1), native.PositionsAfter);
                Assert.All(output, row => { Assert.Equal(128, row.Length); Assert.All(row, v => Assert.True(float.IsFinite(v))); });
                for (int lane = 0; lane < 3; ++lane) positions[lane]++;
            }
            Assert.False(model.TryForwardBatchedFusedDecode(ids, [17, 31, 43],
                [positions[0] + 1, positions[1], positions[2]], new float[3][]));
            Assert.Null(common.LastAcceptedFusedBatch); Assert.Null(model.LastNativeBatchedExecution);
            model.OnSequenceReleased("a");
            Assert.True(model.TryForwardBatchedFusedDecode(["b", "c"], [31, 43],
                [positions[1], positions[2]], new float[2][]));
            Assert.Equal(accepted + 1, common.LastAcceptedFusedBatch!.NativeAcceptedExecutionId);
            Assert.Equal(2, common.LastAcceptedFusedBatch.ActiveSequenceCount);
            Assert.Equal(new[] { positions[1], positions[2] }, model.LastNativeBatchedExecution.PositionsBefore);
            Assert.Equal(new[] { positions[1] + 1, positions[2] + 1 }, model.LastNativeBatchedExecution.PositionsAfter);
            foreach (string id in ids) model.OnSequenceReleased(id);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void ExactTerminalDonationPreservesNativeKdaAndMlaLogitsAndRejectsRewind()
    {
        using var environment = new EnvScope();
        environment.ClearSpeculationVars(); environment.Set("MAX_CONTEXT", "128");
        environment.Set("TS_GLM_NATIVE", null);
        environment.Set("TS_CPU_MOE_THREADS", "2");
        string directory = Path.Combine(Path.GetTempPath(), "ts-glm-donation-" + Guid.NewGuid());
        Directory.CreateDirectory(directory);
        try
        {
            string path = GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(Path.Combine(directory, "tiny.gguf"),
                4, true, numLayers: 2, mixedAttention: true, routedExperts: true,
                quantizeExperts: true, contextLength: 128, indexerTopK: 8, architecture: "glm5-next");
            using var model = (GlmDsaModel)ModelBase.Create(path, BackendType.GgmlCpu);
            IPrefixCacheModel prefix = model;
            var capabilities = prefix.GetPrefixCacheCapabilities();
            Assert.Equal(EndStateSupport.DonateOnly, capabilities.EndState);
            Assert.Equal(TruncationKind.None, capabilities.Truncation);
            Assert.False(capabilities.CanCaptureCopy);
            int[] prompt = Enumerable.Range(0, 36).Select(i => 3 + i * 11 % 90).ToArray();
            Assert.True(model.BindSequenceCache("reference"));
            model.ForwardRefill(prompt);
            float[] expected = (float[])model.Forward([43]).Clone();
            Assert.True(model.BindSequenceCache("donor"));
            model.ForwardRefill(prompt);
            Assert.False(prefix.TryCaptureDonate("donor", "short", 35, out _));
            Assert.True(prefix.TryCaptureDonate("donor", "exact", 36, out var footprint));
            Assert.Equal(36, footprint.Tokens);
            Assert.False(prefix.CanMaterialize("exact", 36, 35));
            Assert.False(prefix.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, "exact", "clone", 36, 36)));
            Assert.True(prefix.CanMaterialize("exact", 36, 36));
            Assert.True(prefix.TryMaterialize(new MaterializeRequest(MaterializeOp.Donate, "exact", "donee", 36, 36)));
            Assert.False(prefix.CanMaterialize("exact", 36, 36)); // one physical slot has one owner
            Assert.False(model.BindSequenceCache("donee"));
            float[] actual = model.Forward([43]);
            Assert.Equal(expected, actual); // complete native vocabulary, KDA and MLA
            model.OnSequenceReleased("reference"); model.OnSequenceReleased("donor"); model.OnSequenceReleased("donee");
        }
        finally { Directory.Delete(directory, recursive: true); }
    }
}
