// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json.Nodes;
using TensorSharp.Models.Moe;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests
{
    /// <summary>
    /// Tests for the standalone MoE module loader and registry. Expected values are
    /// re-derived with naive index math rather than reusing loader internals, so a
    /// wrong stride or shape interpretation in the loader surfaces here.
    /// </summary>
    public sealed class MoeModuleTests : IDisposable
    {
        private readonly string _dir = Directory.CreateTempSubdirectory("ts-moe-").FullName;

        public void Dispose()
        {
            try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
        }

        [Fact]
        public void Load_SplitsLayersAndShapes_AndReadsMetadata()
        {
            const int numExperts = 3, nFf = 4, hidden = 5;

            // Fill gate/up/down with a value that encodes (expert, row, col) so the
            // round-trip proves the loader kept the expert-major [N, R, C] order.
            float[] gate = Pattern(numExperts, nFf, hidden);
            float[] up = Pattern(numExperts, nFf, hidden);
            float[] down = Pattern(numExperts, hidden, nFf);
            float[] router = new float[numExperts * hidden]; // layer 11 only

            string path = Path.Combine(_dir, "coding.safetensors");
            WriteSafetensors(path,
                new Dictionary<string, string>
                {
                    ["architecture"] = "gemma4",
                    ["num_experts"] = "3",
                    ["target_layers"] = "[10, 11]",
                    ["expert_type"] = "coding",
                },
                ("experts.10.ffn_gate.weight", SafetensorDtype.F32, new long[] { numExperts, nFf, hidden }, F32Bytes(gate)),
                ("experts.10.ffn_up.weight", SafetensorDtype.F32, new long[] { numExperts, nFf, hidden }, F32Bytes(up)),
                ("experts.10.ffn_down.weight", SafetensorDtype.F32, new long[] { numExperts, hidden, nFf }, F32Bytes(down)),
                ("experts.11.ffn_gate.weight", SafetensorDtype.F32, new long[] { numExperts, nFf, hidden }, F32Bytes(gate)),
                ("experts.11.ffn_up.weight", SafetensorDtype.F32, new long[] { numExperts, nFf, hidden }, F32Bytes(up)),
                ("experts.11.ffn_down.weight", SafetensorDtype.F32, new long[] { numExperts, hidden, nFf }, F32Bytes(down)),
                ("router.11.weight", SafetensorDtype.F32, new long[] { numExperts, hidden }, F32Bytes(router)));

            var module = MoeModuleLoader.Load("coding", path);

            Assert.Equal("coding", module.Id);
            Assert.Equal("gemma4", module.Architecture);
            Assert.Equal("coding", module.ExpertType);
            Assert.Equal(2, module.Experts.Count);

            foreach (var layer in new[] { 10, 11 })
            {
                var set = module.Experts[layer];
                Assert.Equal(numExperts, set.NumExperts);
                Assert.Equal(nFf, set.Nff);
                Assert.Equal(hidden, set.Hidden);

                // Independent stride check: element [e,r,c] at e*R*C + r*C + c.
                Assert.Equal(gate, set.GateWeights);
                Assert.Equal(up, set.UpWeights);
                Assert.Equal(down, set.DownWeights);
                Assert.Equal(Pattern(3, 4, 5), set.GateWeights); // same reference content
            }

            Assert.False(module.Experts[10].HasRouter);
            Assert.True(module.Experts[11].HasRouter);
            Assert.Equal(router, module.Experts[11].RouterWeights);
        }

        [Fact]
        public void Load_MissingUp_Throws()
        {
            string path = Path.Combine(_dir, "partial.safetensors");
            WriteSafetensors(path,
                new Dictionary<string, string> { ["architecture"] = "gemma4" },
                ("experts.0.ffn_gate.weight", SafetensorDtype.F32, new long[] { 2, 4, 5 }, F32Bytes(new float[2 * 4 * 5])),
                ("experts.0.ffn_down.weight", SafetensorDtype.F32, new long[] { 2, 5, 4 }, F32Bytes(new float[2 * 5 * 4])));

            Assert.Throws<InvalidDataException>(() => MoeModuleLoader.Load("partial", path));
        }

        [Fact]
        public void Load_DownShapeMismatch_Throws()
        {
            string path = Path.Combine(_dir, "badshape.safetensors");
            WriteSafetensors(path,
                new Dictionary<string, string> { ["architecture"] = "gemma4" },
                ("experts.0.ffn_gate.weight", SafetensorDtype.F32, new long[] { 2, 4, 5 }, F32Bytes(new float[2 * 4 * 5])),
                ("experts.0.ffn_up.weight", SafetensorDtype.F32, new long[] { 2, 4, 5 }, F32Bytes(new float[2 * 4 * 5])),
                ("experts.0.ffn_down.weight", SafetensorDtype.F32, new long[] { 2, 5, 5 }, F32Bytes(new float[2 * 5 * 5]))); // wrong n_ff

            Assert.Throws<InvalidDataException>(() => MoeModuleLoader.Load("badshape", path));
        }

        [Fact]
        public void Load_ExpertCountMismatch_Throws()
        {
            string path = Path.Combine(_dir, "badcount.safetensors");
            WriteSafetensors(path,
                new Dictionary<string, string> { ["architecture"] = "gemma4", ["num_experts"] = "3" },
                ("experts.0.ffn_gate.weight", SafetensorDtype.F32, new long[] { 2, 4, 5 }, F32Bytes(new float[2 * 4 * 5])),
                ("experts.0.ffn_up.weight", SafetensorDtype.F32, new long[] { 2, 4, 5 }, F32Bytes(new float[2 * 4 * 5])),
                ("experts.0.ffn_down.weight", SafetensorDtype.F32, new long[] { 2, 5, 4 }, F32Bytes(new float[2 * 5 * 4])));

            Assert.Throws<InvalidDataException>(() => MoeModuleLoader.Load("badcount", path));
        }

        [Fact]
        public void Load_TargetLayersMetadataMismatch_Throws()
        {
            string path = Path.Combine(_dir, "badlayers.safetensors");
            WriteSafetensors(path,
                new Dictionary<string, string> { ["architecture"] = "gemma4", ["target_layers"] = "[7]" },
                ("experts.0.ffn_gate.weight", SafetensorDtype.F32, new long[] { 2, 4, 5 }, F32Bytes(new float[2 * 4 * 5])),
                ("experts.0.ffn_up.weight", SafetensorDtype.F32, new long[] { 2, 4, 5 }, F32Bytes(new float[2 * 4 * 5])),
                ("experts.0.ffn_down.weight", SafetensorDtype.F32, new long[] { 2, 5, 4 }, F32Bytes(new float[2 * 5 * 4])));

            Assert.Throws<InvalidDataException>(() => MoeModuleLoader.Load("badlayers", path));
        }

        [Fact]
        public void Registry_LoadListGetUnload_AndRegisterInProcess()
        {
            var registry = new MoeRegistry();
            try
            {
                string path = Path.Combine(_dir, "m.safetensors");
                WriteSafetensors(path,
                    new Dictionary<string, string> { ["architecture"] = "gemma4", ["num_experts"] = "2" },
                    ("experts.0.ffn_gate.weight", SafetensorDtype.F32, new long[] { 2, 2, 2 }, F32Bytes(new float[8])),
                    ("experts.0.ffn_up.weight", SafetensorDtype.F32, new long[] { 2, 2, 2 }, F32Bytes(new float[8])),
                    ("experts.0.ffn_down.weight", SafetensorDtype.F32, new long[] { 2, 2, 2 }, F32Bytes(new float[8])));

                registry.Load("m", path);
                Assert.True(registry.Contains("m"));
                Assert.NotNull(registry.Get("m"));
                Assert.Single(registry.List());
                Assert.Equal(1, registry.List()[0].LayerCount);
                Assert.Equal(2, registry.List()[0].NumExperts);
                Assert.False(registry.List()[0].HasRouter);

                // RegisterInProcess adopts an already-built module without copying.
                var module = MoeModuleLoader.Load("m2", path);
                registry.RegisterInProcess(module);
                Assert.Equal(2, registry.List().Count);
                Assert.Same(module, registry.Get("m2"));

                Assert.True(registry.Unload("m"));
                Assert.False(registry.Unload("m"));
            }
            finally
            {
                registry.Dispose();
            }
        }

        [Fact]
        public void Registry_ActiveSelection_IsJustData()
        {
            var registry = new MoeRegistry();
            try
            {
                registry.ActiveSelection = new MoeSelection
                {
                    LayerOverrides = new Dictionary<int, string> { [10] = "coding" },
                };
                Assert.Equal("coding", registry.ActiveSelection!.LayerOverrides[10]);
                Assert.False(registry.ActiveSelection.LayerOverrides.ContainsKey(11));
            }
            finally
            {
                registry.Dispose();
            }
        }

        // ---- helpers --------------------------------------------------------

        // Fills [n0][n1][n2] with element index-encoded values so a bad stride is visible.
        private static float[] Pattern(int n0, int n1, int n2)
        {
            var v = new float[n0 * n1 * n2];
            for (int a = 0; a < n0; a++)
                for (int b = 0; b < n1; b++)
                    for (int c = 0; c < n2; c++)
                        v[a * n1 * n2 + b * n2 + c] = a * 1000 + b * 100 + c;
            return v;
        }

        private static byte[] F32Bytes(float[] vals)
        {
            var b = new byte[vals.Length * 4];
            Buffer.BlockCopy(vals, 0, b, 0, b.Length);
            return b;
        }

        private static void WriteSafetensors(
            string path,
            IReadOnlyDictionary<string, string> metadata,
            params (string name, SafetensorDtype dt, long[] shape, byte[] data)[] tensors)
        {
            var header = new JsonObject();
            if (metadata.Count > 0)
            {
                var meta = new JsonObject();
                foreach (var (k, v) in metadata)
                    meta[k] = v;
                header["__metadata__"] = meta;
            }

            long offset = 0;
            var blobs = new List<byte[]>();
            foreach (var (name, dt, shape, data) in tensors)
            {
                var shapeArr = new JsonArray();
                foreach (var d in shape) shapeArr.Add(d);
                header[name] = new JsonObject
                {
                    ["dtype"] = dt switch
                    {
                        SafetensorDtype.F32 => "F32",
                        SafetensorDtype.F16 => "F16",
                        SafetensorDtype.BF16 => "BF16",
                        _ => "F32",
                    },
                    ["shape"] = shapeArr,
                    ["data_offsets"] = new JsonArray(offset, offset + data.Length),
                };
                offset += data.Length;
                blobs.Add(data);
            }

            byte[] hb = Encoding.UTF8.GetBytes(header.ToJsonString());
            using var fs = File.Create(path);
            fs.Write(BitConverter.GetBytes((ulong)hb.Length));
            fs.Write(hb);
            foreach (var blob in blobs)
                fs.Write(blob);
        }
    }
}