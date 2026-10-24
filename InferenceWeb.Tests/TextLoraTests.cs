// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.IO;
using System.Text;
using System.Text.Json.Nodes;
using TensorSharp.Models.Lora;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests
{
    /// <summary>
    /// Tests for the text-LoRA loader and registry. The loader's correctness rests on the
    /// A/B transpose and the alpha/rank scale fold, so each test here re-derives the expected
    /// arrays with an independent naive nested-loop reference rather than reusing loader code.
    /// </summary>
    public sealed class TextLoraTests : IDisposable
    {
        private readonly string _dir = Directory.CreateTempSubdirectory("ts-lora-").FullName;

        public void Dispose()
        {
            try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
        }

        [Fact]
        public void Load_TransposesAB_AndFoldsScale()
        {
            const int r = 2, inDim = 3, outDim = 4;
            float alpha = 6f;          // scale = alpha / r = 3
            float scale = alpha / r;

            // A [r, in] row-major, B [out, r] row-major.
            float[] a = { 1, 2, 3, 4, 5, 6 };
            float[] b = { 10, 11, 12, 13, 14, 15, 16, 17 };

            string path = Path.Combine(_dir, "adapter.safetensors");
            WriteSafetensors(path,
                ("blk.0.ffn_down.weight.lora_a", SafetensorDtype.F32, new long[] { r, inDim }, F32Bytes(a)),
                ("blk.0.ffn_down.weight.lora_b", SafetensorDtype.F32, new long[] { outDim, r }, F32Bytes(b)));

            // alpha from config makes scale = alpha / r = 3.
            File.WriteAllText(Path.Combine(_dir, "adapter_config.json"), "{ \"r\": 2, \"lora_alpha\": 6.0 }");

            var adapter = TextLoraLoader.Load("test", path);

            Assert.Equal("test", adapter.Name);
            Assert.Equal(r, adapter.Rank);
            Assert.Equal(alpha, adapter.Alpha);
            Assert.Single(adapter.Deltas);

            var delta = adapter.Deltas["blk.0.ffn_down.weight"];
            Assert.Equal(inDim, delta.In);
            Assert.Equal(outDim, delta.Out);
            Assert.Equal(r, delta.Rank);

            // Independent reference: AT[i, j] = A[j, i]; BT[j, k] = scale * B[k, j].
            float[] expectedAt = new float[inDim * r];
            for (long j = 0; j < r; j++)
                for (long i = 0; i < inDim; i++)
                    expectedAt[i * r + j] = a[j * inDim + i];

            float[] expectedBt = new float[r * outDim];
            for (long k = 0; k < outDim; k++)
                for (long j = 0; j < r; j++)
                    expectedBt[j * outDim + k] = scale * b[k * r + j];

            Assert.Equal(expectedAt, delta.AT);
            Assert.Equal(expectedBt, delta.BT);
        }

        [Fact]
        public void Load_ReadsRankAndAlpha_FromAdapterConfig()
        {
            const int r = 3, inDim = 2, outDim = 2;
            float alpha = 12f; // scale = 4

            string config = Path.Combine(_dir, "adapter_config.json");
            File.WriteAllText(config, "{ \"r\": 3, \"lora_alpha\": 12.0 }");

            string path = Path.Combine(_dir, "adapter.safetensors");
            WriteSafetensors(path,
                ("output.weight.lora_a", SafetensorDtype.F32, new long[] { r, inDim }, F32Bytes(new float[r * inDim])),
                ("output.weight.lora_b", SafetensorDtype.F32, new long[] { outDim, r }, F32Bytes(new float[outDim * r])));

            var adapter = TextLoraLoader.Load("cfg", path, config);

            Assert.Equal(r, adapter.Rank);
            Assert.Equal(alpha, adapter.Alpha);
        }

        [Fact]
        public void Load_EndToEndDelta_MatchesReference()
        {
            const int r = 2, inDim = 3, outDim = 2;
            float alpha = 4f; // scale = 2
            float scale = alpha / r;

            float[] a = { 1, 0, -2, 0.5f, 3, 1 };
            float[] b = { 2, 1, -1, 4 };

            string path = Path.Combine(_dir, "adapter.safetensors");
            WriteSafetensors(path,
                ("w.lora_a", SafetensorDtype.F32, new long[] { r, inDim }, F32Bytes(a)),
                ("w.lora_b", SafetensorDtype.F32, new long[] { outDim, r }, F32Bytes(b)));

            // alpha from config makes scale = alpha / r = 2.
            File.WriteAllText(Path.Combine(_dir, "adapter_config.json"), "{ \"r\": 2, \"lora_alpha\": 4.0 }");

            var adapter = TextLoraLoader.Load("e2e", path);
            var delta = adapter.Deltas["w"];

            // x = [1.0, -0.5, 2.0]
            float[] x = { 1.0f, -0.5f, 2.0f };

            // Reference 1: delta = scale * x @ A^T @ B^T, computed from the ORIGINAL A and B
            // with an explicit three-step nested loop (independent of the loader's path).
            float[] h = new float[r];
            for (int j = 0; j < r; j++)
                for (int i = 0; i < inDim; i++)
                    h[j] += x[i] * a[j * inDim + i]; // x @ A^T

            float[] reference = new float[outDim];
            for (int k = 0; k < outDim; k++)
                for (int j = 0; j < r; j++)
                    reference[k] += scale * h[j] * b[k * r + j]; // (scale*B)^T over h

            // Reference 2: same delta, but from the loader's AT/BT through the generic
            // matmul (x @ AT @ BT) so a faulty transpose inside the loader shows up.
            float[] viaLoaded = MatMul(MatMul(x, delta.AT, inDim, r), delta.BT, r, outDim);

            for (int k = 0; k < outDim; k++)
                Assert.Equal(reference[k], viaLoaded[k], 6);
        }

        [Fact]
        public void Load_MissingLoraA_Throws()
        {
            string path = Path.Combine(_dir, "partial.safetensors");
            WriteSafetensors(path,
                ("w.lora_b", SafetensorDtype.F32, new long[] { 4, 2 }, F32Bytes(new float[8])));

            Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("partial", path));
        }

        [Fact]
        public void Load_RankMismatch_Throws()
        {
            string path = Path.Combine(_dir, "mismatch.safetensors");
            WriteSafetensors(path,
                ("w.lora_a", SafetensorDtype.F32, new long[] { 2, 3 }, F32Bytes(new float[6])),
                ("w.lora_b", SafetensorDtype.F32, new long[] { 4, 3 }, F32Bytes(new float[12])));

            Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("mismatch", path));
        }

        [Fact]
        public void Registry_LoadListGetUnload_ReplacesByName()
        {
            var registry = new TextLoraRegistry();
            try
            {
                string path = Path.Combine(_dir, "a.safetensors");
                WriteSafetensors(path,
                    ("w.lora_a", SafetensorDtype.F32, new long[] { 2, 2 }, F32Bytes(new float[4])),
                    ("w.lora_b", SafetensorDtype.F32, new long[] { 2, 2 }, F32Bytes(new float[4])));

                registry.Load("a", path);
                Assert.True(registry.Contains("a"));
                Assert.NotNull(registry.Get("a"));
                Assert.Single(registry.List());
                Assert.Equal("a", registry.List()[0].Name);

                // Reload under the same name replaces (no duplicate), and does not throw.
                registry.Load("a", path);
                Assert.Single(registry.List());

                Assert.True(registry.Unload("a"));
                Assert.False(registry.Contains("a"));
                Assert.False(registry.Unload("a")); // already gone
            }
            finally
            {
                registry.Dispose();
            }
        }

        // ---- helpers --------------------------------------------------------

        private static float[] MatMul(float[] left, float[] right, int k, int n)
        {
            // left [1, k], right [k, n], stored row-major.
            var result = new float[n];
            for (int j = 0; j < n; j++)
                for (int t = 0; t < k; t++)
                    result[j] += left[t] * right[t * n + j];
            return result;
        }

        private static byte[] F32Bytes(float[] vals)
        {
            var b = new byte[vals.Length * 4];
            Buffer.BlockCopy(vals, 0, b, 0, b.Length);
            return b;
        }

        private static void WriteSafetensors(string path, params (string name, SafetensorDtype dt, long[] shape, byte[] data)[] tensors)
        {
            var header = new JsonObject();
            long offset = 0;
            var blobs = new System.Collections.Generic.List<byte[]>();
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