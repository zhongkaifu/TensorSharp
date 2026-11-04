// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Text.Json;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q35CudaEngine
    {
        private readonly CudaW8HeadWeightDesc _outputW8;
        private CudaW8G32HeadModule _outputW8Module;
        private IntPtr _outputW8Weights, _outputW8Input, _outputW8Logits;
        private string _outputW8Namespace = string.Empty;
        internal const int OutputW8ScratchRows = CudaW8G32HeadModule.MaximumRows;
        internal const long OutputW8InputBytes = (long)OutputW8ScratchRows * CudaW8G32HeadModule.Width * 2;
        internal const long OutputW8LogitsBytes = (long)OutputW8ScratchRows * CudaW8G32HeadModule.Vocabulary * 2;
        public bool OutputW8Enabled => _outputW8 != null;
        public bool OutputW8TargetEnabled => UsesW8TargetHead(_outputW8);
        public bool OutputW8DraftEnabled => _outputW8 != null && _m.Ninfer == null;
        internal static bool UsesW8TargetHead(CudaW8HeadWeightDesc head)
            => head != null && head.Scope == CudaW8HeadScope.TargetAndDraft;
        public readonly record struct OutputW8Diagnostic(string Profile, string PayloadSha256,
            string ModuleImageSha256, string ModulePath, string Provenance, long PayloadBytes,
            long Bf16InputBytes, long Bf16LogitsBytes, string Consumers, CudaW8HeadScope Scope);
        public OutputW8Diagnostic OutputW8Info { get; private set; }
        private string OutputW8NamespaceSuffix => _outputW8Namespace ?? string.Empty;

        internal static void ValidateOutputW8Model(ModelDesc model, CudaW8HeadWeightDesc head,
            bool verifyGraphs, bool draftGraphs)
        {
            if (head == null) return;
            if (model.Ninfer == null && (model.NEmbd != CudaW8G32HeadModule.Width || model.NVocab != CudaW8G32HeadModule.Vocabulary
                || model.Output.GgmlType != TQ6_K || model.Output.Ne0 != CudaW8G32HeadModule.Width
                || model.Output.Ne1 != CudaW8G32HeadModule.Vocabulary || model.Output.Ne2 != 1
                || model.Output.RowBytes != CudaW8G32HeadModule.Width / 256 * 210))
                throw new NotSupportedException("The W8 companion only replaces the existing 248320x5120 Q6_K output head.");
            if (verifyGraphs || draftGraphs)
                throw new NotSupportedException("The W8 companion has not validated either CUDA graph path; both graph opt-ins must be OFF.");
        }
        internal static string OutputW8StateSuffix(string profile, string payloadSha, string moduleSha,
            CudaW8HeadScope scope = CudaW8HeadScope.TargetAndDraft)
        {
            if (scope != CudaW8HeadScope.TargetAndDraft && scope != CudaW8HeadScope.DraftOnly)
                throw new ArgumentOutOfRangeException(nameof(scope));
            if (string.IsNullOrEmpty(profile) || profile.Length > 128)
                throw new ArgumentException("Loaded companion profile and full payload/module SHA identities are required.");
            foreach (char c in profile)
                if (!char.IsAsciiLetterOrDigit(c) && c != '-' && c != '_' && c != '.')
                    throw new ArgumentException("The companion profile contains an unsupported character.");
            foreach (string sha in new[] { payloadSha, moduleSha })
            {
                if (sha?.Length != 64) throw new ArgumentException("Full loaded SHA256 identities are required.");
                foreach (char c in sha)
                    if (!Uri.IsHexDigit(c)) throw new ArgumentException("A loaded SHA256 identity is not hexadecimal.");
            }
            string policy = scope == CudaW8HeadScope.DraftOnly ? "draft-head-w8-bf16-chunk33-v1" : "output-head-w8-bf16-chunk33-v1";
            return "|" + policy + ":" + profile + ":" + payloadSha.ToLowerInvariant() + ":" + moduleSha.ToLowerInvariant();
        }
        internal readonly record struct OutputW8Chunk(int FirstRow, int Rows, long InputByteOffset, long OutputByteOffset);
        internal static OutputW8Chunk NextOutputW8Chunk(int firstRow, int totalRows, long inputBytes, long outputBytes)
        {
            if (firstRow < 0 || totalRows <= 0 || firstRow >= totalRows
                || inputBytes < checked((long)totalRows * CudaW8G32HeadModule.Width * 4)
                || outputBytes < checked((long)totalRows * CudaW8G32HeadModule.Vocabulary * 4))
                throw new ArgumentOutOfRangeException(nameof(totalRows), "Complete original F32 input/output row capacity is required.");
            return new OutputW8Chunk(firstRow, Math.Min(OutputW8ScratchRows, totalRows - firstRow),
                checked((long)firstRow * CudaW8G32HeadModule.Width * 4),
                checked((long)firstRow * CudaW8G32HeadModule.Vocabulary * 4));
        }
        internal static void ValidateOutputW8Buffers(IntPtr input, int rows, long inputBytes, IntPtr output, long outputBytes)
        {
            if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
            long xBytes = checked((long)rows * CudaW8G32HeadModule.Width * 4);
            long yBytes = checked((long)rows * CudaW8G32HeadModule.Vocabulary * 4);
            ulong x = CudaW8G32HeadModule.CheckBuffer(input, inputBytes, xBytes, 4);
            ulong y = CudaW8G32HeadModule.CheckBuffer(output, outputBytes, yBytes, 4);
            CudaW8G32HeadModule.Disjoint(x, xBytes, y, yBytes);
        }
        internal static IntPtr OutputW8RowPointer(IntPtr pointer, long byteOffset)
        {
            if (pointer.ToInt64() <= 0 || byteOffset < 0) throw new ArgumentOutOfRangeException(nameof(byteOffset));
            return new IntPtr(checked(pointer.ToInt64() + byteOffset));
        }
        internal static void ValidateOutputW8Scales(ReadOnlySpan<byte> bytes, long relativeOffset)
        {
            if (relativeOffset < 0 || (relativeOffset & 1) != 0 || (bytes.Length & 1) != 0
                || relativeOffset > CudaW8G32HeadModule.WeightBytes - bytes.Length)
                throw new ArgumentOutOfRangeException(nameof(relativeOffset));
            long begin = Math.Max(relativeOffset, CudaW8G32HeadModule.ScalesOffset);
            for (long at = begin; at < relativeOffset + bytes.Length; at += 2)
            {
                ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(bytes.Slice(checked((int)(at - relativeOffset)), 2));
                if ((bits & 0x7c00) == 0x7c00)
                    throw new InvalidOperationException("The original W8 payload contains a nonfinite F16 group scale.");
            }
        }
        private void InitializeOutputW8Module()
        {
            if (_outputW8 == null) return;
            if (_alloc.Kernels == null) throw new NotSupportedException("The original flat CUDA module is required for the companion BF16/F32 bit bridge.");
            _alloc.Kernels.EnsureW8HeadBitBridge();
            _outputW8Module = CudaW8G32HeadModule.LoadFromFile(_outputW8.ModulePath);
        }
        private void AllocateOutputW8Scratch()
        {
            if (_outputW8 == null) return;
            _outputW8Input = Alloc(OutputW8InputBytes);
            _outputW8Logits = Alloc(OutputW8LogitsBytes);
            var spans = new List<ScratchBufferSpan>(ScratchBufferSpans);
            spans.Add(new("_outputW8Input", "output-head-io", _outputW8Input, OutputW8InputBytes, OutputW8InputBytes));
            spans.Add(new("_outputW8Logits", "output-head-io", _outputW8Logits, OutputW8LogitsBytes, OutputW8LogitsBytes));
            ScratchBufferSpans = spans.AsReadOnly();
            OutputW8Info = new(_outputW8.Profile, _outputW8PayloadSha, _outputW8Module.ImageSha256.ToLowerInvariant(),
                _outputW8.ModulePath, _outputW8.Provenance, _outputW8.Bytes, OutputW8InputBytes, OutputW8LogitsBytes,
                _m.Ninfer != null ? "target-logits" : OutputW8TargetEnabled ? "target-logits-and-draft-proposals" : "draft-proposals-only", _outputW8.Scope);
            _outputW8Namespace = OutputW8StateSuffix(_outputW8.Profile, _outputW8PayloadSha, _outputW8Module.ImageSha256, _outputW8.Scope);
            Console.Error.WriteLine((_m.Ninfer != null ? "[q35-cuda] NInfer target W8 output " : "[q35-cuda] experimental output-head companion ") + JsonSerializer.Serialize(OutputW8Info));
        }
        private string _outputW8PayloadSha;
        private void OutputW8Head(IntPtr input, int rows, IntPtr output, long outputBytes)
        {
            if (_outputW8Module == null) throw new InvalidOperationException("The explicit W8 companion is not loaded.");
            long inputBytes = checked((long)rows * _m.NEmbd * 4);
            ValidateOutputW8Buffers(input, rows, inputBytes, output, outputBytes);
            for (int first = 0; first < rows;)
            {
                var chunk = NextOutputW8Chunk(first, rows, inputBytes, outputBytes);
                _outputW8Module.LaunchF32(_outputW8Weights, _outputW8.Bytes,
                    OutputW8RowPointer(input, chunk.InputByteOffset), checked((long)chunk.Rows * _m.NEmbd * 4),
                    _outputW8Input, OutputW8InputBytes, _outputW8Logits, OutputW8LogitsBytes, chunk.Rows, Stream);
                // This ordinary same-stream consumer completes its reads before the next
                // chunk can reuse the BF16 scratch. No extra synchronization is introduced.
                _alloc.Kernels.LaunchW8HeadBitBridge(_outputW8Logits, OutputW8RowPointer(output, chunk.OutputByteOffset),
                    checked(chunk.Rows * _m.NVocab), Stream);
                first = checked(first + chunk.Rows);
            }
        }
    }
}
