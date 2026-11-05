// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q35CudaEngine
    {
        private CudaNinferKernels _ninfer;
        private CudaNinferAttention _ninferAttention;
        private NinferLayer[] _ninferLayers;
        private NinferMatrix _ninferHead;
        // The last layer's residual state: a fused pass whose final Down was not fused (an FP8
        // layer) leaves its projection in _block for the final norm to add.
        private NinferResidualState _ninferFinalResidualState;
        private IntPtr _fp8ActCodes, _fp8ActScales, _fp8ActBf16;
        private IntPtr _ninferAlphaBeta;
        private sealed class NinferMatrix
        {
            public CudaNinferProjectionDesc Descriptor;
            public IntPtr Data, Scales;
        }
        private sealed class NinferLayer
        {
            public NinferMatrix Input, AlphaBeta, Output, GateUp, Down;
        }
        public bool IsNinferModel => _m.Ninfer != null;
        public bool NinferPrefillAttentionEnabled => _ninferAttention?.PrefillEnabled == true;
        public string NinferPrefillAttentionImageSha256 => _ninferAttention?.PrefillImageSha256;
        public long NinferPrefillAttentionCoreSubmissions => _ninferAttention?.PrefillCoreSubmissions ?? 0;
        public long NinferPrefillAttentionWorkspaceBytes => _ninferAttention?.PrefillWorkspaceBytes ?? 0;
        public long NinferPrefill128CoreSubmissions => _ninferAttention?.Prefill128CoreSubmissions ?? 0;

        internal static void ValidateNinferModel(ModelDesc model)
        {
            var recipe = model.Ninfer;
            if (recipe == null) return;
            if (model.NLayer != 64 || model.NEmbd != 5120 || model.NFf != 17408 || model.NVocab != 248320
                || model.NHead != 24 || model.NKvHead != 4 || model.HeadDim != 256
                || model.GdnKHeads != 16 || model.GdnVHeads != 48 || model.DConv != 4
                || recipe.Layers?.Length != model.NLayer || model.Layers?.Length != model.NLayer
                || recipe.Embedding?.Format != CudaNinferProjectionFormat.Fp8Row
                || recipe.Embedding.Rows != model.NVocab || recipe.Embedding.Columns != model.NEmbd
                || recipe.Head?.Format != CudaNinferProjectionFormat.Fp8Row
                || recipe.Head.Rows != model.NVocab || recipe.Head.Columns != model.NEmbd
                || string.IsNullOrWhiteSpace(recipe.Provenance))
                throw new NotSupportedException("The complete NInfer route requires the validated Qwen3.8-27B target recipe.");
            if (model.NCtx < 1 || model.NUbatch < 1 || model.NUbatch > model.NCtx
                || !float.IsFinite(model.Eps) || model.Eps <= 0 || !float.IsFinite(model.RopeBase) || model.RopeBase <= 0)
                throw new ArgumentException("Invalid NInfer context or normalization/rotary parameters.");
            Vector(model.OutputNorm, model.NEmbd);
            foreach (string option in new[] { "TS_Q35_VERIFY_DECODE_Q81", "TS_Q35_VERIFY_GRAPHS", "TS_Q35_CUDA_GRAPHS", "TS_Q35_GDN_PREPARED_PREFILL", "TS_Q35_SHARED_SCRATCH" })
                if (Environment.GetEnvironmentVariable(option) == "1")
                    throw new NotSupportedException("The NInfer numerical recipe does not accept the GGUF experimental option " + option);
            for (int i = 0; i < model.NLayer; i++)
            {
                var layer = recipe.Layers[i] ?? throw new ArgumentException("Missing NInfer layer.");
                bool recurrent = (i + 1) % 4 != 0;
                if (model.Layers[i].Recurrent != recurrent)
                    throw new ArgumentException("NInfer layer schedule differs.");
                Vector(model.Layers[i].AttnNorm, model.NEmbd);
                Vector(model.Layers[i].PostNorm, model.NEmbd);
                Shape(layer.Input, recurrent ? 16384 : 14336, model.NEmbd);
                Shape(layer.Output, model.NEmbd, 6144);
                Shape(layer.GateUp, 34816, model.NEmbd);
                Shape(layer.Down, model.NEmbd, model.NFf);
                if (recurrent)
                {
                    Shape(layer.AlphaBeta, 96, model.NEmbd);
                    if (layer.AlphaBeta.Format != CudaNinferProjectionFormat.BFloat16)
                        throw new ArgumentException("NInfer GDN alpha/beta must retain BF16 weights.");
                    Vector(model.Layers[i].ConvW, 40960);
                    Vector(model.Layers[i].DtBias, 48);
                    Vector(model.Layers[i].SsmA, 48);
                    Vector(model.Layers[i].SsmNorm, 128);
                }
                else
                {
                    if (layer.AlphaBeta != null) throw new ArgumentException("An attention layer cannot have GDN gates.");
                    Vector(model.Layers[i].QNorm, 256);
                    Vector(model.Layers[i].KNorm, 256);
                }
            }
            static void Shape(CudaNinferProjectionDesc d, int rows, int columns)
            {
                if (d == null || d.Rows != rows || d.Columns != columns || d.Format == CudaNinferProjectionFormat.W8G32)
                    throw new ArgumentException("NInfer projection geometry differs from the target plan.");
            }
            static void Vector(float[] values, int count)
            {
                if (values?.Length != count) throw new ArgumentException("NInfer parameter vector geometry differs.");
                foreach (float value in values)
                    if (!float.IsFinite(value)) throw new ArgumentException("NInfer parameters must be finite.");
            }
        }
        private void InitializeNinferModules()
        {
            InitializeNinferQualifiedDefaults();
            InitializeNinferVerify8Gemm();
            InitializeNinferA16Candidate();
            InitializeNinferLoadRepack();
            InitializeNinferExactVerify();
            InitializeNinferFlashInfer();
            InitializeNinferResidualNorm();
            InitializeNinferGateUpSwiGlu();
            if (_ninferDecodeGraphsEnabled && _m.Ninfer == null)
                throw new NotSupportedException(NinferDecodeGraphsEnvironment + " requires an original NInfer target.");
            if (_ninferVerificationEnabled && _m.Ninfer == null)
                throw new NotSupportedException(NinferVerificationEnvironment + " requires an original NInfer target.");
            if (_m.Ninfer == null) return;
            _ninfer = CudaNinferKernels.Create(_m.Ninfer.ModulePath);
            _ninferAttention = new CudaNinferAttention(_m.RopeBase, _m.NUbatch, _m.NCtx, _sms,
                _m.NHead, _m.NKvHead, _m.HeadDim, _ninferAutomaticPrefillAttention, _ninferKvFp8);
            Console.Error.WriteLine($"[q35-cuda] NInfer KV cache: {(_ninferKvFp8 ? "FP8 E4M3, group-128 F32 scales (272-byte rows)" : "BF16")}");
            _ninferLayers = new NinferLayer[_m.NLayer];
            InitializeNinferGdnChunk();
        }
        /// <summary>TS_Q35_NINFER_EMBEDDING=host (default) keeps the FP8 embedding table in pinned,
        /// device-mapped host memory: a token gather reads one 5 KB row over PCIe, and the 1.27 GB
        /// of VRAM goes to KV capacity. =device uploads it like the other matrices.</summary>
        internal const string NinferEmbeddingEnvironment = "TS_Q35_NINFER_EMBEDDING";
        private IntPtr _hostEmbedding;
        private IntPtr LoadNinferEmbedding(Loader loader, CudaNinferProjectionDesc d)
        {
            string mode = Environment.GetEnvironmentVariable(NinferEmbeddingEnvironment) ?? "host";
            if (mode == "device") return LoadNinferMatrix(loader, d).Data;
            if (mode != "host") throw new ArgumentException(NinferEmbeddingEnvironment + " accepts only host or device.");
            CudaDriverApi.cuMemHostAlloc(out _hostEmbedding, new UIntPtr((ulong)d.Bytes),
                CudaDriverApi.CU_MEMHOSTALLOC_PORTABLE | CudaDriverApi.CU_MEMHOSTALLOC_DEVICEMAP).ThrowOnError();
            for (long offset = 0; offset < d.Bytes; offset += 64L << 20)
                d.Source.Read(d.SourceOffset + offset, _hostEmbedding + (nint)offset, Math.Min(64L << 20, d.Bytes - offset));
            CudaDriverApi.cuMemHostGetDevicePointer(out IntPtr device, _hostEmbedding, 0).ThrowOnError();
            Console.Error.WriteLine($"[q35-cuda] NInfer FP8 embedding in mapped host memory ({d.Bytes / (1 << 20)} MiB of VRAM kept for KV)");
            return device;
        }
        private void FreeNinferHostEmbedding()
        {
            if (_hostEmbedding != IntPtr.Zero) { CudaDriverApi.cuMemFreeHost(_hostEmbedding); _hostEmbedding = IntPtr.Zero; }
        }

        private NinferMatrix LoadNinferMatrix(Loader loader, CudaNinferProjectionDesc d)
        {
            if (d == null) return null;
            if (TryLoadNinferFlashInferMatrix(loader, d, out var rawMatrix)) return rawMatrix;
            IntPtr data = Alloc(d.Bytes);
            if (!loader.TryRepackNinfer(d, data))
                loader.RawInto(new CudaWeightDesc { Source = d.Source, SourceOffset = d.SourceOffset,
                    Ne0 = d.Columns, Ne1 = 1, Ne2 = 1, RowBytes = d.Bytes }, data);
            return new NinferMatrix { Descriptor = d, Data = data,
                Scales = d.Format switch
                {
                    CudaNinferProjectionFormat.NvFp4Planes => data + (nint)((long)d.Rows*d.Columns/2),
                    CudaNinferProjectionFormat.Fp8Row => data + (nint)((long)d.Rows*d.Columns),
                    _ => IntPtr.Zero
                } };
        }
        private Layer LoadNinferLayer(Loader loader, int index)
        {
            var d = _m.Layers[index];
            var source = _m.Ninfer.Layers[index];
            _ninferLayers[index] = new NinferLayer { Input = LoadNinferMatrix(loader, source.Input),
                AlphaBeta = LoadNinferMatrix(loader, source.AlphaBeta), Output = LoadNinferMatrix(loader, source.Output),
                GateUp = LoadNinferMatrix(loader, source.GateUp), Down = LoadNinferMatrix(loader, source.Down) };
            return new Layer { Recurrent = d.Recurrent, AttnNorm = UploadF32(d.AttnNorm), PostNorm = UploadF32(d.PostNorm),
                QNorm = UploadF32(d.QNorm), KNorm = UploadF32(d.KNorm), ConvW = UploadF32(d.ConvW),
                DtBias = UploadF32(d.DtBias), SsmA = UploadF32(d.SsmA), SsmNorm = UploadF32(d.SsmNorm) };
        }
        private void FinalNorm(IntPtr x, IntPtr delta, IntPtr weight, IntPtr output, int rows, int width, float eps, IntPtr stream)
        {
            if (_m.Ninfer == null) _k.AddRmsNorm(x, delta, weight, output, rows, width, eps, stream);
            else _ninfer.AddRmsNorm(x, delta, weight, output, rows, width, eps, stream);
        }
        private void NinferProject(NinferMatrix matrix, IntPtr input, IntPtr output, int rows,
            int a4Threshold = 4, bool round = true, bool isVerify = false)
        {
            if (matrix.Descriptor.Format == CudaNinferProjectionFormat.Fp8Row)
            {
                ProjectNinferFp8(matrix, input, output, rows, round ? Q35Kernels.Fp8Epilogue.Round : Q35Kernels.Fp8Epilogue.Plain, isVerify);
                return;
            }
            if (TryProjectNinferFlashInfer(matrix, input, output, rows))
            {
                if (round) _ninfer.Round(output, (long)rows * matrix.Descriptor.Rows, Stream);
                return;
            }
            matrix = PrepareNinferFlashInferFallback(matrix);
            var d = matrix.Descriptor;
            if (rows == 1 && TryGemvNinferNvfp4(matrix, input, output, swiGlu: false)) { }
            else if (TryNinferExactVerify(matrix, input, output, rows, a4Threshold, isVerify)) { }
            else if (d.Format == CudaNinferProjectionFormat.BFloat16)
            {
                _ninfer.ProjectBf16(matrix.Data, input, output, d.Rows, d.Columns, rows, Stream);
            }
            else if (d.AllowA4 && rows >= a4Threshold)
            {
                QuantizeNinferInput(input, matrix, rows);
                float alpha = 1.0f / (d.InputScaleDivisor * d.WeightScaleDivisor);
                if (UsesNinferPrefillEpilogue(rows, isVerify)
                    && Q35Kernels.NinferPrefillEpilogueFits(matrix.Data, matrix.Scales, d.Rows, d.Columns))
                {
                    _k.GemmW4A4NinferPrefill(matrix.Data, matrix.Scales, d.Rows, d.Columns, _fp4c, _fp4s, _fp4r,
                        rows, alpha, output, round ? Q35Kernels.NinferPrefillEpilogue.Round : Q35Kernels.NinferPrefillEpilogue.Plain, Stream);
                    return;
                }
                _k.GemmW4A4(matrix.Data, matrix.Scales, d.Rows, d.Columns, _fp4c, _fp4s, _fp4r,
                    rows, alpha, output, _partial, _counters, _sms, rows <= 64 ? 8 : 1, Stream,
                    ninferVerify8: isVerify && rows == 8);
            }
            else if (!TryNinferA16(matrix, input, output, rows))
                _ninfer.ProjectA16(matrix.Data, matrix.Scales, input, output, d.Rows, d.Columns, rows,
                    1.0f / d.WeightScaleDivisor, Stream);
            if (round) _ninfer.Round(output, (long)rows*d.Rows, Stream);
        }
        /// <summary>One-row NVFP4 projection on the CUDA cores (non-speculative decode), unless
        /// TS_Q35_T1_GEMV=off. Same A16 weights; the summation order differs from the legacy kernel.</summary>
        private bool TryGemvNinferNvfp4(NinferMatrix matrix, IntPtr input, IntPtr output, bool swiGlu)
        {
            var d = matrix.Descriptor;
            return Q35Kernels.T1GemvEnabled && d.Format == CudaNinferProjectionFormat.NvFp4Planes
                && _ninfer.TryGemvNvfp4(matrix.Data, matrix.Scales, input, output, d.Rows, d.Columns, 1.0f / d.WeightScaleDivisor, swiGlu, Stream);
        }
        /// <summary>FP8 row-scaled projection: the small-T A16 route for decode/verify rows, the
        /// W8A8 TMA GEMM over per-row E4M3 activations for prefill (NInfer's AllowA8 uses).
        /// SwiGlu takes gate|up interleaved by 8 rows and writes [rows][n / 2].</summary>
        private void ProjectNinferFp8(NinferMatrix matrix, IntPtr input, IntPtr output, int rows, Q35Kernels.Fp8Epilogue epilogue, bool isVerify)
        {
            var d = matrix.Descriptor;
            if (!isVerify && rows >= Q35Kernels.Fp8TmaMinRows && Q35Kernels.Fp8TmaFits(matrix.Data, IntPtr.Zero, d.Rows, d.Columns))
            {
                if (_fp8ActCodes == IntPtr.Zero)
                {
                    _fp8ActCodes = Alloc(checked((long)RowCap * Math.Max(_m.NFf, Math.Max(_m.NEmbd, 6144))));
                    _fp8ActScales = Alloc((long)RowCap * 4);
                }
                _k.QuantizeFp8Activations(input, _fp8ActCodes, _fp8ActScales, d.Columns, rows, Stream);
                var tma = epilogue switch
                {
                    Q35Kernels.Fp8Epilogue.Round => Q35Kernels.Fp8TmaEpilogue.Round,
                    Q35Kernels.Fp8Epilogue.SwiGlu => Q35Kernels.Fp8TmaEpilogue.SwiGlu,
                    _ => Q35Kernels.Fp8TmaEpilogue.Plain
                };
                _k.GemmW8A8(matrix.Data, matrix.Scales, d.Rows, d.Columns, _fp8ActCodes, _fp8ActScales, rows, output, tma, Stream);
                return;
            }
            if (Q35Kernels.Fp8SmallTakesBf16(rows))
            {
                // Wide verify/decode tiles read BF16 activations, cast once for every CTA.
                if (_fp8ActBf16 == IntPtr.Zero) _fp8ActBf16 = Alloc((long)Math.Max(MaxVerifyRows, 64) * Math.Max(_m.NFf, Math.Max(_m.NEmbd, 6144)) * 2);
                _ninfer.CastToBf16(input, _fp8ActBf16, (long)rows * d.Columns, Stream);
                input = _fp8ActBf16;
            }
            _k.ProjectFp8(matrix.Data, matrix.Scales, d.Rows, d.Columns, input, rows, output, epilogue, Stream);
        }

        private void RunNinferLayers(int rows, int tapFirst, RowGroup[] groups, int[] maxKeys)
        {
            if (_d != null)
                throw new NotSupportedException("NInfer target inference cannot use a GGUF drafter.");
            foreach (var group in groups)
                if (group.Verify && (!_ninferVerificationEnabled || groups.Length != 1 || group.Counts != null
                    || group.Rows > MaxVerifyRows || _vQkv == IntPtr.Zero))
                    throw new NotSupportedException("NInfer target verification requires one complete prepared verification group.");
            int e = _m.NEmbd;
            bool isVerify = groups.Length == 1 && groups[0].Verify;
            bool candidatePass = groups.Length == 1 && groups[0].PassMode == NinferLayerPass.FusedResidualNorm;
            bool fusedGateUp = UsesNinferGateUpSwiGlu(groups, tapFirst);
            var residualState = NinferResidualState.InitialNoDelta;
            for (int il = 0; il < _m.NLayer; il++)
            {
                var l = _layers[il]; var weights = _ninferLayers[il];
                NormalizeNinferResidual(residualState, l.AttnNorm, rows, candidatePass, postProjection: false, weights.Input);
                CaptureNinferDraftTargetTap(il, tapFirst, rows);
                NinferProject(weights.Input, _xn, _proj, rows, l.Recurrent ? 1 : 4, isVerify: isVerify);
                if (l.Recurrent)
                {
                    if (_ninferAlphaBeta == IntPtr.Zero) _ninferAlphaBeta = Alloc((long)RowCap*96*4);
                    NinferProject(weights.AlphaBeta, _xn, _ninferAlphaBeta, rows, round: false, isVerify: isVerify);
                    for (int gi = 0, r0 = 0, q0 = 0; gi < groups.Length; r0 += groups[gi].Rows, q0 += groups[gi].Slots.Length, gi++)
                    {
                        var g = groups[gi];
                        long verifyOffset = (long)l.Ordinal * MaxVerifyRows;
                        IntPtr proj = _proj + (nint)((long)r0*16384*4);
                        IntPtr qkv = g.Verify ? _vQkv + (nint)(verifyOffset * ConvDim * 4) : _gdnQkv + (nint)((long)r0*ConvDim*4);
                        IntPtr gd = g.Verify ? _vG + (nint)(verifyOffset * _m.GdnVHeads * 4) : _gdnG + (nint)((long)r0*_m.GdnVHeads*4);
                        IntPtr beta = g.Verify ? _vBeta + (nint)(verifyOffset * _m.GdnVHeads * 4) : _gdnBeta + (nint)((long)r0*_m.GdnVHeads*4);
                        IntPtr raw = g.Verify ? _vRaw + (nint)(verifyOffset * ConvDim * 4) : IntPtr.Zero;
                        IntPtr count = g.Counts != null ? _countDev + (nint)(4L*q0) : IntPtr.Zero;
                        _ninfer.GdnConv(proj, 16384, g.RowsPerSeq, g.Slots.Length, ConvDim, _m.DConv,
                            l.ConvW, TabConv(l) + (nint)(8L*q0), qkv, raw, !g.Verify, Stream, count);
                        _ninfer.GdnPrep(qkv, _ninferAlphaBeta + (nint)((long)r0*96*4), 96, g.Rows, _m.GdnKHeads,
                            _m.GdnVHeads, ConvDim, l.DtBias, l.SsmA, gd, beta, Stream, count, g.RowsPerSeq);
                        if (UsesNinferGdnChunk(g))
                            RunNinferGdnScan(l, g, q0, qkv, gd, beta, _gdnO + (nint)((long)r0*ValueDim*4));
                        else
                            _ninfer.GdnScan(qkv, ConvDim, _m.GdnKHeads, _m.GdnVHeads, g.RowsPerSeq, g.Slots.Length,
                                gd, beta, TabState(l) + (nint)(8L*q0), _gdnO + (nint)((long)r0*ValueDim*4), g.Verify, Stream, count);
                    }
                    _ninfer.GdnOut(_gdnO, _proj, 16384, ConvDim, rows, _m.GdnVHeads, l.SsmNorm, _m.Eps, _gdnY, Stream);
                    residualState = ProjectNinferResidual(weights.Output, _gdnY, rows, isVerify, candidatePass);
                }
                else
                {
                    for (int gi = 0, r0 = 0, q0 = 0; gi < groups.Length; r0 += groups[gi].Rows, q0 += groups[gi].Slots.Length, gi++)
                    {
                        var g = groups[gi]; long kOff = KvLayerOffset(l), vOff = kOff + KvLayerBytes;
                        IntPtr tab = _tabBlocks + (nint)((long)q0*_maxBlocks*8);
                        nint qOff = (nint)((long)r0*QDim*4);
                        _ninferAttention.Prepare(_proj + (nint)((long)r0*AttnProjWidth*4), g.Rows, g.RowsPerSeq,
                            _m.NHead, _m.NKvHead, l.QNorm, l.KNorm, _m.Eps, _ropePosDev + (nint)(4L*r0), RowCap,
                            _m.SectT, _m.SectH, _m.SectW, _m.RopeBase, _cacheRowDev + (nint)(4L*r0), tab,
                            _maxBlocks, kOff, vOff, _q + qOff, _gate + qOff, Stream);
                        if (UsesAttn8(g, maxKeys[gi]) && Attn8FillsDevice(g) && TryEnsureAttn8Band(maxKeys[gi], out int cap))
                        {
                            _k.Attn8Band(_ninferKvFp8 ? Q35Kernels.Attn8Cache.Fp8 : Q35Kernels.Attn8Cache.Bf16, tab, kOff, vOff,
                                maxKeys[gi], cap, _m.NKvHead, _a8K, _a8Ks, _a8Vt, _a8Vs, Stream);
                            _k.Attn8Prefill(_q + qOff, _gate + qOff, _keyEndDev + (nint)(4L*r0), _a8K, _a8Ks, _a8Vt, _a8Vs, cap,
                                g.RowsPerSeq, _m.NHead, _m.NKvHead, 1.0f/MathF.Sqrt(_m.HeadDim), _attn + qOff, Stream);
                            continue;
                        }
                        _ninferAttention.Attention(_q + qOff, _gate + qOff, _keyEndDev + (nint)(4L*r0), tab,
                            _maxBlocks, kOff, vOff, g.RowsPerSeq, g.Slots.Length, _m.NHead, _m.NKvHead,
                            1.0f/MathF.Sqrt(_m.HeadDim), maxKeys[gi], g.Rows <= MaxVerifyRows ? _attnPartial : IntPtr.Zero, _attn + qOff, Stream,
                            isVerify: g.Verify, allRowsValid: g.Counts == null);
                    }
                    residualState = ProjectNinferResidual(weights.Output, _attn, rows, isVerify, candidatePass);
                }
                NormalizeNinferResidual(residualState, l.PostNorm, rows, candidatePass, postProjection: true, weights.GateUp);
                if (weights.GateUp.Descriptor.Format == CudaNinferProjectionFormat.Fp8Row)
                {
                    ProjectNinferFp8(weights.GateUp, _xn, _act, rows, Q35Kernels.Fp8Epilogue.SwiGlu, isVerify);
                    residualState = ProjectNinferResidual(weights.Down, _act, rows, isVerify, candidatePass);
                    continue;
                }
                if (TryNinferPrefillGateUpSwiGluQuant(weights.GateUp, weights.Down, rows, isVerify))
                {
                    residualState = ProjectNinferPrequantizedDown(weights.Down, rows);
                    continue;
                }
                if (rows == 1 && !_ninferFiResident.ContainsKey(weights.GateUp) && TryGemvNinferNvfp4(weights.GateUp, _xn, _act, swiGlu: true)) { }
                else if (!TryNinferGateUpSwiGlu(weights.GateUp, rows, fusedGateUp)
                    && !TryNinferPrefillGateUpSwiGlu(weights.GateUp, rows, isVerify))
                {
                    NinferProject(weights.GateUp, _xn, _proj, rows, 2, round: rows > 128 && rows != 1024, isVerify: isVerify);
                    _ninfer.SwiGlu(_proj, _act, _m.NFf, rows, Stream);
                }
                residualState = ProjectNinferResidual(weights.Down, _act, rows, isVerify, candidatePass);
            }
            _ninferFinalResidualState = residualState;
        }
    }
}
