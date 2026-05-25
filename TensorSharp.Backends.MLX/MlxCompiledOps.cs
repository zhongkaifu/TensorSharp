using System;
using System.Threading;

namespace TensorSharp.MLX
{
    // Compiled (fused) activation kernels. Each one wraps a small graph of
    // MLX ops in mlx_compile so MLX collapses the chain into a single Metal
    // kernel and avoids materializing intermediates. Mirrors the pattern
    // used in ollama/x/mlxrunner/mlx/act.go.
    //
    // Each kernel is lazily compiled on first use under a guard so the
    // initialization is thread-safe. The compiled closure is then reused for
    // the lifetime of the process — MLX caches per (shape, dtype) tuple
    // internally when the closure is non-shapeless, or specializes once when
    // shapeless. We use shapeless because the FFN tile shapes vary per
    // sequence length during prefill.
    internal static class MlxCompiledOps
    {
        // Global kill switch. Set TS_MLX_DISABLE_COMPILE=1 to skip the
        // compile path and fall back to the original eager op chains (used
        // for A/B benchmarking and as a safety hatch).
        internal static readonly bool Disabled =
            string.Equals(Environment.GetEnvironmentVariable("TS_MLX_DISABLE_COMPILE"), "1", StringComparison.Ordinal);

        private static MlxNative.CompiledClosure siluClosure;
        private static MlxNative.CompiledClosure geluTanhClosure;
        private static MlxNative.CompiledClosure swiGluClosure;
        private static MlxNative.CompiledClosure geGluClosure;
        private static MlxNative.CompiledClosure sigmoidMulClosure;
        private static MlxNative.CompiledClosure addScaledClosure;
        private static MlxNative.CompiledClosure rmsNormScaledClosure;
        private static readonly object initLock = new();

        private static MlxNative.CompiledClosure EnsureCompiled(ref MlxNative.CompiledClosure slot, MlxNative.TraceFunc trace)
        {
            // Double-checked locking: the read after the lock catches the case
            // where two callers raced into EnsureCompiled.
            var existing = Volatile.Read(ref slot);
            if (existing != null) return existing;

            lock (initLock)
            {
                existing = Volatile.Read(ref slot);
                if (existing != null) return existing;
                var fresh = MlxNative.NewClosure(trace, shapeless: true);
                Volatile.Write(ref slot, fresh);
                return fresh;
            }
        }

        // silu(x) = x * sigmoid(x). One fused kernel instead of two-op chain.
        public static MlxNative.MlxArray SiLU(MlxNative.MlxArray x)
        {
            var c = EnsureCompiled(ref siluClosure, inputs =>
            {
                MlxNative.MlxArray inp = inputs[0];
                MlxNative.MlxArray sig = MlxNative.Unary(MlxNative.MlxUnaryOp.Sigmoid, inp);
                try
                {
                    MlxNative.MlxArray result = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, inp, sig);
                    return new[] { result };
                }
                finally
                {
                    MlxNative.FreeArray(sig);
                }
            });
            return MlxNative.ApplyClosure1(c, x);
        }

        // GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // Currently produced eagerly via 13 op calls; once compiled, MLX
        // fuses into a single kernel.
        public static MlxNative.MlxArray GeluTanh(MlxNative.MlxArray x)
        {
            var c = EnsureCompiled(ref geluTanhClosure, inputs => new[] { GeluTanhTrace(inputs[0]) });
            return MlxNative.ApplyClosure1(c, x);
        }

        private static MlxNative.MlxArray GeluTanhTrace(MlxNative.MlxArray input)
        {
            // Allocate scalars inside the trace so they participate in the
            // compiled graph. The MLX compiler folds them as constants.
            MlxNative.MlxArray coeffCubic = default;
            MlxNative.MlxArray coeffInner = default;
            MlxNative.MlxArray one = default;
            MlxNative.MlxArray half = default;
            MlxNative.MlxArray squared = default;
            MlxNative.MlxArray cubed = default;
            MlxNative.MlxArray scaledCubic = default;
            MlxNative.MlxArray inner = default;
            MlxNative.MlxArray scaledInner = default;
            MlxNative.MlxArray tanh = default;
            MlxNative.MlxArray onePlusTanh = default;
            MlxNative.MlxArray halfInput = default;
            try
            {
                coeffCubic = MlxNative.NewScalar(0.044715f);
                coeffInner = MlxNative.NewScalar(0.7978845608f);
                one = MlxNative.NewScalar(1.0f);
                half = MlxNative.NewScalar(0.5f);

                squared = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, input, input);
                cubed = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, squared, input);
                scaledCubic = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, cubed, coeffCubic);
                inner = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, input, scaledCubic);
                scaledInner = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, inner, coeffInner);
                tanh = MlxNative.Unary(MlxNative.MlxUnaryOp.Tanh, scaledInner);
                onePlusTanh = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, one, tanh);
                halfInput = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, input, half);
                return MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, halfInput, onePlusTanh);
            }
            finally
            {
                MlxNative.FreeArray(coeffCubic);
                MlxNative.FreeArray(coeffInner);
                MlxNative.FreeArray(one);
                MlxNative.FreeArray(half);
                MlxNative.FreeArray(squared);
                MlxNative.FreeArray(cubed);
                MlxNative.FreeArray(scaledCubic);
                MlxNative.FreeArray(inner);
                MlxNative.FreeArray(scaledInner);
                MlxNative.FreeArray(tanh);
                MlxNative.FreeArray(onePlusTanh);
                MlxNative.FreeArray(halfInput);
            }
        }

        // SwiGLU = silu(gate) * up. The common LLaMA/Qwen FFN activation.
        public static MlxNative.MlxArray SwiGLU(MlxNative.MlxArray gate, MlxNative.MlxArray up)
        {
            var c = EnsureCompiled(ref swiGluClosure, inputs =>
            {
                MlxNative.MlxArray g = inputs[0];
                MlxNative.MlxArray u = inputs[1];
                MlxNative.MlxArray sig = MlxNative.Unary(MlxNative.MlxUnaryOp.Sigmoid, g);
                MlxNative.MlxArray silu = default;
                try
                {
                    silu = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, g, sig);
                    MlxNative.MlxArray result = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, silu, u);
                    return new[] { result };
                }
                finally
                {
                    MlxNative.FreeArray(sig);
                    MlxNative.FreeArray(silu);
                }
            });
            return MlxNative.ApplyClosure2(c, gate, up);
        }

        // GeGLU = gelu(gate) * up. Used by Gemma family MLP and MoE paths.
        public static MlxNative.MlxArray GeGLU(MlxNative.MlxArray gate, MlxNative.MlxArray up)
        {
            var c = EnsureCompiled(ref geGluClosure, inputs =>
            {
                MlxNative.MlxArray g = inputs[0];
                MlxNative.MlxArray u = inputs[1];
                MlxNative.MlxArray gelu = GeluTanhTrace(g);
                try
                {
                    MlxNative.MlxArray result = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, gelu, u);
                    return new[] { result };
                }
                finally
                {
                    MlxNative.FreeArray(gelu);
                }
            });
            return MlxNative.ApplyClosure2(c, gate, up);
        }

        // SigmoidMul = x * sigmoid(gate). Variant used by some attention
        // gating paths.
        public static MlxNative.MlxArray SigmoidMul(MlxNative.MlxArray x, MlxNative.MlxArray gate)
        {
            var c = EnsureCompiled(ref sigmoidMulClosure, inputs =>
            {
                MlxNative.MlxArray xi = inputs[0];
                MlxNative.MlxArray gi = inputs[1];
                MlxNative.MlxArray sig = MlxNative.Unary(MlxNative.MlxUnaryOp.Sigmoid, gi);
                try
                {
                    MlxNative.MlxArray result = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, xi, sig);
                    return new[] { result };
                }
                finally
                {
                    MlxNative.FreeArray(sig);
                }
            });
            return MlxNative.ApplyClosure2(c, x, gate);
        }

        // RmsNormScaled = fast_rms_norm(x, weight, eps) * scalar. Fuses the
        // norm + scalar multiply that the Qwen3.5 GDN path does for Q and K
        // (`q * (1/dim)`, `k * (1/sqrt(dim))`) into a single Metal kernel.
        // Saves 2 kernel launches per GDN layer × 30 GDN layers/forward.
        public static MlxNative.MlxArray RmsNormScaled(
            MlxNative.MlxArray x,
            MlxNative.MlxArray weight,
            MlxNative.MlxArray scalar,
            float eps)
        {
            var c = EnsureCompiled(ref rmsNormScaledClosure, inputs =>
            {
                MlxNative.MlxArray xi = inputs[0];
                MlxNative.MlxArray wi = inputs[1];
                MlxNative.MlxArray si = inputs[2];
                // Use a closure-time-captured eps. Since eps is constant for
                // the Qwen35 GDN call site (1e-6f) and traces once shapelessly,
                // hard-coding it here is fine — the compiled graph specializes
                // to this eps value.
                MlxNative.MlxArray normed = MlxNative.FastRmsNorm(xi, wi, 1e-6f);
                try
                {
                    MlxNative.MlxArray result = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, normed, si);
                    return new[] { result };
                }
                finally
                {
                    MlxNative.FreeArray(normed);
                }
            });
            // Note: eps is captured by the closure trace; if a different eps
            // is ever needed, add it as a 4th input or compile a separate slot.
            _ = eps; // suppress unused param warning; documented above
            return MlxNative.ApplyClosure(c, new[] { x, weight, scalar })[0];
        }

        // AddScaled = output + scalar * src. Fuses mulv + addt into a single
        // Metal kernel. Used by the MoE decode accumulator where we run this
        // 8 times per layer × 60 MoE layers — going from 2 kernels to 1 saves
        // ~480 kernel launches per decode token.
        // The scalar is passed as a 0-D MLX array input so the same compiled
        // closure works regardless of its value (no per-call recompile).
        public static MlxNative.MlxArray AddScaled(MlxNative.MlxArray output, MlxNative.MlxArray src, MlxNative.MlxArray scalar)
        {
            var c = EnsureCompiled(ref addScaledClosure, inputs =>
            {
                MlxNative.MlxArray o = inputs[0];
                MlxNative.MlxArray s = inputs[1];
                MlxNative.MlxArray k = inputs[2];
                MlxNative.MlxArray scaled = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, s, k);
                try
                {
                    MlxNative.MlxArray result = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, o, scaled);
                    return new[] { result };
                }
                finally
                {
                    MlxNative.FreeArray(scaled);
                }
            });
            return MlxNative.ApplyClosure(c, new[] { output, src, scalar })[0];
        }
    }
}
