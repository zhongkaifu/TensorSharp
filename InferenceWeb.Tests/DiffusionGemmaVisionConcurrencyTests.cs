// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Image spans must belong to a SEQUENCE, not to the model.
//
// They used to be a model-global list, which meant two image requests in flight would overwrite
// each other -- the scheduler worked around it by denoising every image turn alone, serializing
// all multimodal traffic. vLLM and SGLang both hang multimodal features off the request
// (MultiModalFeatureSpec on Request, MultimodalInputs on Req) and never off the runner; these
// tests pin that same property here.
//
// They exercise the state machine directly rather than through a server, so they need no weights
// and run in milliseconds.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests
{
    public class DiffusionGemmaVisionConcurrencyTests
    {
        private readonly ITestOutputHelper _output;
        public DiffusionGemmaVisionConcurrencyTests(ITestOutputHelper output) { _output = output; }

        private static Tensor Span(IAllocator allocator, int rows, int hidden, float fill)
        {
            var t = new Tensor(allocator, DType.Float32, rows, hidden);
            var data = new float[(long)rows * hidden];
            for (long i = 0; i < data.LongLength; i++) data[i] = fill;
            t.SetElementsAsFloat(data);
            return t;
        }

        /// <summary>
        /// Two sequences hold their own spans simultaneously, and neither sees the other's. This is
        /// the property that lets the scheduler batch image turns instead of running them alone.
        /// </summary>
        [Fact]
        public void TwoSequences_HoldIndependentSpans()
        {
            var allocator = new CpuAllocator(BlasEnum.DotNet);
            const int hidden = 8;

            var a = NewSeq(2);
            var b = NewSeq(2);

            AttachSpan(a, Span(allocator, 4, hidden, 1f), position: 3);
            AttachSpan(b, Span(allocator, 6, hidden, 2f), position: 11);

            Assert.Equal(new[] { (3, 4) }, SpansOf(a));
            Assert.Equal(new[] { (11, 6) }, SpansOf(b));

            // Re-attaching to one must not disturb the other.
            AttachSpan(a, Span(allocator, 5, hidden, 3f), position: 40);
            Assert.Equal(new[] { (3, 4), (40, 5) }, SpansOf(a));
            Assert.Equal(new[] { (11, 6) }, SpansOf(b));

            _output.WriteLine($"a={string.Join(",", SpansOf(a))}  b={string.Join(",", SpansOf(b))}");
        }

        /// <summary>
        /// Freeing a sequence drops its spans, so a finished request cannot leak its picture into
        /// whatever the scheduler admits next.
        /// </summary>
        [Fact]
        public void DisposingASequence_DropsOnlyItsOwnSpans()
        {
            var allocator = new CpuAllocator(BlasEnum.DotNet);
            const int hidden = 8;

            var a = NewSeq(2);
            var b = NewSeq(2);
            AttachSpan(a, Span(allocator, 4, hidden, 1f), position: 0);
            AttachSpan(b, Span(allocator, 4, hidden, 2f), position: 0);

            DisposeVision(a);

            Assert.Empty(SpansOf(a));
            Assert.Equal(new[] { (0, 4) }, SpansOf(b));
        }

        /// <summary>
        /// Overlapping spans on the SAME sequence are a caller bug and must throw, because silently
        /// accepting them writes one image's rows over another's.
        /// </summary>
        [Fact]
        public void OverlappingSpansOnOneSequence_Throw()
        {
            var allocator = new CpuAllocator(BlasEnum.DotNet);
            const int hidden = 8;
            var a = NewSeq(2);

            AttachSpan(a, Span(allocator, 10, hidden, 1f), position: 5);
            Assert.ThrowsAny<ArgumentException>(
                () => AttachSpan(a, Span(allocator, 4, hidden, 2f), position: 12));
        }

        // ---- helpers: the model's own per-sequence span sets (VisionSet), without a 26B model --------

        private static DiffusionSeqState NewSeq(int unused) => new DiffusionSeqState("test-seq-" + Guid.NewGuid().ToString("N"));

        private static void AttachSpan(DiffusionSeqState seq, Tensor embeddings, int position)
            => (seq.Vision ??= new VisionSet()).Set(embeddings, position);

        private static void DisposeVision(DiffusionSeqState seq) => seq.DisposeVision();

        private static (int Start, int Length)[] SpansOf(DiffusionSeqState seq)
            => seq.Vision == null ? Array.Empty<(int, int)>()
                : seq.Vision.Spans.Select(x => (x.Position, (int)x.Rows.Sizes[0])).OrderBy(x => x.Position).ToArray();
    }
}
