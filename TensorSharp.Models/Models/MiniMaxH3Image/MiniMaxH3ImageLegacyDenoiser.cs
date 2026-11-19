// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// A test seam: the legacy MiniMax-H3-Image denoiser (MiniMaxH3DiT.ForwardImage, one native graph a step through
// TSGgml_MiniMaxH3DitForward) behind the runtime's IDenoiser, so the runtime's denoiser can be compared with it step for
// step on the same inputs. Not a production path: production runs MiniMaxH3ImageDiffusionModel on the runtime.
using System;
using System.Threading;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3Image
{
    internal sealed class MiniMaxH3ImageLegacyDenoiser : IDenoiser
    {
        private readonly bool _owns;

        public MiniMaxH3ImageLegacyDenoiser(string path, BackendType backend)
        {
            Dit = new MiniMaxH3DiT(path, backend, null);
            _owns = true;
        }

        /// <summary>Over an existing legacy DiT (the caller keeps it).</summary>
        public MiniMaxH3ImageLegacyDenoiser(MiniMaxH3DiT dit) => Dit = dit ?? throw new ArgumentNullException(nameof(dit));

        public MiniMaxH3DiT Dit { get; }

        public DenoiserCaps Caps { get; } = new();

        public long WeightBytes => 0;

        public long WorkingBytes(DiffusionWorkload workload) => 0;

        public StepOutput Step(DenoiseStepInputs inputs, AdapterSet adapters, CancellationToken cancellationToken)
        {
            ArgumentNullException.ThrowIfNull(inputs);
            if (adapters is { IsEmpty: false }) throw new NotSupportedException("the legacy MiniMax-H3-Image denoiser takes no adapters.");
            var layout = (MiniMaxH3PackedLayout)inputs.Layout.FamilyLayout;
            float[] references = null;
            int referenceRows = 0;
            if (inputs.References is { Count: > 0 })
            {
                long values = 0;
                foreach (LatentRef r in inputs.References) values += r.Rows.LongLength;
                references = new float[values];
                long at = 0;
                foreach (LatentRef r in inputs.References)
                {
                    Array.Copy(r.Rows, 0, references, at, r.Rows.LongLength);
                    at += r.Rows.LongLength;
                }
                referenceRows = checked((int)(values / Dit.Config.VideoPatchDim));
            }
            int imageRows = checked((int)(inputs.Latent.LongLength / Dit.Config.VideoPatchDim));
            return new StepOutput(Dit.ForwardImage(inputs.Latent, imageRows, inputs.Conditioning.Hidden, inputs.Conditioning.Rows, layout,
                inputs.Recipe.Grid, inputs.Step.HeadStart, inputs.Step.HeadStop, references, referenceRows));
        }

        public void ReleaseDevice() => Dit.ReleaseDeviceResidency();

        public void ReleaseWorkspace() { }

        public void Dispose()
        {
            if (_owns) Dit.Dispose();
        }
    }
}
