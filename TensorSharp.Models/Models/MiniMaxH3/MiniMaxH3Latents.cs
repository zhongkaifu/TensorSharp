// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>MiniMax-H3's video latent as the denoiser's patch tokens and back.</summary>
    public static class MiniMaxH3Latents
    {
        /// <summary>Patchify a video latent into transformer tokens.
        /// <paramref name="latent"/> is [T][H][W][C] with the channel axis contiguous;
        /// the result is [tokens][C * patchH * patchW] over the (t, h/2, w/2) grid.
        /// <para>Within a token the 96 values are CHANNEL-MAJOR, patch-minor:
        /// <c>c * (patchH*patchW) + ky * patchW + kx</c>. That is the reference's
        /// "patch_last" layout. Getting it the other way round leaves the RMS
        /// unchanged and silently scrambles every token, which the patch projection
        /// then turns into noise.</para></summary>
        public static float[] PatchifyVideo(float[] latent, int latentT, int latentH, int latentW, int channels)
        {
            int ph = MiniMaxH3Geometry.PatchH, pw = MiniMaxH3Geometry.PatchW;
            int gh = latentH / ph, gw = latentW / pw;
            int patchDim = channels * ph * pw;
            var outp = new float[(long)latentT * gh * gw * patchDim];
            for (int t = 0; t < latentT; t++)
                for (int y = 0; y < gh; y++)
                    for (int xx = 0; xx < gw; xx++)
                    {
                        long dst = (((long)t * gh + y) * gw + xx) * patchDim;
                        for (int ky = 0; ky < ph; ky++)
                            for (int kx = 0; kx < pw; kx++)
                            {
                                long src = (((long)t * latentH + (y * ph + ky)) * latentW
                                            + (xx * pw + kx)) * channels;
                                long spatial = (long)ky * pw + kx;
                                for (int c = 0; c < channels; c++)
                                    outp[dst + (long)c * ph * pw + spatial] = latent[src + c];
                            }
                    }
            return outp;
        }

        /// <summary>Inverse of <see cref="PatchifyVideo"/>.</summary>
        public static float[] UnpatchifyVideo(float[] tokens, int latentT, int latentH, int latentW, int channels)
        {
            int ph = MiniMaxH3Geometry.PatchH, pw = MiniMaxH3Geometry.PatchW;
            int gh = latentH / ph, gw = latentW / pw;
            int patchDim = channels * ph * pw;
            var outp = new float[(long)latentT * latentH * latentW * channels];
            for (int t = 0; t < latentT; t++)
                for (int y = 0; y < gh; y++)
                    for (int xx = 0; xx < gw; xx++)
                    {
                        long src = (((long)t * gh + y) * gw + xx) * patchDim;
                        for (int ky = 0; ky < ph; ky++)
                            for (int kx = 0; kx < pw; kx++)
                            {
                                long dst = (((long)t * latentH + (y * ph + ky)) * latentW
                                            + (xx * pw + kx)) * channels;
                                long spatial = (long)ky * pw + kx;
                                for (int c = 0; c < channels; c++)
                                    outp[dst + c] = tokens[src + (long)c * ph * pw + spatial];
                            }
                    }
            return outp;
        }
    }
}
