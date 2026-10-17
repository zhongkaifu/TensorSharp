// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>Qwen-Image-2.1 architecture plug-in (any other Qwen-Image transformer is refused by <see cref="QwenImageModel"/>).</summary>
    internal static class QwenImageArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "qwen_image",
            DisplayName = "Qwen-Image",
            Aliases = new[] { "qwen_image", "qwen-image" },
            // --tp shards the diffusion transformer (heads and MLP columns per GPU);
            // the text encoder and VAE stay on the first GPU.
            Factory = c => new QwenImageModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup),
            SupportsDistributedTensorParallel = false,
            DetectFromTensors = LooksLikeVersion21,
            LoraPluginType = LoraCliFlags.QwenImagePluginType,
        };

        /// <summary>
        /// Community Qwen-Image-2.1 GGUFs can contain no metadata at all. Identify
        /// their single-stream transformer from its input, conditioning and block
        /// layout, without opening tensor data or relying on the file name.
        /// </summary>
        private static bool LooksLikeVersion21(GgufFile file)
        {
            // These are the two namespaces accepted by QwenImage21DiT. Use the
            // same precedence as its loader rather than mixing tensor namespaces.
            string prefix = file.Tensors.ContainsKey("img_in.weight") ? "" : "model.diffusion_model.";
            bool Has(string name, params ulong[] shape)
            {
                if (!file.Tensors.TryGetValue(prefix + name, out var tensor) || tensor.Shape.Length != shape.Length)
                    return false;
                for (int i = 0; i < shape.Length; i++)
                    if (tensor.Shape[i] != shape[i]) return false;
                return true;
            }

            // A text norm alone is insufficient: nearby diffusion architectures
            // share common projection names. Check Qwen-Image-2.1's dimensions
            // and gated image MLP as well. The loader validates remaining blocks.
            return Has("img_in.weight", 64, 4096)
                && Has("txt_in.text_norm.weight", 4096)
                && Has("txt_in.in_layer.weight", 4096, 4096)
                && Has("txt_in.out_layer.weight", 4096, 4096)
                && Has("time_text_embed.timestep_embedder.linear_1.weight", 256, 4096)
                && Has("time_text_embed.timestep_embedder.linear_2.weight", 4096, 4096)
                && Has("modulation.1.weight", 4096, 16384)
                && Has("norm_out.linear.weight", 4096, 4096)
                && Has("proj_out.weight", 4096, 64)
                && Has("transformer_blocks.0.attn.to_q.weight", 4096, 4096)
                && Has("transformer_blocks.0.attn.norm_q.weight", 128)
                && Has("transformer_blocks.0.img_mlp.out.weight", 12288, 4096)
                && (Has("transformer_blocks.0.img_mlp.gate_up.weight", 4096, 24576)
                    || (Has("transformer_blocks.0.img_mlp.gate_layer.weight", 4096, 12288)
                        && Has("transformer_blocks.0.img_mlp.proj.weight", 4096, 12288)));
        }
    }
}
