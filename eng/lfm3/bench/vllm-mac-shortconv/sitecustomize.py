# Opt-in workaround for vLLM's CPU backend on macOS (vllm-cpu-mac.sh's `vllm-*fix` arms; put this directory on
# PYTHONPATH). Not a TensorSharp component and not loaded by anything else.
#
# vLLM 0.34.0's short convolution (LFM2/LFM3 `ShortConv.forward_native`) runs, on Arm, its decode step through
# `causal_conv1d_update_torch` and every prompt through `causal_conv1d_fn_cpu`, both a depthwise `F.conv1d`
# (groups = 2048) meant for oneDNN/ACL. PyTorch's macOS wheels have no oneDNN (`torch.backends.mkldnn.is_available()`
# is False), so each call falls back to `slow_conv2d` once per group: 2048 unfold + OpenMP fork/join rounds a layer,
# ~1.5 s for the 12 conv layers of one LFM3-1.7B decode step (0.65 tok/s). This replaces the two functions with
# equivalent ones: decode calls vLLM's own `causal_conv1d_update_cpu_vec` kernel (what x86 uses; float32
# accumulation), prompts sum the width-4 taps as shifted multiply-adds in float32. The numerics are the reference
# convolution's up to float32 summation order.
import importlib.abc
import importlib.machinery
import sys

_TARGET = "vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d"


def _patch(module):
    import torch

    def causal_conv1d_update_torch(x, conv_state, weight, bias=None, activation=None):
        # x (batch, dim, seqlen); conv_state (batch, dim, width - 1), updated in place, as the original.
        return torch.ops._C.causal_conv1d_update_cpu_vec(x, conv_state, weight, bias, activation, None, None, -1)

    def taps(conv_input, weight, bias, length):
        # conv_input (1, dim, state_len + length); weight (dim, 1, width) -> (1, dim, length), float32 sums
        width = weight.shape[-1]
        start = conv_input.shape[-1] - length - (width - 1)
        xin = conv_input.float()
        w = weight[:, 0, :].float()
        acc = xin[..., start:start + length] * w[:, 0:1]
        for k in range(1, width):
            acc = acc + xin[..., start + k:start + k + length] * w[:, k:k + 1]
        if bias is not None:
            acc = acc + bias.float()[:, None]
        return acc.to(conv_input.dtype)

    def causal_conv1d_fn_cpu(x, weight, bias, conv_states, query_start_loc, cache_indices=None,
                             has_initial_state=None, activation="silu", pad_slot_id=module.PAD_SLOT_ID, **kwargs):
        # The original's control flow, with F.conv1d replaced by `taps`.
        if isinstance(activation, bool):
            activation = "silu" if activation else None
        original_x_dtype = x.dtype
        x = x.to(conv_states.dtype)
        out = torch.empty_like(x)
        state_len = weight.shape[1] - 1
        bounds = [(int(query_start_loc[i].item()), int(query_start_loc[i + 1].item()))
                  for i in range(query_start_loc.shape[0] - 1)]
        weight = weight.unsqueeze(1)
        for seq_idx, (bos, eos) in enumerate(bounds):
            if bos == eos:
                continue
            slot = int(cache_indices[seq_idx].item()) if cache_indices is not None else seq_idx
            if slot == pad_slot_id:
                continue
            seq_x = x[:, bos:eos].unsqueeze(0)
            if has_initial_state is not None and bool(has_initial_state[seq_idx].item()):
                initial_state = conv_states[slot, :, :state_len].unsqueeze(0)
            else:
                initial_state = torch.zeros(1, weight.shape[0], state_len, device=seq_x.device, dtype=seq_x.dtype)
            conv_input = torch.cat([initial_state, seq_x], dim=-1).to(weight.dtype)
            seq_out = taps(conv_input, weight, bias, seq_x.shape[-1]).to(dtype=x.dtype)
            if activation in ("silu", "swish"):
                seq_out = torch.nn.functional.silu(seq_out)
            out[:, bos:eos] = seq_out.squeeze(0)
            conv_states[slot, :, :state_len].copy_(conv_input[..., -state_len:].squeeze(0))
        return out.to(original_x_dtype)

    module.causal_conv1d_update_torch = causal_conv1d_update_torch
    module.causal_conv1d_fn_cpu = causal_conv1d_fn_cpu
    print("[vllm-mac-shortconv] short-conv CPU paths replaced (decode: causal_conv1d_update_cpu_vec, "
          "prompt: float32 tap sums)", file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


sys.meta_path.insert(0, _Finder())
