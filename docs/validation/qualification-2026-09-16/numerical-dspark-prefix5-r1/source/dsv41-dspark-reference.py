#!/usr/bin/env python3
"""Independent CPU reference for V4.1 DSpark GGUF draft graphs.

Equations follow the pinned official inference/model.py DSparkBlock and
DSparkAttention. This is a numerical test oracle, not a serving implementation
or evidence that a speculative acceptance/rollback loop passed.
"""
import importlib.util
import math
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location("dsv41_reference", Path(__file__).with_name("dsv41-reference.py"))
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


class TargetWithDraftFeatures(reference.Reference):
    """Capture the mean stream state after Engram and BEFORE each target block."""
    def __init__(self, *args, target_layers, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_layers = tuple(target_layers)
        self.features = {}

    def eng_inject(self, layer, h, tokens):
        h = super().eng_inject(layer, h, tokens)
        if layer in self.target_layers:
            self.features[layer] = h.mean(dim=1).clone()
        return h

    def draft_features(self):
        return torch.cat([self.features[layer] for layer in self.target_layers], dim=-1)


class DSparkReference:
    def __init__(self, draft_weights, target_weights, config, *, block_size, stages,
                 expert_count, expert_used_count, noise_token, markov_rank):
        self.w, self.target = draft_weights, target_weights
        self.c = config.get("text_config", config)
        self.block_size, self.stages = block_size, stages
        self.experts, self.used = expert_count, expert_used_count
        self.noise, self.rank = noise_token, markov_rank
        self.position = 0
        self.keys = [None] * stages
        # The cache quantizer/rotary equations are independent PyTorch code.
        self.math = reference.Reference(target_weights, config, {}, "model")

    def norm(self, x, name):
        return reference.rms(x, self.c["rms_norm_eps"], self.w.weight(name).flatten())

    def commit_features(self, features):
        """Append committed target rows, independently of any speculative draft."""
        count = len(features)
        main = self.norm(self.w.mm("mtp.0.main_proj.weight", features), "mtp.0.main_norm.weight")
        self.committed_features = features.clone()
        self.committed_main = main.clone()
        self.committed_prequant = []
        positions = list(range(self.position, self.position + count))
        for stage in range(self.stages):
            prefix = f"mtp.{stage}."
            kv = self.norm(self.w.mm(prefix + "attn_kv.weight", main), prefix + "attn_kv_a_norm.weight")
            kv = self.math.rope(kv, positions, 0)
            self.committed_prequant.append(kv.clone())
            kv = self.math.cache(kv)
            self.keys[stage] = kv if self.keys[stage] is None else torch.cat([self.keys[stage], kv])
        self.position += count

    def rewind(self, position):
        if not 0 <= position <= self.position:
            raise ValueError("Invalid reference rewind")
        self.keys = [None if rows is None else rows[:position].clone() for rows in self.keys]
        self.position = position

    def attention(self, stage, x):
        c, prefix = self.c, f"mtp.{stage}."
        width, heads = c["head_dim"], c["num_attention_heads"]
        positions = list(range(self.position, self.position + self.block_size))
        qr = self.norm(self.w.mm(prefix + "attn_q_a.weight", x), prefix + "attn_q_a_norm.weight")
        # V4.1 does not RMS-normalize the projected query heads.
        q = self.w.mm(prefix + "attn_q_b.weight", qr).reshape(self.block_size, heads, width)
        q = self.math.rope(q, positions, 0)
        kv = self.norm(self.w.mm(prefix + "attn_kv.weight", x), prefix + "attn_kv_a_norm.weight")
        kv = self.math.cache(self.math.rope(kv, positions, 0))
        committed = self.keys[stage][-c["sliding_window"]:]
        keys = torch.cat([committed, kv])
        scores = torch.einsum("thd,kd->thk", q, keys) / math.sqrt(width)
        sinks = self.w.weight(prefix + "attn_sinks.weight").flatten().view(1, heads, 1)
        weights = torch.cat([scores, sinks.expand(self.block_size, -1, -1)], -1).softmax(-1)[..., :-1]
        out = torch.einsum("thk,kd->thd", weights, keys)
        out = self.math.rope(out, positions, 0, inverse=True)
        groups, rank = c["o_groups"], c["o_lora_rank"]
        grouped = out.reshape(self.block_size, groups, -1)
        wa = self.w.weight(prefix + "attn_output_a.weight").reshape(groups, rank, -1)
        projected = torch.einsum("tgd,grd->tgr", grouped, wa).flatten(1)
        return self.w.mm(prefix + "attn_output_b.weight", projected)

    def moe(self, stage, x):
        c, prefix = self.c, f"mtp.{stage}."
        scores = torch.nn.functional.softplus(self.w.mm(prefix + "ffn_gate_inp.weight", x)).sqrt()
        if scores.shape[1] != self.experts:
            raise ValueError("Draft router width differs from draft expert count")
        # Draft tokens are text even when the target prompt contained images.
        bias = self.w.weight(prefix + "exp_probs_b.bias").flatten()
        selected = torch.argsort(scores + bias, dim=-1, descending=True, stable=True)[:, :self.used]
        weights = scores.gather(1, selected)
        if c["norm_topk_prob"] and self.used > 1:
            weights /= weights.sum(-1, keepdim=True)
        weights *= c["routed_scaling_factor"]

        def ffn(value, suffix, expert=None):
            rank = c["moe_intermediate_size"]
            gate = self.w.mm(prefix + f"ffn_gate_{suffix}.weight", value, expert, rank)
            up = self.w.mm(prefix + f"ffn_up_{suffix}.weight", value, expert, rank)
            limit = c.get("swiglu_limit", 0)
            if limit > 0:
                gate = gate.clamp(max=limit)
                up = up.clamp(-limit, limit)
            h = torch.nn.functional.silu(gate) * up
            return self.w.mm(prefix + f"ffn_down_{suffix}.weight", h, expert, c["hidden_size"])

        result = ffn(x, "shexp")
        for token in range(len(x)):
            for slot in range(self.used):
                expert = int(selected[token, slot])
                result[token] += ffn(x[token:token+1], "exps", expert)[0] * weights[token, slot]
        return result

    def draft(self, anchor):
        if self.position <= 0:
            raise ValueError("Drafting requires committed target features")
        c = self.c
        tokens = [anchor] + [self.noise] * (self.block_size - 1)
        h = self.target.rows("token_embd.weight", tokens).unsqueeze(1).repeat(1, c["hc_mult"], 1)
        pre = torch.zeros((self.block_size, c["hc_mult"]), device=h.device)
        pre[:, 0] = 1  # Official initial one-hot mix; embedding streams are equal.
        trace = {}
        for stage in range(self.stages):
            prefix = f"mtp.{stage}."
            for kind, operation in (("attn", self.attention), ("ffn", self.moe)):
                next_pre, post, comb = reference.hc_mixes(h,
                    lambda value: self.w.mm(prefix + f"hc_{kind}_fn.weight", value),
                    self.w.weight(prefix + f"hc_{kind}_scale.weight").flatten(),
                    self.w.weight(prefix + f"hc_{kind}_base.weight").flatten(),
                    c["hc_sinkhorn_iters"], c["hc_eps"], c["rms_norm_eps"])
                x = self.norm((h * pre.unsqueeze(-1)).sum(1), prefix + kind + "_norm.weight")
                trace[f"stage{stage}_{kind}_input"] = x.clone()
                y = operation(stage, x)
                h = post.unsqueeze(-1) * y.unsqueeze(1) + torch.einsum("tij,tid->tjd", comb, h)
                pre = next_pre
            trace[f"stage{stage}_hidden"] = h.clone()
        hidden = (h * pre.unsqueeze(-1)).sum(1)
        prefix = f"mtp.{self.stages-1}."
        logits = self.target.mm("output.weight", self.norm(hidden, prefix + "norm.weight"))
        output, confidence_logits = [], []
        previous = anchor
        for index in range(self.block_size):
            markov = self.w.rows(prefix + "markov_head.markov_w1.weight", [previous])
            logits[index] += self.w.mm(prefix + "markov_head.markov_w2.weight", markov)[0]
            previous = int(logits[index].argmax())
            output.append(previous)
            combined = torch.cat([hidden[index:index+1], markov], dim=-1)
            confidence_logits.append(self.w.mm(prefix + "confidence_head.proj.weight", combined).flatten()[0])
        trace.update(hidden=hidden, logits=logits, confidence_logits=torch.stack(confidence_logits))
        # TensorSharp's block speculator consumes probabilities; the official
        # forward_head returns raw confidence logits. Keep both for comparison.
        return output, torch.sigmoid(trace["confidence_logits"]), trace
