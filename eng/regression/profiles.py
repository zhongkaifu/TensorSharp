"""Explicit residency profiles; no fit estimates inferred from MoE file size."""
from __future__ import annotations

import math

CPU_MOE_BACKENDS = ("ggml_cuda", "ggml_cpu")
CPU_MOE_FAMILIES = ("qwen35", "qwen4exp")
CPU_MOE_SOURCES = [
    "TensorSharp.Models/MoeCpuOffloadConfig.cs",
    "TensorSharp.Models/ModelBase.WeightLoading.cs",
    "TensorSharp.Models/SharedDecoder/StandardDecoderModel.ExpertPlacement.cs",
    "TensorSharp.Models/Models/Qwen4Exp/Qwen4ExpModel.ExpertPlacement.cs",
]


def residency_profile(cpu_moe=False, ram_required_gib=None, vram_required_gib=None,
                      preset=None, backend=None):
    """Describe requested placement separately from runtime confirmation.

    User requirements include workload-dependent resident weights, staging, KV,
    and graph memory. They do not shrink the on-disk/download weight footprint.
    """
    profile = {"name": "cpu_moe" if cpu_moe else "conservative_full_resident",
               "requested": bool(cpu_moe), "supported": True, "environment": {},
               "runtime_confirmation_required": bool(cpu_moe)}
    if not cpu_moe:
        profile["reason"] = "Conservative full-weight planning; no offload policy is requested by the harness"
        return profile
    profile.update(
        environment={"TS_CPU_MOE": "1"},
        estimate_source="explicit_user_requirements",
        ram_required_gib=ram_required_gib, vram_required_gib=vram_required_gib,
        supported_backends=list(CPU_MOE_BACKENDS),
        supported_families=list(CPU_MOE_FAMILIES), required_layout="moe",
        scope="single-device GGML; all routed experts on host; no tensor parallel or layer split",
        source_evidence=CPU_MOE_SOURCES,
        reason="Explicit host routed-expert policy; runtime configuration and backend identity must still be confirmed",
        limitation="Caller-supplied peak requirements are not measured guarantees; mmap resident pages and KV/graph demand depend on the workload")
    for name, value in (("ram", ram_required_gib), ("vram", vram_required_gib)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            profile.update(supported=False, reason=f"--cpu-moe requires an explicit positive finite --{name}-required-gib")
            return profile
    if preset is not None:
        profile.update(family=preset.get("family"), layout=preset.get("layout"))
        if preset.get("family") not in CPU_MOE_FAMILIES or preset.get("layout") != "moe":
            profile.update(supported=False, reason="--cpu-moe is source-verified only for qwen35/moe and qwen4exp/moe presets")
            return profile
    if backend is not None:
        profile["backend"] = backend
        if backend not in CPU_MOE_BACKENDS:
            profile.update(supported=False, reason="--cpu-moe profile supports only ggml_cpu and ggml_cuda; requested backend is unchanged")
        elif backend == "ggml_cpu":
            profile["placement_effect"] = "Experts already execute on host; this profile does not claim VRAM savings on ggml_cpu"
            profile["vram_requirement_applied"] = False
        else:
            profile["placement_effect"] = "All routed experts execute on host; attention/router/shared experts remain on requested GGML CUDA backend"
            profile["vram_requirement_applied"] = True
    return profile


def annotate_resource_plan(plan, profile):
    """Keep the normal budget checks while accurately labeling explicit residency."""
    plan = {**plan, "offload_policy": profile}
    if profile["requested"]:
        plan["offload"] = "explicit CPU MoE; caller-supplied resident RAM/VRAM requirements"
        plan["warnings"] = [*plan.get("warnings", []), profile["limitation"]]
        # A full-weight GPU fraction would misrepresent the explicit policy.
        plan["estimates"] = {**plan.get("estimates", {}), "gpu_weight_fraction": None,
                             "requirements_source": "explicit_user_requirements"}
        if plan.get("eligible"):
            plan["reason"] = "Within explicit CPU MoE requirements and current resource budgets; runtime attestation still required"
    return plan
