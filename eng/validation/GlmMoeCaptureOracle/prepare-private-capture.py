#!/usr/bin/env python3
"""Add bounded TEST_HOOKS captures to a NEW TensorSharp-owned source copy.

Never configures/builds, modifies ggml, touches original sources or overwrites outputs.
Supports pinned pre-bulk and bulk graph layouts; unfamiliar anchors fail closed.
"""
import argparse, datetime, hashlib, json, pathlib, re, shutil, subprocess
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=pathlib.Path, required=True, help='Immutable build snapshot containing TensorSharp.GGML.Native/')
parser.add_argument('--manifest', type=pathlib.Path, required=True, help='Successful native build-summary.json with all source_snapshot hashes')
parser.add_argument('--native-reference', type=pathlib.Path, required=True, help='Immutable production DLL/library pinned by that build')
parser.add_argument('--upstream', type=pathlib.Path, required=True, help='Unmodified ggml checkout, inspected read-only')
parser.add_argument('--output', type=pathlib.Path, required=True, help='Fresh private candidate directory, never the workspace source/build directory')
args = parser.parse_args()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source_root=args.source_root.resolve(); upstream=args.upstream.resolve(); out=args.output.resolve()
if out.exists() or out.is_relative_to(source_root) or out.is_relative_to(upstream):
    raise ValueError('Output must be new and outside original source/upstream trees.')
original=json.loads(args.manifest.read_text())
if not original.get('passed') or not original.get('source_snapshot'):
    raise ValueError('A successfully frozen complete native build manifest is required.')
identity=original.get('production_artifacts',{}).get(args.native_reference.name)
if not identity or sha(args.native_reference)!=identity:
    raise ValueError('Native reference does not match the original production artifact identity.')
def upstream_state():
    return {'revision':subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip(),
        'status':subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True)}
before_upstream=upstream_state()
if before_upstream.get('status') or before_upstream!=original.get('upstream_after'):
    raise ValueError('Upstream must remain clean and match the pinned build revision.')
for name, expected in original['source_snapshot'].items():
    if not name.replace('\\','/').startswith('TensorSharp.GGML.Native/'):
        raise ValueError('Only TensorSharp-owned native inputs may be copied.')
    path=(source_root/name).resolve()
    if not path.is_relative_to(source_root) or sha(path)!=expected:
        raise ValueError('Original input identity/path changed: '+name)
out.mkdir(parents=True, exist_ok=False)
for name in original['source_snapshot']:
    destination=out/'source'/name; destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source_root/name,destination)
native=out/'source/TensorSharp.GGML.Native'
header = r'''// Private diagnostic candidate. Not a production ABI or performance path.
#pragma once
#if defined(TSG_GGML_TEST_HOOKS)
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
namespace tsg_moe_diag {
struct Context {
    std::uintptr_t model = 0, graph = 0;
    std::uint64_t evaluation = 0;
    int tokens = 0, layer = -1, row = 0, rows = 0;
    double clamp = 0;
    const char* mode = "ordinary";
};
inline Context& context() { static thread_local Context value; return value; }
inline const char* root() { return std::getenv("TS_GLM_MOE_DIAG_ROOT"); }
inline std::size_t limit(const char* key, std::size_t fallback) {
    const char* p = std::getenv(key);
    if (!p || !*p) return fallback;
    for (const char* s = p; *s; ++s) if (*s < '0' || *s > '9') throw std::runtime_error("Invalid private diagnostic limit");
    errno = 0; char* end = nullptr;
    auto value = std::strtoull(p, &end, 10);
    if (errno || !end || *end || value > std::numeric_limits<std::size_t>::max()) throw std::runtime_error("Private diagnostic limit overflow");
    return static_cast<std::size_t>(value);
}
inline bool enabled() { const char* p = root(); return p && *p; }
inline bool layer_enabled(int layer) { return enabled() && layer == static_cast<int>(limit("TS_GLM_MOE_DIAG_LAYER", 1)); }
inline bool capture_enabled() {
    return enabled() && context().tokens >= static_cast<int>(limit("TS_GLM_MOE_DIAG_MIN_ROWS", 16));
}
inline std::string escape(const char* text) {
    std::string out;
    for (const unsigned char c : std::string(text ? text : "")) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c < 32) { char s[7]; std::snprintf(s, sizeof(s), "\\u%04x", c); out += s; }
        else out += c;
    }
    return out;
}
struct State { std::mutex mutex; std::size_t bytes = 0; std::uint64_t next = 0; };
inline State& state() { static State value; return value; }
struct Scope {
    Context previous;
    Scope(int layer, int row, int rows, double clamp) : previous(context()) {
        context().layer = layer; context().row = row; context().rows = rows; context().clamp = clamp;
    }
    ~Scope() { context() = previous; }
};
inline void capture(const char* name, const ggml_tensor* tensor) {
    if (!capture_enabled() || !tensor || !tensor->buffer) return;
    const auto bytes = ggml_nbytes(tensor);
    auto& s = state(); std::lock_guard<std::mutex> lock(s.mutex);
    const auto maximum = limit("TS_GLM_MOE_DIAG_MAX_BYTES", 64u << 20);
    if (bytes > (8u << 20) || bytes > maximum || s.bytes > maximum - bytes)
        throw std::runtime_error("Private diagnostic bounded capture refused");
    const auto id = ++s.next;
    char file[64]; std::snprintf(file, sizeof(file), "%06" PRIu64 ".bin", id);
    const std::string path = std::string(root()) + "/" + file;
    FILE* data = std::fopen(path.c_str(), "wb");
    if (!data) throw std::runtime_error("Private diagnostic capture directory unavailable");
    std::vector<unsigned char> buffer(std::min<std::size_t>(bytes, 1u << 20));
    for (std::size_t offset = 0; offset < bytes; offset += buffer.size()) {
        const auto count = std::min(buffer.size(), bytes - offset);
        ggml_backend_tensor_get(tensor, buffer.data(), offset, count);
        if (std::fwrite(buffer.data(), 1, count, data) != count) { std::fclose(data); throw std::runtime_error("Private diagnostic payload write failed"); }
    }
    if (std::fclose(data)) throw std::runtime_error("Private diagnostic payload close failed");
    FILE* index = std::fopen((std::string(root()) + "/index.jsonl").c_str(), "ab");
    if (!index) throw std::runtime_error("Private diagnostic index unavailable");
    const auto c = context();
    const int written = std::fprintf(index,
        "{\"id\":%" PRIu64 ",\"file\":\"%s\",\"name\":\"%s\",\"mode\":\"%s\",\"model\":%" PRIuPTR ",\"graph\":%" PRIuPTR ",\"evaluation\":%" PRIu64
        ",\"tokens\":%d,\"layer\":%d,\"row\":%d,\"rows\":%d,\"clamp\":%.17g,\"type\":%d,\"type_name\":\"%s\",\"op\":%d,\"ne\":[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
        "],\"nb\":[%zu,%zu,%zu,%zu],\"nbytes\":%zu,\"view_offs\":%zu,\"contiguous\":%s,\"diagnostic_output_lifetime\":true}\n",
        id, file, escape(name).c_str(), c.mode, c.model, c.graph, c.evaluation,
        c.tokens, c.layer, c.row, c.rows, c.clamp, static_cast<int>(tensor->type), ggml_type_name(tensor->type), static_cast<int>(tensor->op),
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
        tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3], bytes, tensor->view_offs, ggml_is_contiguous(tensor) ? "true" : "false");
    const int closed = std::fclose(index);
    if (written < 0 || closed) throw std::runtime_error("Private diagnostic index write failed");
    s.bytes += bytes;
}
} // namespace tsg_moe_diag
#endif
'''
(native / 'glm_moe_private_diagnostic.h').write_text(header, newline='\n')

def replace_once(text, old, new):
    if text.count(old) != 1: raise ValueError((old[:80], text.count(old)))
    return text.replace(old, new)

p = native / 'ggml_ops_glm_dsa.cpp'
t = p.read_text()
t = replace_once(t, '#include "ggml_ops_internal.h"', '#include "ggml_ops_internal.h"\n#include "glm_moe_private_diagnostic.h"')
t = replace_once(t, '    std::vector<trace_entry> traces;', '''    std::vector<trace_entry> traces;
#if defined(TSG_GGML_TEST_HOOKS)
    struct diagnostic_tensor { std::string name; int layer; ggml_tensor* tensor; };
    std::vector<diagnostic_tensor> diagnostic_tensors;
    std::uintptr_t diagnostic_model = 0;
    std::uint64_t diagnostic_evaluation = 0;
    double diagnostic_clamp = 0;
#endif''')
t = replace_once(t, '    void trace(const char * tag, int il, int rank, ggml_tensor * t)', '''    void diagnostic(const char* tag, int il, int rank, ggml_tensor* t)
    {
#if defined(TSG_GGML_TEST_HOOKS)
        if (!t || rank != 0 || !tsg_moe_diag::layer_enabled(il)) return;
        res.diagnostic_model = reinterpret_cast<std::uintptr_t>(&m);
        res.diagnostic_clamp = hp.swiglu_clamp;
        ggml_set_output(t);
        res.diagnostic_tensors.push_back({tag, il, t});
        ggml_build_forward_expand(gf, t);
#else
        (void)tag; (void)il; (void)rank; (void)t;
#endif
    }

    void trace(const char * tag, int il, int rank, ggml_tensor * t)''')
t = replace_once(t, '        ggml_build_forward_expand(gf, weights);\n\n        if (m.selected_expert_seams', '''        ggml_build_forward_expand(gf, weights);
        diagnostic("input", il, rank, cur);
        diagnostic("router_logits", il, rank, logits);
        diagnostic("router_probs", il, rank, probs);
        diagnostic("router_selection", il, rank, selection);
        diagnostic("router_ids", il, rank, selected);
        diagnostic("route_weights", il, rank, weights);
        diagnostic("gate_weight", il, rank, LW.ffn_gate_exps);
        diagnostic("up_weight", il, rank, LW.ffn_up_exps);
        diagnostic("down_weight", il, rank, LW.ffn_down_exps);

        if (m.selected_expert_seams''')
t = replace_once(t, '        ggml_tensor * gate = decode_mul_mat_id(ctx, LW.ffn_gate_exps, cur3, selected);\n        ggml_tensor * h = swiglu(gate, up);', '''        ggml_tensor * gate = decode_mul_mat_id(ctx, LW.ffn_gate_exps, cur3, selected);
        diagnostic("gate_projection", il, rank, gate);
        diagnostic("up_projection", il, rank, up);
        ggml_tensor * h = swiglu(gate, up);
        diagnostic("activation", il, rank, h);''')
t = replace_once(t, '        trace("moe_down", il, rank, experts);', '        trace("moe_down", il, rank, experts);\n        diagnostic("down_projection", il, rank, experts);')
t = replace_once(t, '        if (n_used == 1) moe_out = ggml_cont(ctx, moe_out);\n        return moe_out;', '        if (n_used == 1) moe_out = ggml_cont(ctx, moe_out);\n        diagnostic("output", il, rank, moe_out);\n        return moe_out;')
t = replace_once(t, '            res.selected_expert_seams.push_back({segment, boundary, m.backends[0]});\n            return output;', '            res.selected_expert_seams.push_back({segment, boundary, m.backends[0]});\n            diagnostic("output", il, rank, output);\n            return output;')
t = replace_once(t, '        const auto routes = static_cast<size_t>(seam.segment.n_used) * seam.segment.seq_len;', '''#if defined(TSG_GGML_TEST_HOOKS)
        tsg_moe_diag::Scope diagnostic_scope(seam.segment.layer, 0, seam.segment.seq_len, seam.segment.swiglu_clamp);
        if (tsg_moe_diag::layer_enabled(seam.segment.layer)) {
            tsg_moe_diag::capture("selected_boundary_input", seam.segment.moe_in);
            tsg_moe_diag::capture("selected_boundary_ids", seam.segment.sel_ids);
            tsg_moe_diag::capture("selected_boundary_weights", seam.segment.weights);
        }
#endif
        const auto routes = static_cast<size_t>(seam.segment.n_used) * seam.segment.seq_len;''')
t = replace_once(t, '    graph.selected_expert_completed = 0; graph.selected_expert_failed = false;', '''    graph.selected_expert_completed = 0; graph.selected_expert_failed = false;
#if defined(TSG_GGML_TEST_HOOKS)
    auto& diagnostic = tsg_moe_diag::context();
    diagnostic = {};
    diagnostic.graph = reinterpret_cast<std::uintptr_t>(&graph);
    diagnostic.model = graph.diagnostic_model;
    diagnostic.evaluation = ++graph.diagnostic_evaluation;
    diagnostic.tokens = static_cast<int>(graph.nt);
    diagnostic.clamp = graph.diagnostic_clamp;
    diagnostic.mode = graph.selected_expert_seams.empty() ? "ordinary" : "compact";
#endif''')
t = replace_once(t, 'static void print_traces(const graph_build_result * gr)\n{', '''static void print_traces(const graph_build_result * gr)
{
#if defined(TSG_GGML_TEST_HOOKS)
    for (const auto& capture : gr->diagnostic_tensors) {
        tsg_moe_diag::Scope scope(capture.layer, 0, static_cast<int>(gr->nt), gr->diagnostic_clamp);
        tsg_moe_diag::capture(capture.name.c_str(), capture.tensor);
    }
#endif''')
p.write_text(t, newline='\n')

p = native / 'ggml_ops_host_moe_cache.cpp'
t = p.read_text()
t = replace_once(t, '#include "ggml_ops_internal.h"', '#include "ggml_ops_internal.h"\n#include "glm_moe_private_diagnostic.h"')
t = replace_once(t, '            std::size_t test_physical_bytes = 0;', '''            std::size_t test_physical_bytes = 0;
            struct diagnostic_tensor { std::string name; ggml_tensor* tensor; };
            std::vector<std::vector<diagnostic_tensor>> diagnostic_widths;''')
plan_rows = 'variants' if 'entry.width_graphs.reserve(variants);' in t else 'graph_rows'
t = replace_once(t, '            entry.width_graphs.reserve(' + plan_rows + ');', '''            entry.width_graphs.reserve(''' + plan_rows + ''');
#if defined(TSG_GGML_TEST_HOOKS)
            if (tsg_moe_diag::enabled()) entry.diagnostic_widths.resize(''' + plan_rows + ''');
#endif''')
t = replace_once(t, '                auto* down = ggml_mul_mat_id(ctx, entry.weight[2], activation, ids);', '''                auto* down = ggml_mul_mat_id(ctx, entry.weight[2], activation, ids);
#if defined(TSG_GGML_TEST_HOOKS)
                if (tsg_moe_diag::enabled()) {
                    entry.diagnostic_widths[index] = {
                        {"compact_input", input}, {"compact_remapped_ids", ids}, {"compact_weights", routes},
                        {"compact_gate_projection", gate_projection}, {"compact_up_projection", up_projection},
                        {"compact_gate_clamped", gate}, {"compact_up_clamped", up},
                        {"compact_activation", activation}, {"compact_down_projection", down}};
                    // Debug capture changes lifetimes/fusion; not a production benchmark.
                    for (const auto& capture : entry.diagnostic_widths[index]) ggml_set_output(capture.tensor);
                }
#endif''')
t = replace_once(t, '            drain.active = false; // graph_compute drained all preceding uploads.', '''            drain.active = false; // graph_compute drained all preceding uploads.
#if defined(TSG_GGML_TEST_HOOKS)
            if (tsg_moe_diag::layer_enabled(hm.layer) && !entry.diagnostic_widths.empty()) {
                sync_backend(entry.backend);
                tsg_moe_diag::Scope scope(hm.layer, row, execution_rows, hm.swiglu_clamp);
                for (const auto& capture : entry.diagnostic_widths[variant])
                    tsg_moe_diag::capture(capture.name.c_str(), capture.tensor);
                tsg_moe_diag::capture("compact_resident_gate_weight", entry.weight[0]);
                tsg_moe_diag::capture("compact_resident_up_weight", entry.weight[1]);
                tsg_moe_diag::capture("compact_resident_down_weight", entry.weight[2]);
                tsg_moe_diag::capture("compact_output", entry.width_outputs[variant]);
            }
#endif''')
p.write_text(t, newline='\n')

changes={}
for name, expected in original['source_snapshot'].items():
    current=sha(out/'source'/name)
    if current!=expected: changes[name]={'before':expected,'after':current}
expected_changes={'TensorSharp.GGML.Native/ggml_ops_glm_dsa.cpp','TensorSharp.GGML.Native/ggml_ops_host_moe_cache.cpp'}
if set(changes)!=expected_changes: raise ValueError('Unexpected owned source changes.')
source_after={name:sha(source_root/name) for name in original['source_snapshot']}
after_upstream=upstream_state()
if source_after!=original['source_snapshot'] or after_upstream!=before_upstream:
    raise ValueError('Original source/upstream changed during private preparation.')
report={
    'stage':'Private diagnostic source candidate; not compiled or run.',
    'utc_created':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'original_source_count':len(original['source_snapshot']),
    'original_manifest':str(args.manifest.resolve()),'original_manifest_sha256':sha(args.manifest),
    'original_sources':original['source_snapshot'],'original_sources_after':source_after,
    'native_reference':str(args.native_reference.resolve()),'native_reference_sha256':identity,
    'upstream_before':before_upstream,'upstream_after':after_upstream,'changes':changes,
    'new_header_sha256':sha(native/'glm_moe_private_diagnostic.h'),
    'candidate_sources':{str(p.relative_to(out/'source')):sha(p) for p in (out/'source').rglob('*') if p.is_file()},
    'limits':{'default_layer':1,'default_minimum_rows':16,'default_payload_bytes':64<<20,'maximum_single_tensor_bytes':8<<20,'staging_chunk_bytes':1<<20},
    'limitations':['Additional OUTPUT lifetimes change fusion; this is not a production performance build.',
        'Captures are bounded fixture diagnostics, not full model memory-budget qualification.',
        'Private source must be compiled separately with TEST_HOOKS; do not publish it or overwrite original native outputs.',
        'Capture exact row/type/ne/nb/nbytes metadata and use the CPU oracle; aggregate sums are insufficient.']}
(out/'capture-source-manifest.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'path':str(out),'original_source_count':report['original_source_count'],'changes':list(changes),'new_headers':1}))
