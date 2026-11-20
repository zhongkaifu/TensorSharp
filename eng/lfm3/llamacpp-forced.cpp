// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// llama.cpp reference for eng/ForcedLogitProbe: every case's prompt (a UTF-8 text file, special tokens parsed, no BOS
// added) is decoded greedily for N steps with libllama, and the run is written in ForcedLogitProbe's reference layout -
// cases.json (name, prompt_tokens, generated_tokens) and <name>.f32 (the logits of every step, step 0 after the
// prompt, step i after forcing generated_tokens[i - 1]) - so `ForcedLogitProbe MODEL OUT BACKEND --reference DIR`
// teacher-forces TensorSharp through the same histories and compares every row.
//
// Build against a llama.cpp build tree (the sources stay unchanged):
//   c++ -std=c++17 -O2 eng/lfm3/llamacpp-forced.cpp -I$LLAMA/include -I$LLAMA/ggml/include \
//       -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o artifacts/lfm3/llamacpp-forced
// Run:
//   llamacpp-forced MODEL OUT_DIR STEPS NGL [OPTIONS] CASE.txt [CASE.txt ...]
//   (NGL 99 = every layer on the GPU (Metal, CUDA, Vulkan: the build's); -1 = ggml-cpu only, no GPU device or op offload;
//   the case name is the file's stem)
// OPTIONS (to match a TensorSharp lane's arithmetic; defaults are libllama's context defaults):
//   --threads N        decode and batch threads (default 4, GGML_DEFAULT_N_THREADS)
//   --fa on|off|auto   flash attention (default auto)
//   --kv f16|f32       K and V cache type (default f16); TensorSharp's ggml_cpu lane uses f16 for quantized and BF16
//                      checkpoints unless KV_CACHE_DTYPE says otherwise
//   --no-repack        keep weights in their GGUF layout (no CPU repacked buffer types), as TensorSharp's ggml_cpu
//                      binds them with TS_GGML_CPU_REPACK=0 (ggml-cpu's plain vec_dot products); by default on ARM64 it
//                      reads them from ggml-cpu's repacked layouts, as llama.cpp does without this option
//   --ubatch N         physical batch rows (default 512); llama.cpp's own run at another prompt batch measures how far
//                      its kernel choice alone moves the logits
//   --force DIR        teacher-force the generated tokens of DIR/cases.json (another reference run, same prompts)
//                      instead of decoding greedily, so two llama.cpp configurations compare row by row on one history
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(2); }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static std::string stem(const std::string & path) {
    size_t slash = path.find_last_of('/');
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

// The ids of `key` in the case named `name` of a cases.json (this tool's or ForcedLogitProbe's layout).
static std::vector<llama_token> case_ids(const std::string & json, const std::string & name, const std::string & key) {
    size_t at = json.find("\"name\": \"" + name + "\"");
    if (at == std::string::npos) { fprintf(stderr, "--force: no case %s\n", name.c_str()); exit(2); }
    size_t k = json.find("\"" + key + "\"", at);
    size_t open = json.find('[', k), close = json.find(']', open);
    if (k == std::string::npos || open == std::string::npos || close == std::string::npos) {
        fprintf(stderr, "--force: case %s has no %s\n", name.c_str(), key.c_str()); exit(2);
    }
    std::vector<llama_token> ids;
    std::stringstream ss(json.substr(open + 1, close - open - 1));
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.find_first_of("0123456789") != std::string::npos) ids.push_back((llama_token) atoi(item.c_str()));
    }
    return ids;
}

static void append_ids(std::string & json, const std::vector<llama_token> & ids) {
    json += "[";
    for (size_t i = 0; i < ids.size(); i++) {
        if (i) json += ", ";
        json += std::to_string(ids[i]);
    }
    json += "]";
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s MODEL OUT_DIR STEPS NGL [--threads N] [--fa on|off|auto] [--kv f16|f32] "
                        "[--no-repack] [--ubatch N] [--force DIR] CASE.txt [CASE.txt ...]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1], out_dir = argv[2];
    const int steps = atoi(argv[3]), ngl = atoi(argv[4]);

    int threads = 0, ubatch = 512;
    std::string force_json;
    bool repack = true;
    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
    ggml_type kv_type = GGML_TYPE_F16;
    int first_case = 5;
    for (; first_case < argc && strncmp(argv[first_case], "--", 2) == 0; first_case++) {
        const std::string opt = argv[first_case];
        auto value = [&]() -> std::string {
            if (first_case + 1 >= argc) { fprintf(stderr, "%s needs a value\n", opt.c_str()); exit(2); }
            return argv[++first_case];
        };
        if (opt == "--threads") {
            threads = atoi(value().c_str());
        } else if (opt == "--ubatch") {
            ubatch = atoi(value().c_str());
        } else if (opt == "--no-repack") {
            repack = false;
        } else if (opt == "--force") {
            force_json = read_file(value() + "/cases.json");
        } else if (opt == "--fa") {
            const std::string v = value();
            if (v == "on") fa = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            else if (v == "off") fa = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            else if (v == "auto") fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
            else { fprintf(stderr, "--fa on|off|auto\n"); return 2; }
        } else if (opt == "--kv") {
            const std::string v = value();
            if (v == "f16") kv_type = GGML_TYPE_F16;
            else if (v == "f32") kv_type = GGML_TYPE_F32;
            else { fprintf(stderr, "--kv f16|f32\n"); return 2; }
        } else {
            fprintf(stderr, "unknown option %s\n", opt.c_str());
            return 2;
        }
    }
    if (first_case >= argc) { fprintf(stderr, "no cases\n"); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.use_extra_bufts = repack;
    ggml_backend_dev_t no_devices[] = { nullptr };
    if (ngl < 0) {
        mp.devices = no_devices;   // ggml-cpu alone
        mp.n_gpu_layers = 0;
    } else {
        mp.n_gpu_layers = ngl;
    }
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::string cases_json = "[\n";
    for (int c = first_case; c < argc; c++) {
        const std::string text = read_file(argv[c]);
        const std::string name = stem(argv[c]);
        std::vector<llama_token> prompt(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), prompt.data(), (int) prompt.size(),
                               /*add_special*/ false, /*parse_special*/ true);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        prompt.resize(n);
        std::vector<llama_token> forced;
        if (!force_json.empty()) {
            if (case_ids(force_json, name, "prompt_tokens") != prompt) {
                fprintf(stderr, "--force: %s prompt differs\n", name.c_str());
                return 2;
            }
            forced = case_ids(force_json, name, "generated_tokens");
            if ((int) forced.size() < steps) {
                fprintf(stderr, "--force: %s has %zu steps\n", name.c_str(), forced.size());
                return 2;
            }
        }

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t) (n + steps + 64);
        cp.n_batch = 2048;
        cp.n_ubatch = (uint32_t) ubatch;
        cp.flash_attn_type = fa;
        cp.type_k = kv_type;
        cp.type_v = kv_type;
        if (threads > 0) { cp.n_threads = threads; cp.n_threads_batch = threads; }
        if (ngl < 0) cp.op_offload = false;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

        FILE * f = fopen((out_dir + "/" + name + ".f32").c_str(), "wb");
        if (!f) { fprintf(stderr, "cannot write %s/%s.f32 (does OUT_DIR exist?)\n", out_dir.c_str(), name.c_str()); return 1; }
        std::vector<llama_token> generated;
        // The prompt in batches of n_batch rows, logits of the last row only.
        for (int at = 0; at < n; at += (int) cp.n_batch) {
            int rows = std::min((int) cp.n_batch, n - at);
            llama_batch batch = llama_batch_init(rows, 0, 1);
            for (int i = 0; i < rows; i++) {
                batch.token[i] = prompt[at + i];
                batch.pos[i] = at + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = at + i == n - 1;
            }
            batch.n_tokens = rows;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            llama_batch_free(batch);
        }
        int pos = n;
        for (int s = 0; s < steps; s++) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            fwrite(logits, sizeof(float), n_vocab, f);
            int top = 0;
            for (int i = 1; i < n_vocab; i++) if (logits[i] > logits[top]) top = i;
            llama_token token = forced.empty() ? top : forced[s];
            generated.push_back(token);
            if (s + 1 == steps) break;
            llama_batch one = llama_batch_get_one(&token, 1);
            (void) pos;
            if (llama_decode(ctx, one) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            pos++;
        }
        fclose(f);
        llama_free(ctx);

        if (c > first_case) cases_json += ",\n";
        cases_json += "  {\"name\": \"" + name + "\", \"prompt_tokens\": ";
        append_ids(cases_json, prompt);
        cases_json += ", \"generated_tokens\": ";
        append_ids(cases_json, generated);
        cases_json += "}";
        fprintf(stderr, "[llamacpp-forced] %s: %d prompt tokens, %d steps\n", name.c_str(), n, steps);
    }
    cases_json += "\n]\n";
    std::ofstream(out_dir + "/cases.json") << cases_json;
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
