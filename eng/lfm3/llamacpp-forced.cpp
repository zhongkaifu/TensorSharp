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
//   llamacpp-forced MODEL OUT_DIR STEPS NGL CASE.txt [CASE.txt ...]
//   (NGL 99 = every layer on the GPU (Metal, CUDA, Vulkan: the build's); -1 = ggml-cpu only, no GPU device or op offload;
//   the case name is the file's stem). LLAMACPP_FORCED_UBATCH sets n_ubatch (default 512): llama.cpp's own run at another
//   prompt batch measures how far its kernel choice alone moves the logits.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
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
        fprintf(stderr, "usage: %s MODEL OUT_DIR STEPS NGL CASE.txt [CASE.txt ...]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1], out_dir = argv[2];
    const int steps = atoi(argv[3]), ngl = atoi(argv[4]);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
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
    for (int c = 5; c < argc; c++) {
        const std::string text = read_file(argv[c]);
        const std::string name = stem(argv[c]);
        std::vector<llama_token> prompt(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), prompt.data(), (int) prompt.size(),
                               /*add_special*/ false, /*parse_special*/ true);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        prompt.resize(n);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t) (n + steps + 64);
        cp.n_batch = 2048;
        cp.n_ubatch = getenv("LLAMACPP_FORCED_UBATCH") ? (uint32_t) atoi(getenv("LLAMACPP_FORCED_UBATCH")) : 512;
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
            generated.push_back(top);
            if (s + 1 == steps) break;
            llama_token token = top;
            llama_batch one = llama_batch_get_one(&token, 1);
            (void) pos;
            if (llama_decode(ctx, one) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            pos++;
        }
        fclose(f);
        llama_free(ctx);

        if (c > 5) cases_json += ",\n";
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
