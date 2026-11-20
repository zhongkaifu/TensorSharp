// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// llama.cpp reference for TensorSharp's spoken chat replies (TensorSharp.Models/Models/Lfm3/Lfm3SpeechDriver.cs): runs
// llama.cpp's speech pipeline (mtmd_helper_gen_audio) the way its realtime server continues a conversation - the turn
// rendered as text, the sequence continued (n_past > 0) with out_mode MIXED or AUDIO, text slots sampled greedily by the
// caller with the pipeline's logit bias - for each question of a TSV, and writes each reply's WAV and transcript.
//
// Build against a llama.cpp build tree (its sources stay unchanged):
//   c++ -std=c++17 -O2 eng/lfm3/speech-chat-ref.cpp -I$LLAMA/include -I$LLAMA/ggml/include -I$LLAMA/tools/mtmd \
//       -I$LLAMA/common -I$LLAMA/vendor -L$LLAMA/build/bin -lllama -lmtmd -lllama-common -lggml -lggml-base \
//       -Wl,-rpath,$LLAMA/build/bin -o artifacts/lfm3/speech-chat-ref
// Run:
//   speech-chat-ref LM MMPROJ QUESTIONS.tsv OUT_DIR [mixed|audio] [SEED]
// QUESTIONS.tsv lines: id <TAB> language <TAB> voice <TAB> question [<TAB> system prompt]. The default system prompt is
// llama.cpp's realtime speech line ("Speak in <Language> using a <voice> voice."), then for MIXED the model card's
// "Respond with interleaved text and audio."; a fifth column replaces it verbatim ("-": no system turn). OUT_DIR gets
// <id>.wav, and replies.tsv (id, transcript, frames, ms).
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "chat.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void write_wav(const std::string & path, const std::vector<float> & pcm, int rate) {
    std::ofstream f(path, std::ios::binary);
    auto u32 = [&](uint32_t v) { f.write((const char *) &v, 4); };
    auto u16 = [&](uint16_t v) { f.write((const char *) &v, 2); };
    const uint32_t data = (uint32_t) pcm.size() * 2;
    f.write("RIFF", 4); u32(36 + data); f.write("WAVE", 4); f.write("fmt ", 4); u32(16); u16(1); u16(1); u32(rate); u32(rate * 2);
    u16(2); u16(16); f.write("data", 4); u32(data);
    for (float x : pcm) {
        const float c = x < -1.0f ? -1.0f : x > 1.0f ? 1.0f : x;
        const int16_t s = (int16_t) lroundf(c * 32767.0f);
        f.write((const char *) &s, 2);
    }
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s LM MMPROJ QUESTIONS.tsv OUT_DIR [mixed|audio] [SEED]\n", argv[0]);
        return 2;
    }
    const std::string lm_path = argv[1], mmproj_path = argv[2], questions_path = argv[3], out_dir = argv[4];
    const std::string mode = argc > 5 ? argv[5] : "mixed";
    const uint32_t seed = argc > 6 ? (uint32_t) std::stoul(argv[6]) : 7;
    const std::map<std::string, std::string> names = { { "en", "English" }, { "zh", "Chinese" }, { "fr", "French" }, { "de", "German" },
                                                       { "es", "Spanish" }, { "ja", "Japanese" }, { "ko", "Korean" } };

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(lm_path.c_str(), mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", lm_path.c_str()); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.embeddings = true;
    llama_context * lctx = llama_init_from_model(model, cp);
    mtmd_context_params mtp = mtmd_context_params_default();
    mtmd_context * mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mtp);
    if (!lctx || !mctx) { fprintf(stderr, "cannot create the contexts\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token bos = llama_vocab_bos(vocab);
    mtmd_helper_gen_audio * gen = mtmd_helper_gen_audio_init(lctx, mctx);

    std::ifstream qf(questions_path);
    std::ofstream replies(out_dir + "/replies.tsv");
    std::string line;
    while (std::getline(qf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> cols;
        std::stringstream ss(line);
        for (std::string c; std::getline(ss, c, '\t');) cols.push_back(c);
        if (cols.size() < 4) continue;
        const std::string & id = cols[0], & lang = cols[1], & voice = cols[2], & question = cols[3];
        std::string system = "Speak in " + (names.count(lang) ? names.at(lang) : lang) + " using a " + voice + " voice.";
        if (mode == "mixed") system += " Respond with interleaved text and audio.";
        // A fifth column replaces the system prompt verbatim ("-": no system turn).
        if (cols.size() > 4 && !cols[4].empty()) system = cols[4];
        const std::string turn = (system == "-" ? std::string() : "<|im_start|>system\n" + system + "<|im_end|>\n") +
                                 "<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n<think></think>";

        // A conversation of one token (BOS) so far: the pipeline continues it with the turn and the mode's token.
        llama_memory_clear(llama_get_memory(lctx), true);
        mtmd_helper_gen_audio_reset(gen);
        llama_token b = bos;
        llama_batch one = llama_batch_get_one(&b, 1);
        if (llama_decode(lctx, one) != 0) { fprintf(stderr, "BOS decode failed\n"); return 1; }
        mtmd_helper_gen_audio_inp inp = mtmd_helper_gen_audio_inp_default();
        inp.seq_id = 0;
        inp.prompt = turn.c_str();
        inp.prompt_len = turn.size();
        inp.n_past = 1;
        inp.out_mode = mode == "mixed" ? MTMD_HELPER_GEN_AUDIO_OUT_MIXED : MTMD_HELPER_GEN_AUDIO_OUT_AUDIO;
        inp.seed = seed;
        inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
        if (mtmd_helper_gen_audio_set_input(gen, &inp) != 0) { fprintf(stderr, "set_input failed\n"); return 1; }
        const double t0 = now_ms();
        while (true) {
            const int r = mtmd_helper_gen_audio_step_prompt(gen, 512);
            if (r < 0) { fprintf(stderr, "prompt failed\n"); return 1; }
            if (r == 0) break;
        }
        const float * h = llama_get_embeddings_ith(lctx, -1);
        size_t n_bias = 0;
        const llama_logit_bias * bias = mtmd_helper_gen_audio_get_logit_bias(gen, &n_bias);
        std::string transcript;
        int frames = 0, slots = 0;
        bool stop = false;
        while (!stop && slots < 600) {
            llama_token tok = LLAMA_TOKEN_NULL;
            const bool text = mtmd_helper_gen_audio_next_is_text(gen);
            if (text) {
                // Greedy, with the pipeline's bias (the server's default sampling is the request's; greedy here).
                std::vector<float> l(llama_get_logits_ith(lctx, -1), llama_get_logits_ith(lctx, -1) + n_vocab);
                for (size_t i = 0; i < n_bias; i++) l[bias[i].token] += bias[i].bias;
                tok = 0;
                for (int v = 1; v < n_vocab; v++) if (l[v] > l[tok]) tok = v;
                if (!llama_vocab_is_control(vocab, tok)) {
                    char piece[256];
                    const int n = llama_token_to_piece(vocab, tok, piece, sizeof piece, 0, false);
                    if (n > 0) transcript.append(piece, n);
                }
            } else {
                frames++;
            }
            const float * next = nullptr;
            if (mtmd_helper_gen_audio_step_gen(gen, tok, h, &next, &stop) != 0) { fprintf(stderr, "step_gen failed\n"); return 1; }
            slots++;
            if (next) h = next;
            else h = llama_get_embeddings_ith(lctx, -1);
        }
        mtmd_helper_gen_audio_flush(gen);
        int32_t rate = 0;
        const char * data = nullptr;
        size_t data_len = 0;
        int64_t n_samples = 0;
        mtmd_helper_gen_audio_get_output(gen, &rate, &data, &data_len, &n_samples);
        std::vector<float> pcm((const float *) data, (const float *) data + n_samples);
        write_wav(out_dir + "/" + id + ".wav", pcm, rate);
        const double ms = now_ms() - t0;
        for (char & c : transcript) if (c == '\t' || c == '\n') c = ' ';
        replies << id << '\t' << transcript << '\t' << frames << '\t' << ms << '\n';
        fprintf(stderr, "%s: %d slots, %d frames, %.2f s audio in %.0f ms: %s\n", id.c_str(), slots, frames, n_samples / (double) rate, ms,
                transcript.c_str());
    }
    mtmd_helper_gen_audio_free(gen);
    return 0;
}
