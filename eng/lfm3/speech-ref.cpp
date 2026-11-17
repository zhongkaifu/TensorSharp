// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// llama.cpp reference for TensorSharp's LFM3-Omni speech output (TensorSharp.Models/Models/Lfm3/Lfm3Speech*.cs).
//
// 1. Runs llama.cpp's own TTS pipeline (mtmd_helper_gen_audio, what llama-tts runs) on the model's "tts" chat template
//    and records the LM hidden row (post final norm) every frame was sampled from, plus the PCM it produced.
// 2. Replays the frames with the raw mtmd API and a copy of the pipeline's MaskGIT sampler (mtmd-helper-gen.cpp
//    lfm3a_gen_audio_pipeline::sample_frame / apply_rep_penalty / sample, same std::mt19937 and libc++ distributions),
//    recording each pass's depthformer logits, the codes, the feedback rows; the replayed codes must decode to the
//    pipeline's PCM bit for bit (checked: that proves the copy is the pipeline).
// 3. Decodes the codes again as one whole-utterance codec call (streaming vs whole in llama.cpp itself).
// 4. Times the pieces on the device: a depthformer pass, the feedback row, a codec frame (graph + host ISTFT).
// 5. Writes a compact fixture (fixture.json: hidden rows of the first frames, per-pass logit statistics, codes, PCM
//    statistics, an RNG check) and the full outputs (pipeline.wav, pcm_*.f32, hidden.f32) to OUT_DIR.
//
// Build against a llama.cpp build tree (its sources stay unchanged), e.g. with eng/lfm3/speech-ref.sh:
//   c++ -std=c++17 -O2 eng/lfm3/speech-ref.cpp -I$LLAMA/include -I$LLAMA/ggml/include -I$LLAMA/tools/mtmd \
//       -I$LLAMA/common -I$LLAMA/vendor -L$LLAMA/build/bin -lllama -lmtmd -lllama-common -lggml -lggml-base \
//       -Wl,-rpath,$LLAMA/build/bin -o artifacts/lfm3/speech/speech-ref
// Run:
//   speech-ref LM MMPROJ OUT_DIR [--text T] [--seed S] [--fixture-frames K] [--ngl N] [--reps R] [--lang L] [--voice V]
//              [--dump-logits 1]   (logits.f32: every pass's 12 x 2017 logits before the penalty, frame by frame)
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "chat.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static constexpr int N_CB = 12;
static constexpr int SCHEDULE[] = { 4, 4, 4 };

struct sampler_cfg {
    float temp = 0.4f;
    float rep_penalty = 1.2f;
    int rep_window = 64;
};

// --- copies of lfm3a_gen_audio_pipeline (tools/mtmd/mtmd-helper-gen.cpp) ------------------------------------------

static void apply_rep_penalty(float * l, size_t n_vocab, int cb, const std::vector<int32_t> & hist, const sampler_cfg & c) {
    if (c.rep_penalty <= 1.0f || c.rep_window <= 0) {
        return;
    }
    const size_t n_hist = hist.size() / N_CB;
    const size_t n_win  = std::min(n_hist, (size_t) c.rep_window);
    std::vector<bool> seen(n_vocab, false);
    for (size_t f = n_hist - n_win; f < n_hist; f++) {
        const int32_t code = hist[f * N_CB + (size_t) cb];
        if (code >= 0 && (size_t) code < n_vocab && !seen[(size_t) code]) {
            seen[(size_t) code] = true;
            l[code] = l[code] > 0.0f ? l[code] / c.rep_penalty : l[code] * c.rep_penalty;
        }
    }
}

static int32_t sample(const float * l, size_t n_vocab, float temp, std::mt19937 & rng) {
    if (temp <= 0.0f) {
        return (int32_t) (std::max_element(l, l + n_vocab) - l);
    }
    const float lmax = *std::max_element(l, l + n_vocab);
    std::vector<double> prob(n_vocab);
    double sum = 0.0;
    for (size_t v = 0; v < n_vocab; v++) {
        prob[v] = std::exp((double) (l[v] - lmax) / temp);
        sum += prob[v];
    }
    double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * sum;
    for (size_t v = 0; v < n_vocab; v++) {
        r -= prob[v];
        if (r <= 0.0) {
            return (int32_t) v;
        }
    }
    return (int32_t) n_vocab - 1;
}

// --- small helpers ---------------------------------------------------------------------------------------------------

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static std::string b64(const void * data, size_t n) {
    static const char * tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const unsigned char * p = (const unsigned char *) data;
    std::string out;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = p[i] << 16;
        if (i + 1 < n) v |= p[i + 1] << 8;
        if (i + 2 < n) v |= p[i + 2];
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += i + 1 < n ? tbl[(v >> 6) & 63] : '=';
        out += i + 2 < n ? tbl[v & 63] : '=';
    }
    return out;
}

static std::string fmt(double v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.9g", v);
    return buf;
}

static std::string json_str(const std::string & v) {
    std::string s = "\"";
    for (char ch : v) {
        if (ch == '"' || ch == '\\') { s += '\\'; s += ch; }
        else if (ch == '\n') s += "\\n";
        else s += ch;
    }
    return s + "\"";
}

static std::string json_floats(const std::vector<float> & v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) s += (i ? "," : "") + fmt(v[i]);
    return s + "]";
}

static std::string json_ints(const int32_t * v, size_t n) {
    std::string s = "[";
    for (size_t i = 0; i < n; i++) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}

// statistics of one codebook's logits: what the TensorSharp test compares
static std::string logit_stats(const float * l, size_t n, int cb, int stride) {
    std::vector<int> idx(n);
    for (size_t i = 0; i < n; i++) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + 8, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
    double mx = l[idx[0]], s = 0, ss = 0, se = 0;
    for (size_t i = 0; i < n; i++) {
        s += l[i];
        ss += (double) l[i] * l[i];
        se += std::exp((double) l[i] - mx);
    }
    std::string top = "[";
    for (int i = 0; i < 8; i++) top += (i ? ",[" : "[") + std::to_string(idx[i]) + "," + fmt(l[idx[i]]) + "]";
    top += "]";
    std::vector<float> sample;
    for (size_t i = 0; i < n; i += (size_t) stride) sample.push_back(l[i]);
    return "{\"cb\":" + std::to_string(cb) + ",\"top\":" + top + ",\"lse\":" + fmt(mx + std::log(se)) + ",\"mean\":" + fmt(s / n) +
           ",\"rms\":" + fmt(std::sqrt(ss / n)) + ",\"stride\":" + std::to_string(stride) + ",\"sample\":" + json_floats(sample) + "}";
}

static void write_f32(const std::string & path, const std::vector<float> & v) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(2); }
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
}

static void write_wav16(const std::string & path, const std::vector<float> & pcm, int rate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(2); }
    const uint32_t data_sz = (uint32_t) (pcm.size() * 2), riff_sz = 36 + data_sz, fmt_sz = 16, rate32 = (uint32_t) rate,
                   byte_rate = (uint32_t) rate * 2;
    const uint16_t fmt_pcm = 1, ch = 1, align = 2, bits = 16;
    fwrite("RIFF", 1, 4, f); fwrite(&riff_sz, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); fwrite(&fmt_sz, 4, 1, f); fwrite(&fmt_pcm, 2, 1, f); fwrite(&ch, 2, 1, f);
    fwrite(&rate32, 4, 1, f); fwrite(&byte_rate, 4, 1, f); fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&data_sz, 4, 1, f);
    for (float v : pcm) {
        const int16_t s = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, v)) * 32767.0f);
        fwrite(&s, 2, 1, f);
    }
    fclose(f);
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return INFINITY;
    float m = 0;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s LM MMPROJ OUT_DIR [--text T] [--seed S] [--fixture-frames K] [--ngl N] [--reps R] [--lang L] [--voice V] [--dump-logits 1]\n", argv[0]);
        return 2;
    }
    const std::string lm_path = argv[1], mmproj_path = argv[2], out_dir = argv[3];
    std::string text = "Hello! This is a short test of speech output, eighty milliseconds at a time.";
    std::string lang = "en", voice = "female";
    uint32_t seed = 1234;
    int fixture_frames = 4, ngl = 99, reps = 20;
    bool dump_logits = false;
    for (int i = 4; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--text") text = v;
        else if (k == "--seed") seed = (uint32_t) std::stoul(v);
        else if (k == "--fixture-frames") fixture_frames = std::stoi(v);
        else if (k == "--ngl") ngl = std::stoi(v);
        else if (k == "--reps") reps = std::stoi(v);
        else if (k == "--lang") lang = v;
        else if (k == "--voice") voice = v;
        else if (k == "--dump-logits") dump_logits = v == "1";
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(lm_path.c_str(), mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", lm_path.c_str()); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.embeddings = true;   // the hidden rows feed the depthformer (llama-tts does the same)
    llama_context * lctx = llama_init_from_model(model, cp);
    if (!lctx) { fprintf(stderr, "cannot create a context\n"); return 1; }

    mtmd_context_params mtp = mtmd_context_params_default();
    mtp.use_gpu = ngl > 0;
    mtmd_context * mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mtp);
    if (!mctx || mtmd_gen_audio_get_info(mctx).type != MTMD_GEN_AUDIO_TYPE_LFM3A) {
        fprintf(stderr, "the mmproj has no LFM3 speech output\n");
        return 1;
    }
    const int rate = mtmd_gen_audio_get_info(mctx).sample_rate;
    const int n_embd = llama_model_n_embd(model);
    const mtmd_gen_inp defaults = mtmd_gen_inp_default(mctx);
    sampler_cfg cfg;
    cfg.temp = defaults.temp;
    cfg.rep_penalty = defaults.penalty_repeat;
    cfg.rep_window = defaults.penalty_last_n;

    auto tmpls = common_chat_templates_init(model, "");
    const std::string prompt = common_chat_get_tts_prompt(tmpls.get(), text, lang, voice);
    if (prompt.empty()) { fprintf(stderr, "the model has no \"tts\" chat template\n"); return 1; }
    std::vector<llama_token> ptoks(prompt.size() + 16);
    const int n_ptoks = llama_tokenize(llama_model_get_vocab(model), prompt.c_str(), (int32_t) prompt.size(), ptoks.data(),
                                       (int32_t) ptoks.size(), false, true);

    // ---- 1. llama.cpp's pipeline, recording the hidden row of every frame ----
    mtmd_helper_gen_audio * gen = mtmd_helper_gen_audio_init(lctx, mctx);
    mtmd_helper_gen_audio_inp inp = mtmd_helper_gen_audio_inp_default();
    inp.seq_id = 0;
    inp.prompt = prompt.c_str();
    inp.prompt_len = prompt.size();
    inp.seed = seed;
    inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
    inp.stream = false;
    if (mtmd_helper_gen_audio_set_input(gen, &inp) != 0) { fprintf(stderr, "set_input failed\n"); return 1; }
    const double t_prompt0 = now_ms();
    while (true) {
        const int r = mtmd_helper_gen_audio_step_prompt(gen, 512);
        if (r < 0) { fprintf(stderr, "prompt failed\n"); return 1; }
        if (r == 0) break;
    }
    const double t_prompt = now_ms() - t_prompt0;
    std::vector<std::vector<float>> hidden;
    const float * h = llama_get_embeddings_ith(lctx, -1);
    std::vector<double> step_ms;
    bool stop = false;
    const int max_frames = mtmd_helper_gen_audio_max_frames(gen);
    while (!stop && (int) hidden.size() < max_frames) {
        hidden.emplace_back(h, h + n_embd);
        const float * next = nullptr;
        const double t0 = now_ms();
        if (mtmd_helper_gen_audio_step_gen(gen, LLAMA_TOKEN_NULL, h, &next, &stop) != 0) { fprintf(stderr, "step_gen failed\n"); return 1; }
        step_ms.push_back(now_ms() - t0);
        if (!next) break;
        h = next;
    }
    int32_t out_rate = 0;
    const char * data = nullptr;
    size_t data_len = 0;
    int64_t n_samples = 0;
    if (mtmd_helper_gen_audio_get_output(gen, &out_rate, &data, &data_len, &n_samples) != 0) { fprintf(stderr, "get_output failed\n"); return 1; }
    std::vector<float> pcm_pipeline((const float *) data, (const float *) data + n_samples);
    fprintf(stderr, "pipeline: %d prompt tokens (%.1f ms), %zu steps, %lld samples\n", n_ptoks, t_prompt, hidden.size(), (long long) n_samples);

    // ---- 2. replay: the sampler copy on the raw API ----
    mtmd_gen_state_params sp = mtmd_gen_state_params_default();
    sp.seed = seed;
    mtmd_gen_state * st = mtmd_gen_state_init(mctx, sp);
    std::mt19937 rng(seed);
    std::vector<int32_t> hist, all_codes;
    std::vector<float> pcm_stream, logits, all_logits;   // all_logits: every pass's [12 x 2017] before the penalty
    std::string passes_json = "[", fb_json = "[";
    int eos_frame = -1;
    std::vector<double> pass_ms, fb_ms, wav_ms;
    for (size_t f = 0; f < hidden.size(); f++) {
        std::vector<int> order(N_CB);
        for (int i = 0; i < N_CB; i++) order[i] = i;
        std::shuffle(order.begin(), order.end(), rng);
        std::vector<int32_t> codes(N_CB, -1);
        mtmd_gen_inp gi = mtmd_gen_inp_default(mctx);
        gi.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        gi.embd = hidden[f].data();
        gi.codes = codes.data();
        gi.n_codes = codes.size();
        mtmd_gen_out go{};
        const bool keep = (int) f < fixture_frames;
        std::string fj = "{\"order\":" + std::string(json_ints(order.data(), N_CB)) + ",\"pass\":[";
        int k = 0;
        for (int p = 0; p < 3; p++) {
            const std::vector<int32_t> known = codes;
            const double t0 = now_ms();
            if (mtmd_gen_audio_process(mctx, st, &gi, &go) != 0 || go.n_feats % N_CB != 0) { fprintf(stderr, "depthformer failed\n"); return 1; }
            pass_ms.push_back(now_ms() - t0);
            const size_t n_vocab = go.n_feats / N_CB;
            logits.assign(go.feats, go.feats + go.n_feats);
            if (dump_logits) all_logits.insert(all_logits.end(), logits.begin(), logits.end());
            std::string pj = "{\"known\":" + json_ints(known.data(), N_CB) + ",\"logits\":[";
            for (int j = 0; j < SCHEDULE[p]; j++, k++) {
                const int i = order[(size_t) k];
                float * l = logits.data() + (size_t) i * n_vocab;
                if (keep) pj += (j ? "," : "") + logit_stats(l, n_vocab, i, 41);   // before the penalty: the network's output
                apply_rep_penalty(l, n_vocab, i, hist, cfg);
                codes[(size_t) i] = sample(l, n_vocab, cfg.temp, rng);
            }
            pj += "]}";
            if (keep) fj += (p ? "," : "") + pj;
        }
        fj += "]}";
        if (keep) passes_json += (f ? "," : "") + fj;
        const int32_t eos = (int32_t) (logits.size() / N_CB) - 1;
        const bool is_eos = std::find(codes.begin(), codes.end(), eos) != codes.end();
        all_codes.insert(all_codes.end(), codes.begin(), codes.end());
        if (is_eos) {
            eos_frame = (int) f;
            break;
        }
        hist.insert(hist.end(), codes.begin(), codes.end());
        // the feedback row (all codes known)
        const double t1 = now_ms();
        if (mtmd_gen_audio_process(mctx, st, &gi, &go) != 0) { fprintf(stderr, "feedback failed\n"); return 1; }
        fb_ms.push_back(now_ms() - t1);
        if (keep) {
            double s = 0, ss = 0;
            for (int i = 0; i < n_embd; i++) { s += go.embd[i]; ss += (double) go.embd[i] * go.embd[i]; }
            std::vector<float> head(go.embd, go.embd + 16);
            fb_json += (f ? "," : "") + std::string("{\"first\":") + json_floats(head) + ",\"sum\":" + fmt(s) + ",\"rms\":" + fmt(std::sqrt(ss / n_embd)) + "}";
        }
        mtmd_gen_inp wi = mtmd_gen_inp_default(mctx);
        wi.type = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        wi.codes = codes.data();
        wi.n_codes = codes.size();
        const double t2 = now_ms();
        if (mtmd_gen_audio_process(mctx, st, &wi, &go) != 0) { fprintf(stderr, "codec failed\n"); return 1; }
        wav_ms.push_back(now_ms() - t2);
        pcm_stream.insert(pcm_stream.end(), go.audio, go.audio + go.n_samples);
    }
    passes_json += "]";
    fb_json += "]";
    {
        mtmd_gen_inp wi = mtmd_gen_inp_default(mctx);
        wi.type = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        mtmd_gen_out go{};
        if (mtmd_gen_audio_process(mctx, st, &wi, &go) != 0) { fprintf(stderr, "flush failed\n"); return 1; }
        pcm_stream.insert(pcm_stream.end(), go.audio, go.audio + go.n_samples);
    }
    const int n_audio = (int) (eos_frame >= 0 ? eos_frame : hidden.size());
    const float pipeline_vs_replay = max_abs_diff(pcm_pipeline, pcm_stream);
    fprintf(stderr, "replay: %d audio frames, eos at %d, pipeline vs replay max |d| %g\n", n_audio, eos_frame, pipeline_vs_replay);

    // ---- 3. the whole utterance as one codec call ----
    std::vector<float> pcm_whole;
    {
        mtmd_gen_state_reset(st);
        std::vector<int32_t> codes(all_codes.begin(), all_codes.begin() + (size_t) n_audio * N_CB);
        mtmd_gen_inp wi = mtmd_gen_inp_default(mctx);
        wi.type = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        wi.codes = codes.data();
        wi.n_codes = codes.size();
        mtmd_gen_out go{};
        if (mtmd_gen_audio_process(mctx, st, &wi, &go) != 0) { fprintf(stderr, "whole codec failed\n"); return 1; }
        pcm_whole.assign(go.audio, go.audio + go.n_samples);
        wi.codes = nullptr;
        wi.n_codes = 0;
        if (mtmd_gen_audio_process(mctx, st, &wi, &go) != 0) { fprintf(stderr, "flush failed\n"); return 1; }
        pcm_whole.insert(pcm_whole.end(), go.audio, go.audio + go.n_samples);
    }
    const float stream_vs_whole = max_abs_diff(pcm_stream, pcm_whole);
    fprintf(stderr, "stream vs whole: %zu / %zu samples, max |d| %g\n", pcm_stream.size(), pcm_whole.size(), stream_vs_whole);

    // ---- 4. timings: steady-state medians over reps ----
    std::vector<double> t_pass, t_fb, t_wav;
    {
        mtmd_gen_state_reset(st);
        std::vector<int32_t> masked(N_CB, -1), known(all_codes.begin(), all_codes.begin() + N_CB);
        for (int r = 0; r < reps; r++) {
            mtmd_gen_inp gi = mtmd_gen_inp_default(mctx);
            gi.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
            gi.embd = hidden[0].data();
            gi.codes = masked.data();
            gi.n_codes = N_CB;
            mtmd_gen_out go{};
            double t0 = now_ms();
            mtmd_gen_audio_process(mctx, st, &gi, &go);
            t_pass.push_back(now_ms() - t0);
            gi.codes = known.data();
            t0 = now_ms();
            mtmd_gen_audio_process(mctx, st, &gi, &go);
            t_fb.push_back(now_ms() - t0);
            mtmd_gen_inp wi = mtmd_gen_inp_default(mctx);
            wi.type = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
            wi.codes = all_codes.data() + (size_t) (r % std::max(1, n_audio)) * N_CB;
            wi.n_codes = N_CB;
            t0 = now_ms();
            mtmd_gen_audio_process(mctx, st, &wi, &go);
            t_wav.push_back(now_ms() - t0);
        }
    }
    const double frame_ms = 3 * median(t_pass) + median(t_fb) + median(t_wav);
    fprintf(stderr, "timing (median of %d): depthformer pass %.3f ms, feedback %.3f ms, codec frame %.3f ms -> %.3f ms per 80 ms frame "
                    "(pipeline step_gen median %.3f ms incl. the LM decode)\n",
            reps, median(t_pass), median(t_fb), median(t_wav), frame_ms, median(step_ms));

    // ---- 5. outputs ----
    write_wav16(out_dir + "/pipeline.wav", pcm_pipeline, rate);
    write_f32(out_dir + "/pcm_pipeline.f32", pcm_pipeline);
    write_f32(out_dir + "/pcm_whole.f32", pcm_whole);
    if (dump_logits) write_f32(out_dir + "/logits.f32", all_logits);
    {
        std::vector<float> all;
        for (auto & r : hidden) all.insert(all.end(), r.begin(), r.end());
        write_f32(out_dir + "/hidden.f32", all);
        std::ofstream codes_out(out_dir + "/codes.txt");
        for (size_t i = 0; i < all_codes.size(); i++) codes_out << all_codes[i] << ((i + 1) % N_CB ? ' ' : '\n');
    }

    // the RNG check: three frames' draws of mt19937(seed) as the sampler takes them
    std::string rng_json;
    {
        std::mt19937 r2(seed);
        rng_json = "{\"seed\":" + std::to_string(seed) + ",\"frames\":[";
        for (int f = 0; f < 3; f++) {
            std::vector<int> order(N_CB);
            for (int i = 0; i < N_CB; i++) order[i] = i;
            std::shuffle(order.begin(), order.end(), r2);
            std::string u = "[";
            for (int j = 0; j < N_CB; j++) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%.17g", std::uniform_real_distribution<double>(0.0, 1.0)(r2));
                u += (j ? "," : "") + std::string(buf);
            }
            rng_json += std::string(f ? "," : "") + "{\"order\":" + json_ints(order.data(), N_CB) + ",\"uniform\":" + u + "]}";
        }
        rng_json += "]}";
    }

    const int kf = std::min(fixture_frames, (int) hidden.size());
    std::vector<float> hid;
    for (int f = 0; f < kf; f++) hid.insert(hid.end(), hidden[f].begin(), hidden[f].end());
    // the PCM: per-frame RMS, every 61st sample, totals
    std::vector<float> frame_rms, strided;
    for (size_t at = 0; at < pcm_pipeline.size(); at += 1920) {
        const size_t end = std::min(pcm_pipeline.size(), at + 1920);
        double ss = 0;
        for (size_t i = at; i < end; i++) ss += (double) pcm_pipeline[i] * pcm_pipeline[i];
        frame_rms.push_back((float) std::sqrt(ss / (end - at)));
    }
    double ss = 0, mx = 0;
    for (size_t i = 0; i < pcm_pipeline.size(); i++) {
        ss += (double) pcm_pipeline[i] * pcm_pipeline[i];
        mx = std::max(mx, (double) std::fabs(pcm_pipeline[i]));
        if (i % 61 == 0) strided.push_back(pcm_pipeline[i]);
    }

    std::ofstream js(out_dir + "/fixture.json");
    js << "{\n\"source\":\"eng/lfm3/speech-ref.cpp against llama.cpp tools/mtmd (lfm3a_gen_audio_pipeline)\",\n";
    js << "\"lm\":\"" << lm_path.substr(lm_path.find_last_of('/') + 1) << "\",\"mmproj\":\"" << mmproj_path.substr(mmproj_path.find_last_of('/') + 1) << "\",\n";
    js << "\"text\":" << json_str(text) << ",\"lang\":" << json_str(lang) << ",\"voice\":" << json_str(voice) << ",\"seed\":" << seed << ",\"prompt_tokens\":" << n_ptoks << ",\n";
    js << "\"prompt\":" << json_str(prompt) << ",\n";
    js << "\"temp\":" << fmt(cfg.temp) << ",\"penalty_repeat\":" << fmt(cfg.rep_penalty) << ",\"penalty_last_n\":" << cfg.rep_window << ",\n";
    js << "\"audio_frames\":" << n_audio << ",\"eos_frame\":" << eos_frame << ",\"sample_rate\":" << rate << ",\n";
    js << "\"codes\":" << json_ints(all_codes.data(), all_codes.size()) << ",\n";
    js << "\"hidden_frames\":" << kf << ",\"hidden_f32_b64\":\"" << b64(hid.data(), hid.size() * sizeof(float)) << "\",\n";
    js << "\"passes\":" << passes_json << ",\n";
    js << "\"feedback\":" << fb_json << ",\n";
    js << "\"pcm\":{\"samples\":" << pcm_pipeline.size() << ",\"rms\":" << fmt(std::sqrt(ss / std::max<size_t>(1, pcm_pipeline.size())))
       << ",\"max_abs\":" << fmt(mx) << ",\"frame_rms\":" << json_floats(frame_rms) << ",\"stride\":61,\"sample\":" << json_floats(strided) << "},\n";
    js << "\"pipeline_vs_replay_max_abs\":" << fmt(pipeline_vs_replay) << ",\"stream_vs_whole_max_abs\":" << fmt(stream_vs_whole) << ",\n";
    js << "\"rng\":" << rng_json << ",\n";
    js << "\"timing_ms\":{\"depthformer_pass\":" << fmt(median(t_pass)) << ",\"feedback\":" << fmt(median(t_fb)) << ",\"codec_frame\":"
       << fmt(median(t_wav)) << ",\"frame_total\":" << fmt(frame_ms) << ",\"pipeline_step\":" << fmt(median(step_ms)) << ",\"reps\":" << reps << "}\n}\n";
    js.close();
    fprintf(stderr, "wrote %s/fixture.json\n", out_dir.c_str());

    mtmd_gen_state_free(st);
    mtmd_helper_gen_audio_free(gen);
    mtmd_free(mctx);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return pipeline_vs_replay == 0.0f ? 0 : 3;
}
