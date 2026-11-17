// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// llama.cpp reference for TensorSharp's LFM3-Omni audio input (Lfm3AudioPreprocessor + Lfm3AudioEncoder), through the
// public libmtmd / libllama API only (the llama.cpp sources stay unchanged).
//
//   dump MODEL MMPROJ OUT_DIR REPS CLIP.f32 [CLIP.f32 ...]
//       For every clip (raw little-endian F32 PCM, 16 kHz mono; the name is the file's stem) writes OUT_DIR/<name>/:
//         mel.f32            the encoder's input (graph leaf "inp_raw"): mel-major [128][frames], after (v + 4) / 4
//         downsample.conv.f32 the downsampler's last conv, ggml [rows, 16 mel, 256 channels] (rows innermost)
//         downsample.out.f32 [rows][768]; operator_norm-<l>.f32 [rows][768]; embedding_norm.f32 [rows][768]
//         rows.f32           mtmd_get_output_embd(): [rows][2048], what the LM receives
//         meta.json          samples, frames, rows, n_embd, and timings: "encode_ms" (mtmd_encode_chunk of a chunk whose
//                            mel is already computed: the encoder), "mel_encode_ms" (a freshly tokenized chunk: llama.cpp
//                            computes the mel inside the encode call), medians of REPS runs after one warm run each
//       The graph values are read with mtmd's eval callback (mtmd_context_params.cb_eval) on the first encode only; the
//       timed runs do not observe nodes.
//   decode MODEL MMPROJ FILE OUT.f32
//       The 16 kHz mono PCM llama.cpp's helper makes of an audio file (miniaudio: WAV, MP3, FLAC; its linear resampler
//       with an order-8 low-pass), as raw F32, to compare with TensorSharp's Lfm3AudioPreprocessor.LoadAudio.
//   asr MODEL MMPROJ CLIP.f32 LANG MAX_TOKENS [ROWS.f32]
//       Greedy ASR as llama.cpp's /v1/audio/transcriptions prompts LFM3 ("Perform ASR in <Language>.", the audio alone in
//       the user turn, the assistant turn opened with <think></think>); with ROWS.f32 ([rows][2048], e.g. TensorSharp's)
//       the LM reads those rows in place of llama.cpp's own encode. Prints the transcript between "<<<" and ">>>".
//
// Build against a llama.cpp build tree (see eng/lfm3/audio-in/README.md):
//   c++ -std=c++17 -O2 lfm3-audio-ref.cpp -I$LLAMA/include -I$LLAMA/ggml/include -I$LLAMA/tools/mtmd \
//       -L$LLAMA/build/bin -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o lfm3-audio-ref
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <sys/stat.h>
#include <vector>

static std::vector<float> read_f32(const std::string & path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(2); }
    std::streamsize n = in.tellg();
    in.seekg(0);
    std::vector<float> v((size_t) n / sizeof(float));
    in.read((char *) v.data(), (std::streamsize) (v.size() * sizeof(float)));
    return v;
}

static void write_f32(const std::string & path, const float * data, size_t n) {
    std::ofstream out(path, std::ios::binary);
    out.write((const char *) data, (std::streamsize) (n * sizeof(float)));
}

static std::string stem(const std::string & path) {
    size_t slash = path.find_last_of('/');
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

// ------------------------------------------------------------------ graph value capture

struct capture_state {
    bool active = false;
    std::string dir;
    std::set<std::string> written;
};

static bool wanted(const char * name) {
    return !strcmp(name, "downsample.conv") || !strcmp(name, "downsample.out") || !strcmp(name, "embedding_norm") || !strcmp(name, "projected") ||
           !strncmp(name, "operator_norm-", 14);
}

static void dump_tensor(capture_state * st, const ggml_tensor * t, const std::string & name) {
    if (st->written.count(name)) return;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        fprintf(stderr, "skip %s: not contiguous F32\n", name.c_str());
        return;
    }
    std::vector<float> host(ggml_nelements(t));
    ggml_backend_tensor_get(t, host.data(), 0, ggml_nbytes(t));
    write_f32(st->dir + "/" + name + ".f32", host.data(), host.size());
    st->written.insert(name);
}

static bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (capture_state *) user_data;
    if (!st->active) return false;
    // The scheduler hands a node on another backend a copy of an input leaf, named after it ("MTL0#inp_raw#0").
    bool has_raw = false;
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++)
        if (strstr(t->src[i]->name, "inp_raw")) has_raw = true;
    if (ask) return wanted(t->name) || has_raw;
    if (wanted(t->name)) dump_tensor(st, t, t->name);
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++)
        if (strstr(t->src[i]->name, "inp_raw")) dump_tensor(st, t->src[i], "mel");
    return true;
}

// ------------------------------------------------------------------ setup

struct session {
    llama_model * model = nullptr;
    mtmd_context * mtmd = nullptr;
    capture_state capture;
};

static void open_session(session & s, const char * model_path, const char * mmproj_path, bool weights) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = !weights;   // mtmd needs the model's vocabulary and n_embd only, unless the LM decodes
    mp.n_gpu_layers = 99;
    s.model = llama_model_load_from_file(model_path, mp);
    if (!s.model) { fprintf(stderr, "cannot load %s\n", model_path); exit(2); }
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = true;
    cp.print_timings = false;
    cp.n_threads = 8;
    cp.cb_eval = capture_cb;
    cp.cb_eval_user_data = &s.capture;
    s.mtmd = mtmd_init_from_file(mmproj_path, s.model, cp);
    if (!s.mtmd || !mtmd_support_audio(s.mtmd)) { fprintf(stderr, "cannot load the audio tower of %s\n", mmproj_path); exit(2); }
}

static mtmd_input_chunks * tokenize(session & s, const std::string & text, const std::vector<float> & pcm, size_t * audio_chunk) {
    mtmd_bitmap * bitmap = mtmd_bitmap_init_from_audio(pcm.size(), pcm.data());
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text in = { text.c_str(), text.size(), false, true };
    const mtmd_bitmap * bitmaps[] = { bitmap };
    if (mtmd_tokenize(s.mtmd, chunks, &in, bitmaps, 1) != 0) { fprintf(stderr, "mtmd_tokenize failed\n"); exit(2); }
    mtmd_bitmap_free(bitmap);
    *audio_chunk = SIZE_MAX;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++)
        if (mtmd_input_chunk_get_type(mtmd_input_chunks_get(chunks, i)) == MTMD_INPUT_CHUNK_TYPE_AUDIO) *audio_chunk = i;
    if (*audio_chunk == SIZE_MAX) { fprintf(stderr, "no audio chunk\n"); exit(2); }
    return chunks;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

// ------------------------------------------------------------------ modes

static int run_dump(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s dump MODEL MMPROJ OUT_DIR REPS CLIP.f32 [...]\n", argv[0]); return 2; }
    session s;
    open_session(s, argv[2], argv[3], false);
    const std::string out_dir = argv[4];
    const int reps = std::max(1, atoi(argv[5]));
    mkdir(out_dir.c_str(), 0755);
    const std::string marker = mtmd_default_marker();
    for (int a = 6; a < argc; a++) {
        const std::string name = stem(argv[a]);
        const std::vector<float> pcm = read_f32(argv[a]);
        const std::string dir = out_dir + "/" + name;
        mkdir(dir.c_str(), 0755);

        size_t ai;
        mtmd_input_chunks * chunks = tokenize(s, marker, pcm, &ai);
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, ai);
        const size_t n_rows = mtmd_input_chunk_get_n_tokens(chunk);

        s.capture = capture_state();
        s.capture.active = true;
        s.capture.dir = dir;
        if (mtmd_encode_chunk(s.mtmd, chunk) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
        s.capture.active = false;
        // The LM's width from the projector's output (a vocab-only model reports no n_embd).
        if (!s.capture.written.count("projected")) { fprintf(stderr, "the graph named no 'projected' node\n"); return 1; }
        const int n_embd = (int) (read_f32(dir + "/projected.f32").size() / n_rows);
        write_f32(dir + "/rows.f32", mtmd_get_output_embd(s.mtmd), n_rows * (size_t) n_embd);

        // The encoder alone: the chunk's mel is computed by now.
        std::vector<double> enc;
        mtmd_encode_chunk(s.mtmd, chunk);
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            mtmd_encode_chunk(s.mtmd, chunk);
            enc.push_back(now_ms() - t0);
        }
        mtmd_input_chunks_free(chunks);

        // The mel and the encoder: a freshly tokenized chunk computes its mel in the encode call.
        std::vector<double> full;
        for (int r = 0; r <= reps; r++) {
            size_t aj;
            mtmd_input_chunks * fresh = tokenize(s, marker, pcm, &aj);
            double t0 = now_ms();
            mtmd_encode_chunk(s.mtmd, mtmd_input_chunks_get(fresh, aj));
            double ms = now_ms() - t0;
            if (r > 0) full.push_back(ms);
            mtmd_input_chunks_free(fresh);
        }

        const size_t mel_values = s.capture.written.count("mel") ? read_f32(dir + "/mel.f32").size() : 0;
        FILE * meta = fopen((dir + "/meta.json").c_str(), "w");
        fprintf(meta, "{\"samples\": %zu, \"frames\": %zu, \"rows\": %zu, \"n_embd\": %d, \"encode_ms\": %.3f, \"mel_encode_ms\": %.3f, \"reps\": %d}\n",
                pcm.size(), mel_values / 128, n_rows, n_embd, median(enc), median(full), reps);
        fclose(meta);
        printf("%s: %zu samples, %zu frames, %zu rows; encode %.2f ms, mel + encode %.2f ms (median of %d)\n",
               name.c_str(), pcm.size(), mel_values / 128, n_rows, median(enc), median(full), reps);
    }
    mtmd_free(s.mtmd);
    llama_model_free(s.model);
    return 0;
}

static const char * language_name(const std::string & code) {
    static const std::map<std::string, const char *> names = {
        { "ar", "Arabic" }, { "da", "Danish" }, { "de", "German" }, { "el", "Greek" }, { "en", "English" },
        { "es", "Spanish" }, { "fi", "Finnish" }, { "fr", "French" }, { "he", "Hebrew" }, { "it", "Italian" },
        { "ja", "Japanese" }, { "ko", "Korean" }, { "nl", "Dutch" }, { "pl", "Polish" }, { "pt", "Portuguese" },
        { "sv", "Swedish" }, { "tr", "Turkish" }, { "zh", "Chinese" },
    };
    auto it = names.find(code);
    return it == names.end() ? code.c_str() : it->second;
}

static int run_asr(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s asr MODEL MMPROJ CLIP.f32 LANG MAX_TOKENS [ROWS.f32]\n", argv[0]); return 2; }
    session s;
    open_session(s, argv[2], argv[3], true);
    const std::vector<float> pcm = read_f32(argv[4]);
    const std::string lang = argv[5];
    const int max_tokens = atoi(argv[6]);
    std::vector<float> rows;
    if (argc > 7) rows = read_f32(argv[7]);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 2048;
    llama_context * lctx = llama_init_from_model(s.model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(s.model);

    const std::string system = lang.empty() ? "Perform ASR." : std::string("Perform ASR in ") + language_name(lang) + ".";
    const std::string prompt = "<|startoftext|><|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" +
                               mtmd_default_marker() + "<|im_end|>\n<|im_start|>assistant\n<think></think>";
    size_t ai;
    mtmd_input_chunks * chunks = tokenize(s, prompt, pcm, &ai);
    llama_pos n_past = 0;
    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        bool last = i + 1 == n_chunks;
        int rc;
        if (i == ai && !rows.empty()) {
            const size_t n = mtmd_input_chunk_get_n_tokens(chunk);
            if (rows.size() != n * (size_t) llama_model_n_embd_inp(s.model)) {
                fprintf(stderr, "ROWS.f32 holds %zu values, the clip takes %zu rows of %d\n", rows.size(), n, llama_model_n_embd_inp(s.model));
                return 2;
            }
            rc = mtmd_helper_decode_image_chunk(s.mtmd, lctx, chunk, rows.data(), n_past, 0, 2048, &n_past, nullptr, nullptr);
        } else {
            rc = mtmd_helper_eval_chunk_single(s.mtmd, lctx, chunk, n_past, 0, 2048, last, &n_past);
        }
        if (rc != 0) { fprintf(stderr, "decode failed (%d)\n", rc); return 1; }
    }
    mtmd_input_chunks_free(chunks);

    std::string text;
    llama_batch_ext * batch = llama_batch_ext_init(lctx);
    for (int step = 0; step < max_tokens; step++) {
        const float * logits = llama_get_logits_ith(lctx, -1);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        llama_token best = (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
        if (llama_vocab_is_eog(vocab, best) || best == 125014) break;
        char piece[256];
        int n = llama_token_to_piece(vocab, best, piece, sizeof(piece), 0, true);
        if (n > 0) text.append(piece, (size_t) n);
        llama_batch_ext_clear(batch);
        int32_t idx = llama_batch_ext_add_token(batch, 0, best);
        llama_pos pos = n_past++;
        llama_batch_ext_set_pos(batch, idx, &pos);
        llama_batch_ext_set_output_logits(batch, idx, true);
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    llama_batch_ext_free(batch);
    printf("<<<%s>>>\n", text.c_str());
    llama_free(lctx);
    mtmd_free(s.mtmd);
    llama_model_free(s.model);
    return 0;
}

static int run_decode(int argc, char ** argv) {
    if (argc != 6) { fprintf(stderr, "usage: %s decode MODEL MMPROJ FILE OUT.f32\n", argv[0]); return 2; }
    session s;
    open_session(s, argv[2], argv[3], false);
    mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_file(s.mtmd, argv[4], false, mtmd_helper_init_opt_default());
    if (!w.bitmap || !mtmd_bitmap_is_audio(w.bitmap)) { fprintf(stderr, "cannot decode %s as audio\n", argv[4]); return 1; }
    const size_t n = mtmd_bitmap_get_n_bytes(w.bitmap) / sizeof(float);
    write_f32(argv[5], (const float *) mtmd_bitmap_get_data(w.bitmap), n);
    printf("%s: %zu samples at %d Hz\n", argv[4], n, mtmd_get_audio_sample_rate(s.mtmd));
    mtmd_bitmap_free(w.bitmap);
    mtmd_free(s.mtmd);
    llama_model_free(s.model);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc >= 2 && !strcmp(argv[1], "dump")) return run_dump(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "decode")) return run_decode(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "asr")) return run_asr(argc, argv);
    fprintf(stderr, "usage: %s dump|decode|asr ...\n", argv[0]);
    return 2;
}
