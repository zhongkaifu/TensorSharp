// LFM3-Omni vision reference dumps from llama.cpp's mtmd (libmtmd + libllama), for TensorSharp's
// Lfm3VisionEncoder parity checks (eng/lfm3/vision-reference.sh builds and runs it; vision-compare.py reads
// both sides' dumps).
//
// For each case it runs llama.cpp's own preprocessing, tokenization and encode, and writes to <out dir>:
//   <name>.json        the chunk layout (text chunks: token ids and text; image chunks: rows, positions, graph
//                      input values), the frames a video case read, and encode times: each image chunk is encoded
//                      once with the input capture, then --reps times plainly (median), as an application runs it
//   <name>.pixels.f32  every image chunk's graph input "inp_raw" as llama.cpp fed it (planar [3][H][W] per
//                      frame, frames of a pair one after the other), captured with the scheduler callback
//   <name>.embd.f32    every image chunk's projector rows, [rows][2048] each, chunk after chunk
//
// Cases (one argument each):
//   synth:NAME:W:H:SEED  the synthetic RGB image of Lfm3VisionSynthetic (TensorSharp test project); the same
//                        integer formula is written below, so both sides start from identical bytes
//   file:NAME:PATH       an image file, decoded by mtmd's helper (stb_image, 3 channels: alpha dropped)
//   video:NAME:PATH      a video through mtmd's built-in ffmpeg decoder (frames scaled by swscale bicubic)
//   frames:NAME:DIR      a video whose sampled frames TensorSharp decoded and resized (DIR/meta.txt:
//                        "src_w src_h fps n_frames duration out_w out_h"; DIR/f<index>.rgb: the RGB24 frame of
//                        source frame <index> at out_w x out_h); served by an application decoder, so
//                        llama.cpp's frame selection, frame size, timestamps and pairing run on TensorSharp's
//                        pixels (a frame llama.cpp asks for that TensorSharp did not select is an error)
//   synthvideo:NAME:W:H:FRAMES:FPS  a clip of FRAMES synthetic W x H frames at FPS (frame i is the synthetic image
//                        with seed 1000 + i), served at its own size by an application decoder: W x H must be the
//                        size llama.cpp picks for the sampled frames, so no resampler touches them
//
// usage: lfm3-vision-ref <LM gguf> <mmproj gguf> <out dir> [--cpu] [--reps N] [--threads N] case...
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ------------------------------------------------------------------ the synthetic image (keep in sync with C#)

static std::vector<uint8_t> synth_rgb(int w, int h, uint32_t seed) {
    std::vector<uint8_t> rgb((size_t) w * h * 3);
    const uint32_t cx = (uint32_t) w / 3, cy = (uint32_t) (2 * h) / 5;
    for (uint32_t y = 0; y < (uint32_t) h; y++) {
        for (uint32_t x = 0; x < (uint32_t) w; x++) {
            uint32_t dx = x > cx ? x - cx : cx - x, dy = y > cy ? y - cy : cy - y;
            uint32_t ring = ((dx * dx + dy * dy) / (uint32_t) (w / 16 + 1)) & 63u;
            for (uint32_t c = 0; c < 3; c++) {
                uint32_t k = x * 73856093u ^ y * 19349663u ^ (seed * 83492791u + c * 2654435761u);
                k ^= k >> 13;
                k *= 0x5bd1e995u;
                k ^= k >> 15;
                uint32_t base = (x * (c + 1) * 160u) / (uint32_t) w + (y * (3 - c) * 120u) / (uint32_t) h;
                uint32_t block = ((x / 24u + y / 24u + c) & 1u) * 40u;
                rgb[((size_t) y * w + x) * 3 + c] = (uint8_t) ((base + ring + block + (k >> 27)) & 255u);
            }
        }
    }
    return rgb;
}

// ------------------------------------------------------------------ graph input capture

struct capture_state {
    bool on = false;
    std::vector<float> pixels; // the last captured inp_raw
};

static ggml_tensor * inp_raw_of(ggml_tensor * t) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        ggml_tensor * s = t->src[i];
        if (!s) continue;
        ggml_tensor * base = s->view_src ? s->view_src : s;
        if (strstr(base->name, "inp_raw")) return base;
    }
    return nullptr;
}

static bool eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = (capture_state *) user_data;
    if (!cap->on) return false;
    ggml_tensor * raw = inp_raw_of(t);
    if (!raw) return false;
    if (ask) return true;
    cap->pixels.resize(ggml_nelements(raw));
    ggml_backend_tensor_get(raw, cap->pixels.data(), 0, ggml_nbytes(raw));
    cap->on = false; // the first consumer is enough
    return true;
}

// ------------------------------------------------------------------ application decoder for frames:NAME:DIR

struct frames_dir {
    std::string dir;
    bool synth = false;                 // synthvideo: frames from synth_rgb, no files
    int src_w = 0, src_h = 0, n_frames = 0, out_w = 0, out_h = 0;
    float fps = 0, duration = 0;
    int next = 0;                       // next source frame index
    uint32_t want_w = 0, want_h = 0;    // the size llama.cpp asked for
    std::vector<int> served;            // source frames llama.cpp read
    bool failed = false;
};

static void * fd_open(void * user_data, const char *, const unsigned char *, size_t, mtmd_helper_video_stream_info * info) {
    auto * fd = (frames_dir *) user_data;
    if (!fd->synth) {
        std::ifstream meta(fd->dir + "/meta.txt");
        if (!(meta >> fd->src_w >> fd->src_h >> fd->fps >> fd->n_frames >> fd->duration >> fd->out_w >> fd->out_h)) {
            fprintf(stderr, "frames: cannot read %s/meta.txt\n", fd->dir.c_str());
            return nullptr;
        }
    }
    info->width = fd->src_w;
    info->height = fd->src_h;
    info->fps = fd->fps;
    info->n_frames = fd->n_frames;
    info->duration = fd->duration;
    fd->next = 0;
    return fd;
}

static bool fd_start(void * state, uint32_t width, uint32_t height, float, bool * fps_picked) {
    auto * fd = (frames_dir *) state;
    fd->want_w = width;
    fd->want_h = height;
    if (fps_picked) *fps_picked = false;
    if ((int) width != fd->out_w || (int) height != fd->out_h) {
        fprintf(stderr, "frames: llama.cpp wants %ux%u frames, TensorSharp resized to %dx%d\n", width, height, fd->out_w, fd->out_h);
        fd->failed = true;
    }
    return true;
}

static bool fd_read_frame(void * state, unsigned char * rgb, uint32_t width, uint32_t height) {
    auto * fd = (frames_dir *) state;
    if (fd->next >= fd->n_frames) return false;
    const int index = fd->next++;
    if (!rgb) return true; // skipped
    if (fd->synth) {
        std::vector<uint8_t> frame = synth_rgb((int) width, (int) height, 1000u + (uint32_t) index);
        memcpy(rgb, frame.data(), frame.size());
        fd->served.push_back(index);
        return true;
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/f%d.rgb", fd->dir.c_str(), index);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "frames: llama.cpp reads source frame %d, which TensorSharp did not select\n", index);
        fd->failed = true;
        return false;
    }
    const size_t n = (size_t) width * height * 3;
    f.read((char *) rgb, (std::streamsize) n);
    if ((size_t) f.gcount() != n) {
        fprintf(stderr, "frames: %s is not %ux%u RGB24\n", path, width, height);
        fd->failed = true;
        return false;
    }
    fd->served.push_back(index);
    return true;
}

static void fd_close(void *) {}

// ------------------------------------------------------------------ output

static std::string json_escape(const std::string & s) {
    std::string o;
    for (unsigned char ch : s) {
        if (ch == '"' || ch == '\\') { o += '\\'; o += (char) ch; }
        else if (ch == '\n') o += "\\n";
        else if (ch < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", ch); o += b; }
        else o += (char) ch;
    }
    return o;
}

static bool write_f32(const std::string & path, const std::vector<float> & v) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
    return true;
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static std::vector<std::string> split(const std::string & s, char sep, int max_parts) {
    std::vector<std::string> out;
    size_t start = 0;
    if (max_parts <= 0) max_parts = 1 << 20;
    while ((int) out.size() + 1 < max_parts) {
        size_t p = s.find(sep, start);
        if (p == std::string::npos) break;
        out.push_back(s.substr(start, p - start));
        start = p + 1;
    }
    out.push_back(s.substr(start));
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <LM gguf> <mmproj gguf> <out dir> [--cpu] [--reps N] [--threads N] case...\n", argv[0]);
        return 2;
    }
    const std::string lm_path = argv[1], mmproj_path = argv[2], out_dir = argv[3];
    bool use_gpu = true;
    int reps = 3, threads = 8;
    std::vector<std::string> cases;
    for (int i = 4; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--cpu") use_gpu = false;
        else if (a == "--reps" && i + 1 < argc) reps = std::max(1, atoi(argv[++i]));
        else if (a == "--threads" && i + 1 < argc) threads = std::max(1, atoi(argv[++i]));
        else cases.push_back(a);
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(lm_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "cannot load %s\n", lm_path.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    capture_state cap;
    mtmd_context_params cparams = mtmd_context_params_default();
    cparams.use_gpu = use_gpu;
    cparams.print_timings = false;
    cparams.n_threads = threads;
    cparams.warmup = true;
    cparams.cb_eval = eval_cb;
    cparams.cb_eval_user_data = &cap;
    mtmd_context * mctx = mtmd_init_from_file(mmproj_path.c_str(), model, cparams);
    if (!mctx) { fprintf(stderr, "cannot load %s\n", mmproj_path.c_str()); return 1; }
    const int n_embd = 2048;

    int failures = 0;
    for (const std::string & spec : cases) {
        std::vector<std::string> p = split(spec, ':', spec.rfind("synthvideo:", 0) == 0 ? 6 : 5);
        if (p.size() < 3) { fprintf(stderr, "bad case %s\n", spec.c_str()); failures++; continue; }
        const std::string kind = p[0], name = p[1];
        mtmd::bitmap bmp;
        mtmd_helper_video * vctx = nullptr;
        frames_dir fd;
        mtmd_helper_video_decoder dec{};
        std::ostringstream extra;
        if (kind == "synth" && p.size() == 5) {
            int w = atoi(p[2].c_str()), h = atoi(p[3].c_str());
            uint32_t seed = (uint32_t) strtoul(p[4].c_str(), nullptr, 10);
            std::vector<uint8_t> rgb = synth_rgb(w, h, seed);
            bmp.ptr.reset(mtmd_bitmap_init(w, h, rgb.data()));
            extra << "\"source\":[" << w << "," << h << "],";
        } else if (kind == "file") {
            const std::string path = spec.substr(kind.size() + name.size() + 2);
            auto res = mtmd_helper_bitmap_init_from_file(mctx, path.c_str(), false, mtmd_helper_init_opt_default());
            bmp.ptr.reset(res.bitmap);
            if (res.video_ctx) mtmd_helper_video_free(res.video_ctx);
            if (bmp.ptr) extra << "\"source\":[" << mtmd_bitmap_get_nx(bmp.ptr.get()) << "," << mtmd_bitmap_get_ny(bmp.ptr.get()) << "],";
        } else if (kind == "video" || kind == "frames" || (kind == "synthvideo" && p.size() == 6)) {
            const std::string path = kind == "synthvideo" ? std::string("synthetic") : spec.substr(kind.size() + name.size() + 2);
            mtmd_helper_init_opt opt = mtmd_helper_init_opt_default();
            opt.video_params.no_soundtrack = true; // vision only: the soundtrack rows are the audio port's
            if (kind == "synthvideo") {
                fd.synth = true;
                fd.src_w = fd.out_w = atoi(p[2].c_str());
                fd.src_h = fd.out_h = atoi(p[3].c_str());
                fd.n_frames = atoi(p[4].c_str());
                fd.fps = (float) atof(p[5].c_str());
                fd.duration = fd.n_frames / fd.fps;
            }
            if (kind == "frames" || kind == "synthvideo") {
                fd.dir = path;
                dec.open = fd_open;
                dec.start = fd_start;
                dec.read_frame = fd_read_frame;
                dec.read_audio = nullptr;
                dec.close = fd_close;
                dec.user_data = &fd;
                opt.video_params.decoder = &dec;
            }
            vctx = mtmd_helper_video_init(mctx, path.c_str(), opt.video_params);
            if (vctx) {
                mtmd_helper_video_info vi = mtmd_helper_video_get_info(vctx);
                extra << "\"video\":{\"width\":" << vi.width << ",\"height\":" << vi.height << ",\"fps\":" << vi.fps
                      << ",\"n_frames\":" << vi.n_frames << "},";
                bmp.ptr.reset(mtmd_bitmap_init_lazy(mctx, nullptr, vctx,
                    [](size_t, void * ud, mtmd_bitmap ** out_bitmap, char ** out_text) -> int {
                        return mtmd_helper_video_read_next((mtmd_helper_video *) ud, out_bitmap, out_text);
                    }));
                mtmd_bitmap_set_is_video(bmp.ptr.get(), true);
            }
        } else {
            fprintf(stderr, "bad case %s\n", spec.c_str());
            failures++;
            continue;
        }
        if (!bmp.ptr) {
            fprintf(stderr, "%s: no bitmap\n", name.c_str());
            failures++;
            if (vctx) mtmd_helper_video_free(vctx);
            continue;
        }

        mtmd_input_text text;
        text.text = mtmd_default_marker();
        text.text_len = strlen(text.text);
        text.add_special = false;
        text.parse_special = true;
        mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
        const mtmd_bitmap * bitmaps[1] = { bmp.ptr.get() };
        int32_t tok = mtmd_tokenize(mctx, chunks.get(), &text, bitmaps, 1);
        if (vctx) mtmd_helper_video_free(vctx);
        if (tok != 0 || fd.failed) {
            fprintf(stderr, "%s: tokenize failed (%d)%s\n", name.c_str(), tok, fd.failed ? ", frames mismatch" : "");
            failures++;
            continue;
        }

        std::vector<float> all_pixels, all_embd;
        std::ostringstream js;
        js << "{\"name\":\"" << json_escape(name) << "\",\"case\":\"" << json_escape(spec) << "\",\"backend\":\""
           << (use_gpu ? "gpu" : "cpu") << "\"," << extra.str();
        if (!fd.served.empty()) {
            js << "\"served_frames\":[";
            for (size_t i = 0; i < fd.served.size(); i++) js << (i ? "," : "") << fd.served[i];
            js << "],";
        }
        js << "\"chunks\":[";
        double total_ms = 0;
        const size_t n_chunks = mtmd_input_chunks_size(chunks.get());
        for (size_t ci = 0; ci < n_chunks; ci++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.get(), ci);
            if (ci) js << ",";
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n = 0;
                const llama_token * toks = mtmd_input_chunk_get_tokens_text(chunk, &n);
                js << "{\"type\":\"text\",\"tokens\":[";
                std::string piece_text;
                for (size_t i = 0; i < n; i++) {
                    js << (i ? "," : "") << toks[i];
                    char buf[256];
                    int m = llama_token_to_piece(vocab, toks[i], buf, sizeof(buf), 0, true);
                    if (m > 0) piece_text.append(buf, m);
                }
                js << "],\"text\":\"" << json_escape(piece_text) << "\"}";
                continue;
            }
            const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(chunk);
            const size_t rows = mtmd_input_chunk_get_n_tokens(chunk);
            // capture the graph input once, then time plain encodes (no callback stops, as an application runs it)
            cap.on = true;
            cap.pixels.clear();
            if (mtmd_encode_chunk(mctx, chunk) != 0) { fprintf(stderr, "%s: encode failed\n", name.c_str()); failures++; break; }
            cap.on = false;
            std::vector<double> ms;
            for (int r = 0; r < reps; r++) {
                auto t0 = std::chrono::steady_clock::now();
                if (mtmd_encode_chunk(mctx, chunk) != 0) { fprintf(stderr, "%s: encode failed\n", name.c_str()); failures++; break; }
                ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            const float * embd = mtmd_get_output_embd(mctx);
            all_embd.insert(all_embd.end(), embd, embd + rows * n_embd);
            all_pixels.insert(all_pixels.end(), cap.pixels.begin(), cap.pixels.end());
            double med = median(ms);
            total_ms += med;
            js << "{\"type\":\"image\",\"rows\":" << rows << ",\"n_pos\":" << mtmd_image_tokens_get_n_pos(it) << ",\"pixels\":" << cap.pixels.size() << ",\"encode_ms\":" << med << ",\"encode_ms_all\":[";
            for (size_t i = 0; i < ms.size(); i++) js << (i ? "," : "") << ms[i];
            js << "]}";
        }
        js << "],\"encode_ms_total\":" << total_ms << "}\n";
        std::ofstream(out_dir + "/" + name + ".json") << js.str();
        write_f32(out_dir + "/" + name + ".pixels.f32", all_pixels);
        write_f32(out_dir + "/" + name + ".embd.f32", all_embd);
        fprintf(stderr, "%s: %zu chunks, %zu rows, encode %.1f ms (median of %d)\n", name.c_str(), n_chunks,
                all_embd.size() / n_embd, total_ms, reps);
    }

    mtmd_free(mctx);
    llama_model_free(model);
    llama_backend_free();
    return failures ? 1 : 0;
}
