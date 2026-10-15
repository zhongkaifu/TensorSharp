// TensorSharp-owned diagnostic. Uses unchanged llama.cpp public C APIs only.
// It never creates a llama_context or executes the language model.
#include "llama.h"
#include "mtmd.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
static void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
static std::string read(const fs::path & p, std::size_t limit) {
    const auto bytes = fs::file_size(p);
    require(bytes > 0 && bytes <= limit, "Input size outside bounded diagnostic limits");
    std::ifstream in(p, std::ios::binary); std::string data(static_cast<std::size_t>(bytes), '\0');
    in.read(data.data(), static_cast<std::streamsize>(bytes)); require(bool(in), "Input read failed"); return data;
}
static void write(const fs::path & p, const void * data, std::size_t bytes) {
    std::ofstream out(p, std::ios::binary); out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    require(bool(out), "Output write failed");
}
static int number(const std::map<std::string, std::string> & args, const std::string & key, int maximum) {
    std::size_t used = 0; const int value = std::stoi(args.at(key), &used);
    require(used == args.at(key).size() && value > 0 && value <= maximum, "Invalid numeric option"); return value;
}
template<class T> static void json_array(std::ostream & out, const std::vector<T> & values) {
    out << '['; for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; } out << ']';
}
static int self_test() {
    // These check only the owned format/finite primitives, not any model or device.
    require(std::isfinite(1.0f) && !std::isfinite(std::numeric_limits<float>::quiet_NaN()), "Finite primitive failed");
    const std::vector<int> thw = {4,4,4, 4,4,5, 4,5,4};
    require(thw.size() == 9 && thw[7] == 5, "THW row-major layout failed");
    std::cout << "{\"ModelIndependentChecks\":2,\"ModelOrDeviceValidated\":false}\n"; return 0;
}
int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") return self_test();
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            require(i + 1 < argc && std::string(argv[i]).rfind("--", 0) == 0, "Require --name value pairs");
            require(args.emplace(std::string(argv[i]).substr(2), argv[i + 1]).second, "Duplicate option");
        }
        const std::vector<std::string> names = {"model","mmproj","rgb","width","height","prompt","output","runtime-dir",
            "rows","dimension","prompt-tokens","image-token","threads","gpu"};
        require(args.size() == names.size(), "Missing or unknown option");
        for (const auto & name : names) require(args.count(name) == 1, "Missing required option");
        const int width = number(args, "width", 8192), height = number(args, "height", 8192);
        const int expected_rows = number(args, "rows", 32768), expected_dim = number(args, "dimension", 32768);
        const int expected_tokens = number(args, "prompt-tokens", 65536), image_token = number(args, "image-token", 1048576);
        const int threads = number(args, "threads", 256);
        require(args.at("gpu") == "0" || args.at("gpu") == "1", "gpu must be 0 or 1");
        const bool gpu = args.at("gpu") == "1";
        require(std::uint64_t(width) * height * 3 <= 64u * 1024u * 1024u, "RGB input too large");
        auto rgb = read(args.at("rgb"), 64u * 1024u * 1024u);
        require(rgb.size() == std::size_t(width) * height * 3, "RGB byte length mismatch");
        auto prompt = read(args.at("prompt"), 1u << 20);
        const std::string marker = "<__media__>";
        require(prompt.find(marker) != std::string::npos && prompt.find(marker, prompt.find(marker) + marker.size()) == std::string::npos,
            "Exactly one canonical media marker required");
        const fs::path output = args.at("output"); require(!fs::exists(output), "Fresh output directory required");
        fs::create_directories(output);

        ggml_backend_load_all_from_path(args.at("runtime-dir").c_str());
        llama_backend_init();
        struct backend_cleanup { ~backend_cleanup() { llama_backend_free(); } } cleanup_backend;
        // vocab_only returns from load_hparams before n_embd is populated. The
        // unchanged upstream no_alloc path instead constructs tensor metadata
        // and zero-byte dummy buffers, then returns before load_all_data. NONE
        // plus lazy OFF also prevents init_mappings from mmap/prefetching LM
        // weights. No llama_context is ever created by this diagnostic.
        auto params = llama_model_default_params(); params.vocab_only = false; params.no_alloc = true;
        params.n_gpu_layers = 0; params.load_mode = LLAMA_LOAD_MODE_NONE; params.lazy_mode = LLAMA_LAZY_MODE_OFF;
        params.use_extra_bufts = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(args.at("model").c_str(), params), &llama_model_free);
        require(model != nullptr, "Metadata-only no-alloc model load failed");
        const int dimension = llama_model_n_embd_inp(model.get());
        if (dimension != expected_dim) throw std::runtime_error("Projection width mismatch: actual="
            + std::to_string(dimension) + " expected=" + std::to_string(expected_dim));
        auto vision_params = mtmd_context_params_default();
        vision_params.use_gpu = gpu; vision_params.warmup = false; vision_params.print_timings = true;
        vision_params.n_threads = threads; vision_params.media_marker = marker.c_str(); vision_params.image_marker = nullptr;
        vision_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        vision_params.device = ggml_backend_dev_by_type(gpu ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU);
        require(vision_params.device != nullptr, "Requested backend device unavailable");
        std::unique_ptr<mtmd_context, decltype(&mtmd_free)> vision(
            mtmd_init_from_file(args.at("mmproj").c_str(), model.get(), vision_params), &mtmd_free);
        require(vision != nullptr && mtmd_support_vision(vision.get()) && mtmd_decode_use_mrope(vision.get()), "Expected MRoPE vision encoder");
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(
            mtmd_bitmap_init(width, height, reinterpret_cast<const unsigned char *>(rgb.data())), &mtmd_bitmap_free);
        require(bitmap != nullptr, "Bitmap creation failed");
        std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(mtmd_input_chunks_init(), &mtmd_input_chunks_free);
        require(chunks != nullptr, "Chunk allocation failed");
        const mtmd_bitmap * bitmap_pointer = bitmap.get();
        const mtmd_input_text input{prompt.data(), prompt.size(), true, true};
        require(mtmd_tokenize(vision.get(), chunks.get(), &input, &bitmap_pointer, 1) == 0, "MTMD tokenization failed");
        std::vector<int> expanded, positions, image_positions;
        std::vector<float> embeddings;
        llama_pos next_position = 0; int image_chunks = 0, image_start = -1, rows = 0;
        double encode_ms = 0;
        for (std::size_t c = 0; c < mtmd_input_chunks_size(chunks.get()); ++c) {
            const auto * chunk = mtmd_input_chunks_get(chunks.get(), c);
            const auto type = mtmd_input_chunk_get_type(chunk);
            if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                std::size_t count = 0; const auto * ids = mtmd_input_chunk_get_tokens_text(chunk, &count);
                require(ids != nullptr && count <= 65536, "Invalid text chunk");
                for (std::size_t i = 0; i < count; ++i) {
                    expanded.push_back(ids[i]); positions.insert(positions.end(), {next_position,next_position,next_position}); ++next_position;
                }
            } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                require(++image_chunks == 1, "Exactly one image chunk required");
                const auto * image = mtmd_input_chunk_get_tokens_image(chunk); require(image != nullptr, "Image token metadata missing");
                rows = static_cast<int>(mtmd_input_chunk_get_n_tokens(chunk)); require(rows == expected_rows, "Image row count mismatch");
                image_start = static_cast<int>(expanded.size());
                for (int i = 0; i < rows; ++i) {
                    const auto p = mtmd_image_tokens_get_decoder_pos(image, next_position, static_cast<std::size_t>(i));
                    require(p.t <= 65536 && p.y <= 65536 && p.x <= 65536 && p.z == 0, "Unexpected MRoPE position");
                    const std::vector<int> thw = {static_cast<int>(p.t),static_cast<int>(p.y),static_cast<int>(p.x)};
                    positions.insert(positions.end(), thw.begin(), thw.end()); image_positions.insert(image_positions.end(), thw.begin(), thw.end());
                    expanded.push_back(image_token);
                }
                const auto start = std::chrono::steady_clock::now();
                require(mtmd_encode_chunk(vision.get(), chunk) == 0, "MTMD image encode failed");
                encode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                const float * projected = mtmd_get_output_embd(vision.get()); require(projected != nullptr, "Projected embedding pointer missing");
                require(std::uint64_t(rows) * dimension * sizeof(float) <= 64u * 1024u * 1024u, "Projection exceeds diagnostic limit");
                embeddings.assign(projected, projected + std::size_t(rows) * dimension);
                next_position += mtmd_input_chunk_get_n_pos(chunk);
            } else throw std::runtime_error("Only text and one still image are supported");
        }
        require(image_chunks == 1 && int(expanded.size()) == expected_tokens && positions.size() == 3 * expanded.size(), "Expanded prompt geometry mismatch");
        require(std::all_of(embeddings.begin(), embeddings.end(), [](float value) { return std::isfinite(value); }), "Nonfinite projected embedding");
        write(output / "projected.f32", embeddings.data(), embeddings.size() * sizeof(float));
        { std::ofstream out(output / "expanded-ids.json"); json_array(out, expanded); require(bool(out), "Expanded-ID write failed"); }
        { std::ofstream out(output / "positions-thw.json"); json_array(out, positions); require(bool(out), "Position write failed"); }
        { std::ofstream out(output / "image-positions-thw.json"); json_array(out, image_positions); require(bool(out), "Image position write failed"); }
        double sumsq = 0; for (const float value : embeddings) sumsq += double(value) * value;
        std::ofstream out(output / "native-result.json");
        out << "{\"Schema\":\"qwen-mtmd-native-capture.v2\",\"Completed\":true,\"VocabularyOnly\":false,\"ModelNoAlloc\":true,"
            << "\"ModelLoadMode\":\"none\",\"ModelLazyMode\":\"off\",\"LanguageModelExecuted\":false,\"GpuRequested\":" << (gpu ? "true" : "false")
            << ",\"FlashAttentionRequested\":true,\"Warmup\":false,\"ImageRows\":" << rows << ",\"Dimension\":" << dimension
            << ",\"ImageStartToken\":" << image_start << ",\"ExpandedPromptTokens\":" << expanded.size()
            << ",\"NextRopePosition\":" << next_position << ",\"AllFinite\":true,\"EmbeddingL2\":" << std::sqrt(sumsq)
            << ",\"EncodeMs\":" << encode_ms << ",\"PerformanceQualified\":false,\"AnswerQualityQualified\":false}\n";
        require(bool(out), "Result write failed"); return 0;
    } catch (const std::exception & error) { std::cerr << "QwenMtmdEmbeddingProbe: " << error.what() << '\n'; return 1; }
}
