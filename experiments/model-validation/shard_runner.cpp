// Private experiment against llama.cpp 71255fe. Binary tensor payloads use little-endian IEEE754.
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task_info.h>
#endif

static void read_exact(void * dst, size_t size) {
    if (fread(dst, 1, size, stdin) != size) throw std::runtime_error("truncated request");
}

static uint32_t read_be32() {
    unsigned char data[4];
    read_exact(data, 4);
    return (uint32_t(data[0]) << 24) | (uint32_t(data[1]) << 16) |
           (uint32_t(data[2]) << 8) | uint32_t(data[3]);
}

static uint64_t allocated(llama_context * ctx) {
    uint64_t total = 0;
    for (const auto & item : llama_get_memory_breakdown(ctx)) total += item.second.total();
    return total;
}

static uint64_t max_rss_bytes() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("getrusage failed");
#ifdef __APPLE__
    return usage.ru_maxrss;
#else
    return uint64_t(usage.ru_maxrss) * 1024;
#endif
}

static std::string vm_metrics() {
#ifdef __APPLE__
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS ||
            count < TASK_VM_INFO_REV3_COUNT) throw std::runtime_error("task VM metrics unavailable");
    return "{\"resident_bytes\":" + std::to_string(info.resident_size) +
           ",\"physical_footprint_bytes\":" + std::to_string(info.phys_footprint) +
           ",\"physical_footprint_peak_bytes\":" + std::to_string(info.ledger_phys_footprint_peak) +
           ",\"compressed_bytes\":" + std::to_string(info.compressed) + "}";
#else
    return "null";
#endif
}

int main(int argc, char ** argv) {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    try {
        if (argc != 9) throw std::runtime_error("usage: runner MODEL BEGIN END CTX GPU_LAYERS BUDGET_BYTES PROMPT RETURN_LOGITS");
        const int begin = std::stoi(argv[2]), end = std::stoi(argv[3]);
        const int context = std::stoi(argv[4]), gpu = std::stoi(argv[5]);
        const uint64_t budget = std::stoull(argv[6]);
        const bool return_logits = std::stoi(argv[8]) != 0;
        if (begin < 0 || end <= begin || context < 32 || context > 4096 || budget == 0) {
            throw std::runtime_error("invalid partition/context/budget");
        }
        const uint16_t endian = 1;
        if (*reinterpret_cast<const unsigned char *>(&endian) != 1 || sizeof(float) != 4 ||
                !std::numeric_limits<float>::is_iec559) throw std::runtime_error("unsupported float encoding");
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
#ifdef LLAMA_MESH_SHARD_PARAMS
        mp.shard_begin = begin;
        mp.shard_end = end;
#else
        const std::string range = std::to_string(begin) + ":" + std::to_string(end);
        setenv("MESHTHING_SHARD", range.c_str(), 1);
#endif
        mp.n_gpu_layers = gpu;
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
        model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("model load failed");
        const int layers = llama_model_n_layer(model);
        const int embd = llama_model_n_embd(model);
        const auto * vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        const bool prefix = end < layers;
        if (begin != 0 && prefix) throw std::runtime_error("prototype supports prefix/suffix only");
        if (embd <= 0 || embd > 32768) throw std::runtime_error("unsupported hidden size");
        char source_hash[65] = {};
        const int source_len = llama_model_meta_val_str(model, "meshthing.source_sha256", source_hash, sizeof(source_hash));
        if (source_len >= 0 && source_len != 64) throw std::runtime_error("invalid shard source identity");
        auto cp = llama_context_default_params();
        cp.n_ctx = context;
        cp.n_batch = cp.n_ubatch = 128;
        cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 4;
        cp.embeddings = prefix;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("context init failed");
        uint64_t peak_allocated = allocated(ctx);
        if (peak_allocated > budget) throw std::runtime_error("observed runtime buffers exceed budget");
        std::vector<llama_token> prompt(strlen(argv[7]) + 32);
        const int nt = llama_tokenize(vocab, argv[7], strlen(argv[7]), prompt.data(), prompt.size(), true, true);
        if (nt <= 0) throw std::runtime_error("tokenize failed");
        prompt.resize(nt);
        printf("{\"protocol\":1,\"begin\":%d,\"end\":%d,\"layers\":%d,\"embd\":%d,\"vocab\":%d,\"allocated_bytes\":%llu,\"max_rss_bytes\":%llu,\"vm\":%s,\"source_sha256\":\"%s\",\"tokens\":[",
               begin, end, layers, embd, n_vocab, (unsigned long long)peak_allocated, (unsigned long long)max_rss_bytes(), vm_metrics().c_str(), source_len >= 0 ? source_hash : "");
        for (int i = 0; i < nt; ++i) printf("%s%d", i ? "," : "", prompt[i]);
        printf("]}\n"); fflush(stdout);
        uint32_t expected_step = 0, expected_pos = 0;
        for (;;) {
            int c = fgetc(stdin);
            if (c == EOF) break;
            ungetc(c, stdin);
            const uint32_t magic = read_be32(), op = read_be32(), step = read_be32();
            const uint32_t pos = read_be32(), n = read_be32();
            if (magic != 0x4d534831 || step != expected_step || pos != expected_pos ||
                    n == 0 || n > 128 || pos + n > uint32_t(context) || op != (begin == 0 ? 1u : 2u)) {
                throw std::runtime_error("invalid/out-of-order execution request; session closed");
            }
            std::vector<llama_token> tokens;
            std::vector<float> hidden;
            auto batch = llama_batch_init(n, begin == 0 ? 0 : embd, 1);
            batch.n_tokens = n;
            if (begin == 0) {
                tokens.resize(n);
                for (uint32_t i = 0; i < n; ++i) {
                    tokens[i] = read_be32();
                    if (tokens[i] < 0 || tokens[i] >= n_vocab) throw std::runtime_error("invalid token");
                    batch.token[i] = tokens[i];
                }
            } else {
                hidden.resize(size_t(n)*embd);
                read_exact(hidden.data(), hidden.size()*sizeof(float));
                for (float value : hidden) if (!std::isfinite(value)) throw std::runtime_error("nonfinite activation");
                memcpy(batch.embd, hidden.data(), hidden.size()*sizeof(float));
            }
            for (uint32_t i = 0; i < n; ++i) {
                batch.pos[i] = pos + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = prefix || i == n - 1;
            }
            const auto start = std::chrono::steady_clock::now();
            const int status = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (status != 0) throw std::runtime_error("decode failed; session closed");
            llama_synchronize(ctx);
            const double compute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            peak_allocated = std::max(peak_allocated, allocated(ctx));
            if (peak_allocated > budget) throw std::runtime_error("observed runtime buffers exceed budget");
            float * result = prefix ? llama_get_embeddings(ctx) : llama_get_logits_ith(ctx, -1);
            if (!result) throw std::runtime_error("missing partition result");
            const size_t values = prefix ? size_t(n)*embd : (return_logits ? n_vocab : 0);
            int token = -1;
            bool eog = false;
            std::string piece;
            if (!prefix) {
                for (int i = 0; i < n_vocab; ++i) if (!std::isfinite(result[i])) throw std::runtime_error("nonfinite logits");
                token = std::max_element(result, result + n_vocab) - result;
                eog = llama_vocab_is_eog(vocab, token);
                char bytes[4096];
                const int len = llama_token_to_piece(vocab, token, bytes, sizeof(bytes), 0, false);
                if (len < 0) throw std::runtime_error("token piece exceeds bound");
                const char * hex = "0123456789abcdef";
                for (int i = 0; i < len; ++i) { unsigned char b = bytes[i]; piece += hex[b >> 4]; piece += hex[b & 15]; }
            }
            printf("{\"step\":%u,\"n_floats\":%zu,\"token\":%d,\"eog\":%s,\"piece_hex\":\"%s\",\"compute_ms\":%.6f,\"observed_peak_buffer_bytes\":%llu,\"max_rss_bytes\":%llu,\"vm\":%s}\n",
                   step, values, token, eog ? "true" : "false", piece.c_str(), compute_ms,
                   (unsigned long long)peak_allocated, (unsigned long long)max_rss_bytes(), vm_metrics().c_str());
            if (values && fwrite(result, sizeof(float), values, stdout) != values) throw std::runtime_error("response write failed");
            if (fflush(stdout) != 0) throw std::runtime_error("response flush failed");
            expected_step++; expected_pos += n;
        }
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "shard runner: %s\n", error.what());
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        return 1;
    }
}
