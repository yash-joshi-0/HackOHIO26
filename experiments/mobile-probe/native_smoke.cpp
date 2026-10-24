#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef LLAMA_MESH_SHARD_PARAMS
#error Build with the private per-model shard API header and matching library.
#endif

class Shard {
public:
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    bool prefix;
    int position = 0;

    Shard(const char * path, int begin, int end, bool prefix) : prefix(prefix) {
        auto mp = llama_model_default_params();
        mp.shard_begin = begin;
        mp.shard_end = end;
        mp.n_gpu_layers = 99;
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
        model = llama_model_load_from_file(path, mp);
        if (!model) throw std::runtime_error("model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 128;
        cp.n_batch = cp.n_ubatch = 32;
        cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 4;
        cp.embeddings = prefix;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context = llama_init_from_model(model, cp);
        if (!context) {
            llama_model_free(model);
            model = nullptr;
            throw std::runtime_error("context initialization failed");
        }
    }

    Shard(const Shard &) = delete;
    Shard & operator=(const Shard &) = delete;

    ~Shard() {
        if (context) llama_free(context);
        if (model) llama_model_free(model);
    }

    std::vector<float> execute(const std::vector<llama_token> & tokens, const std::vector<float> * hidden = nullptr) {
        const int n = int(tokens.size());
        const int dim = llama_model_n_embd(model);
        if (n < 1 || n > 32 || position + n > 128 || (hidden && hidden->size() != size_t(n)*dim)) {
            throw std::runtime_error("invalid smoke-test batch");
        }
        auto batch = llama_batch_init(n, hidden ? dim : 0, 1);
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            if (!hidden) batch.token[i] = tokens[i];
            batch.pos[i] = position + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = prefix || i == n - 1;
        }
        if (hidden) memcpy(batch.embd, hidden->data(), hidden->size()*sizeof(float));
        const int status = llama_decode(context, batch);
        llama_batch_free(batch);
        if (status) throw std::runtime_error("decode failed");
        llama_synchronize(context);
        position += n;
        float * data = prefix ? llama_get_embeddings(context) : llama_get_logits_ith(context, -1);
        const size_t count = prefix ? size_t(n)*dim : llama_vocab_n_tokens(llama_model_get_vocab(model));
        if (!data) throw std::runtime_error("missing output");
        return {data, data + count};
    }

    void reset() {
        llama_memory_clear(llama_get_memory(context), true);
        position = 0;
    }

    size_t buffer_bytes() const {
        size_t total = 0;
        for (const auto & item : llama_get_memory_breakdown(context)) total += item.second.total();
        return total;
    }
};

static float compare(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) throw std::runtime_error("logit shape mismatch");
    float error = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("nonfinite logits");
        error = std::max(error, std::abs(a[i] - b[i]));
    }
    if (error > 0.02f || std::max_element(a.begin(), a.end()) - a.begin() !=
                         std::max_element(b.begin(), b.end()) - b.begin()) {
        throw std::runtime_error("partition equivalence failed");
    }
    return error;
}

extern "C" int meshthing_native_smoke(const char * model_path, const char * report_path) {
    try {
        ggml_backend_load_all();
        llama_backend_init();
        Shard reference(model_path, 0, 2, false);
        Shard prefix(model_path, 0, 1, true);
        Shard suffix(model_path, 1, 2, false);
        const char * prompt = "Hello from the iPad";
        std::vector<llama_token> initial(32);
        const int n = llama_tokenize(llama_model_get_vocab(reference.model), prompt, strlen(prompt), initial.data(), initial.size(), true, true);
        if (n < 1 || n > 32) throw std::runtime_error("tokenization failed");
        initial.resize(n);
        auto tokens = initial;
        std::vector<float> first;
        float maximum_error = 0;
        for (int step = 0; step < 16; ++step) {
            auto expected = reference.execute(tokens);
            auto hidden = prefix.execute(tokens);
            auto actual = suffix.execute(tokens, &hidden);
            maximum_error = std::max(maximum_error, compare(expected, actual));
            if (step == 0) first = expected;
            tokens = {llama_token(std::max_element(expected.begin(), expected.end()) - expected.begin())};
        }
        reference.reset(); prefix.reset(); suffix.reset();
        auto expected = reference.execute(initial);
        auto hidden = prefix.execute(initial);
        auto actual = suffix.execute(initial, &hidden);
        maximum_error = std::max({maximum_error, compare(first, expected), compare(expected, actual)});
        std::ofstream report(report_path);
        report << "{\"passed\":true,\"positions\":16,\"reset_passed\":true,\"max_logit_error\":" << maximum_error
               << ",\"prefix_buffer_bytes\":" << prefix.buffer_bytes()
               << ",\"suffix_buffer_bytes\":" << suffix.buffer_bytes()
               << ",\"reference_buffer_bytes\":" << reference.buffer_bytes()
               << ",\"configuration\":\"per-model C API; no environment mutation\",\"scope\":\"GLM two-layer numerical fixture, no Iroh\"}\n";
        if (!report) throw std::runtime_error("report write failed");
        return 0;
    } catch (const std::exception & error) {
        std::ofstream report(report_path);
        report << "{\"passed\":false,\"error\":\"native smoke failed; inspect device log\"}\n";
        fprintf(stderr, "Mobile shard smoke: %s\n", error.what());
        return 1;
    }
}
