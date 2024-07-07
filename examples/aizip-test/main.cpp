#include "common.h"
#include "llama.h"

// const char* MODEL_PATH = "models/full/bge-small-en-v1.5.gguf";
const char* MODEL_PATH = "models/full/multilingual-e5-large-instruct-f16.gguf";

int main() {
    llama_backend_init();

    llama_model_params params = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(MODEL_PATH, params);
    GGML_ASSERT(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_batch = 2048;
    cparams.n_ubatch = 2048;
    cparams.embeddings = true;
    llama_context* ctx = llama_new_context_with_model(model, cparams);
    GGML_ASSERT(ctx);

    std::vector<std::vector<int32_t>> inputs;
    // inputs.push_back({101, 7592, 1010, 2088, 999, 102}); // bert-bge
    inputs.push_back({0, 35378,     4,  8999,    38,     2}); // multilingual

    // check if the last token is SEP
    for (auto & inp : inputs) {
        if (inp.empty() || inp.back() != llama_token_sep(model)) {
            fprintf(stderr, "%s: warning: last token in the prompt is not SEP\n", __func__);
        }
    }


    const int n_batch = cparams.n_batch;

    llama_kv_cache_clear(ctx);

    const int n_prompts = inputs.size();
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    const int n_embd = llama_n_embd(model);
    std::vector<float> embeddings(n_prompts * n_embd, 0.0f);
    float* emb = embeddings.data();

    int s = 0;
    for (int k = 0; k < n_prompts; k++) {
        // clamp to n_batch tokens
        auto& inp = inputs[k];

        GGML_ASSERT(batch.n_tokens + (int)inp.size() <= n_batch);

        int token_count = 0;
        for (size_t i = 0; i < inp.size(); i++) {
            bool is_last = (i == inp.size() - 1);
            if (inp[i]==1)// This is padding token
            {
                llama_batch_add(batch, inp[i], 1, {0}, is_last);
            }
            else
            {
                llama_batch_add(batch, inp[i], token_count+2, {0}, is_last);
                token_count++;
            }
        }
    }

    fprintf(stderr, "%s: n_tokens = %d\n", __func__, batch.n_tokens);
    if (llama_decode(ctx, batch) < 0) {
        GGML_ASSERT(false);
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == NULL) {
            embd = llama_get_embeddings_ith(ctx, i);
        }
        GGML_ASSERT(embd);

        

        fprintf(stdout, "unnormalized embedding:");
        for (int hh = 0; hh < n_embd; hh++) {
            fprintf(stdout, "%9.6f,", embd[hh]);
        }
        fprintf(stdout, "\n");

        // float* out = emb + batch.seq_id[i][0] * n_embd;
        // llama_embd_normalize(embd, out, n_embd);
        // fprintf(stdout, "normalized embedding:");
        // for (int hh = 0; hh < n_embd; hh++) {
        //     fprintf(stdout, "%9.6f ", out[hh]);
        // }
        // fprintf(stdout, "\n");
    }
}
