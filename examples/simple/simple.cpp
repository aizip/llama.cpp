#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static void print_usage(int argc, char ** argv, const gpt_params & params) {
    gpt_params_print_usage(argc, argv, params);

    LOG_TEE("\nexample usage:\n");
    LOG_TEE("\n    %s -m model.gguf -p \"Hello my name is\" -n 32\n", argv[0]);
    LOG_TEE("\n");
}

int main(int argc, char ** argv) {
    // path to the model gguf file
    std::string model_path;
    // prompt to generate text from
    std::string prompt = "Hello my name is";
    // number of layers to offload to the GPU
    // int ngl = 99;
    // number of tokens to predict
    // int n_predict = 32;

    // parse command line arguments

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                // if (i + 1 < argc) {
                //     try {
                //         n_predict = std::stoi(argv[++i]);
                //     } catch (...) {
                //         print_usage(argc, argv);
                //         return 1;
                //     }
                // } else {
                //     print_usage(argc, argv);
                //     return 1;
                // }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                // if (i + 1 < argc) {
                //     try {
                //         ngl = std::stoi(argv[++i]);
                //     } catch (...) {
                //         print_usage(argc, argv);
                //         return 1;
                //     }
                // } else {
                //     print_usage(argc, argv);
                //     return 1;
                // }
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    // load dynamic backends

    ggml_backend_load_all();

    // initialize the model

    llama_model_params model_params = llama_model_default_params();

    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // tokenize the prompt

    // find the number of tokens in the prompt
    // const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // // allocate space for the tokens and tokenize the prompt
    // std::vector<llama_token> prompt_tokens(n_prompt);
    // if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
    //     fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
    //     return 1;
    // }
    

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    // ctx_params.n_ctx = n_prompt + n_predict - 1;
 
    ctx_params.n_ctx = 512;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = 2048;
    ctx_params.n_ubatch = 2048;
    ctx_params.embeddings = true;
    // enable performance counters
    ctx_params.no_perf = false;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_NONE;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    std::vector<llama_token> prompt_tokens =  common_tokenize(ctx, prompt, true, true);

    printf("prompt tokens: ");
    for (auto token: prompt_tokens) {
        printf("%d ", token);
    }
    printf("\n");

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler

    // auto sparams = llama_sampler_chain_default_params();
    // sparams.no_perf = false;
    // llama_sampler * smpl = llama_sampler_chain_init(sparams);

    // llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token

    // for (auto id : prompt_tokens) {
    //     char buf[128];
    //     int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    //     if (n < 0) {
    //         fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
    //         return 1;
    //     }
    //     std::string s(buf, n);
    //     printf("%s", s.c_str());
    // }

    // prepare a batch for the prompt

    // main loop

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();

    // for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
    //     // evaluate the current batch with the transformer model
    //     if (llama_decode(ctx, batch)) {
    //         fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
    //         return 1;
    //     }

    //     n_pos += batch.n_tokens;

    //     // sample the next token
    //     {
    //         new_token_id = llama_sampler_sample(smpl, ctx, -1);

    //         // is it an end of generation?
    //         if (llama_vocab_is_eog(vocab, new_token_id)) {
    //             break;
    //         }

    //         char buf[128];
    //         int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
    //         if (n < 0) {
    //             fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
    //             return 1;
    //         }
    //         std::string s(buf, n);
    //         printf("%s", s.c_str());
    //         fflush(stdout);

    //         // prepare the next batch with the sampled token
    //         batch = llama_batch_get_one(&new_token_id, 1);

    //         n_decode += 1;
    //     }
    // }

    const int n_prompts = prompt_tokens.size();
    struct llama_batch batch = llama_batch_init(ctx_params.n_batch, 0, 1);
    for (int k = 0; k < n_prompts; k++) {
        common_batch_add(batch, prompt_tokens[k], k, {0}, true);
    }
    llama_decode(ctx, batch);

    const int n_embd = llama_model_n_embd(model);
    float * embd = llama_get_embeddings(ctx);
    for (int i = 0; i < n_prompts; i++) {
        printf("prompt token %d: ", i);
        for (int j = 0; j < n_embd; j++) {
            printf("%f ", embd[i*n_embd + j]);
        }
        printf("\n");
    }

    // Save this embedding to a binary file which can be read by python
    FILE *f = fopen("embedding.bin", "wb");
    fwrite(embd, sizeof(float), n_embd*n_prompts, f);
    fclose(f);
 

    LOG_TEE("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "\n");
    // llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    // llama_sampler_free(smpl);
    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
