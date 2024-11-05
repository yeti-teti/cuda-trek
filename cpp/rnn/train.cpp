// Standard C++ libraries
#include <iostream>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <cassert>
#include <ctime>
#include <chrono>
#include <cstdlib>

using namespace std;

// --------------------------------------------------------- //
// DataLoader
typedef struct {
    int B;
    int T;

    FILE* tokens_file;
    long file_size;
    long current_position;

    int* batch;
    int* inputs;
    int* targets;

    int num_batches;

} DataLoader;

void dataloader_init(DataLoader* loader, const char* filename, int B, int T) {
    loader->B = B;
    loader->T = T;

    loader->tokens_file = fopen(filename, "rb");
    if (loader->tokens_file == NULL) {
        cout << "Error opening tokens file\n";
        exit(1);
    }

    fseek(loader->tokens_file, 0, SEEK_END);
    loader->file_size = ftell(loader->tokens_file);
    fseek(loader->tokens_file, 0, SEEK_SET);
    if (loader->file_size < (B * T + 1) * sizeof(int)) {
        cout << "Error: File size is too small for the batch size and sequence length\n";
        exit(1);
    }
    loader->current_position = 0;

    loader->batch = (int*)malloc((B * T + 1) * sizeof(int));
    loader->inputs = loader->batch;
    loader->targets = loader->batch + 1;
    loader->num_batches = loader->file_size / (B * T * sizeof(int));
}

void dataloader_reset(DataLoader* loader) {
    loader->current_position = 0;
}

void dataloader_next_batch(DataLoader* loader) {
    int B = loader->B;
    int T = loader->T;

    if (loader->current_position + (B * T + 1) * sizeof(int) > loader->file_size) {
        loader->current_position = 0;
    }

    fseek(loader->tokens_file, loader->current_position, SEEK_SET);
    fread(loader->batch, sizeof(int), B * T + 1, loader->tokens_file);
    loader->current_position += B * T * sizeof(int);
}

void dataloader_free(DataLoader* loader) {
    fclose(loader->tokens_file);
    free(loader->batch);
}

// --------------------------------------------------------- //
// Sampler

// End-of-text token id
#define EOT 50256

unsigned int random_u32(unsigned long long* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (unsigned int)((*state * 0x2545F4914F6CDD1Dull) >> 32);
}

float random_f32(unsigned long long* state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

// --------------------------------------------------------- //
// Tokenizer (Decoding)
typedef struct {
    uint32_t vocab_size;
    char** token_table;
    int init_ok;
} Tokenizer;

void safe_printf(const char* piece) {
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }

    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    cout << piece;
}

void tokenizer_init(Tokenizer* tokenizer, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        cout << "-------" << endl;
        cout << "Failed to open tokenizer file" << endl;
        cout << "-------" << endl;
        tokenizer->init_ok = 0;
        return;
    }

    uint32_t header[256];
    fread(header, sizeof(uint32_t), 256, file);
    if (header[0] != 20240328) {
        cout << "Error: Invalid tokenizer file format (magic number mismatch)." << endl;
        fclose(file);
        tokenizer->init_ok = 0;
        return;
    }
    if (header[1] != 1) {
        cout << "Error: Unsupported tokenizer version." << endl;
        fclose(file);
        tokenizer->init_ok = 0;
        return;
    }
    tokenizer->vocab_size = header[2];

    unsigned char length;
    tokenizer->token_table = (char**)malloc(tokenizer->vocab_size * sizeof(char*));
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        fread(&length, sizeof(unsigned char), 1, file);
        assert(length > 0);
        char* token_bytes = (char*)malloc(length + 1);
        fread(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';
        tokenizer->token_table[i] = token_bytes;
    }

    fclose(file);
    tokenizer->init_ok = 1;
}

const char* tokenizer_decode(Tokenizer* tokenizer, uint32_t token_id) {
    if (tokenizer->init_ok == 0) {
        return NULL;
    }

    if (token_id < tokenizer->vocab_size) {
        return tokenizer->token_table[token_id];
    }
    else {
        cout << "Invalid token id: " << token_id << endl;
        return NULL;
    }
}

void tokenizer_free(Tokenizer* tokenizer) {
    if (tokenizer->init_ok) {
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->token_table[i]);
        }
        free(tokenizer->token_table);
    }
}

// --------------------------------------------------------- //
// Activation Functions
void sigmoid_forward(float* out, float* inp, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = 1.0f / (1.0f + expf(-inp[i]));
    }
}

void sigmoid_backward(float* dinp, float* out, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        float sigmoid = out[i];
        dinp[i] += dout[i] * sigmoid * (1.0f - sigmoid);
    }
}

void tanh_forward(float* out, float* inp, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = tanhf(inp[i]);
    }
}

void tanh_backward(float* dinp, float* out, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        float tanh_val = out[i];
        dinp[i] += dout[i] * (1.0f - tanh_val * tanh_val);
    }
}

// --------------------------------------------------------- //
// Cross-Entropy Loss
void crossentropy_forward(float* losses, float* probs, int* targets, int B, int T, int V) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* probs_bt = probs + b * T * V + t * V;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

void crossentropy_softmax_backward(float* dlogits, float* dlosses, float* probs, int* targets, int B, int T, int V) {
    // Backward through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * V + t * V;
            float* probs_bt = probs + b * T * V + t * V;
            float dloss = dlosses[b * T + t];

            int ix = targets[b * T + t];
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// --------------------------------------------------------- //
// Embedding Layer
void embedding_forward(float* embeddings, int* input_ids, float* wte, int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int token_id = input_ids[b * T + t];
            float* embedding_vector = wte + token_id * C;
            float* embeddings_bt = embeddings + b * T * C + t * C;
            memcpy(embeddings_bt, embedding_vector, C * sizeof(float));
        }
    }
}

void embedding_backward(float* dwte, float* d_embeddings, int* input_ids, int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int token_id = input_ids[b * T + t];
            float* d_embedding_vector = d_embeddings + b * T * C + t * C;
            float* dwte_token = dwte + token_id * C;
            for (int i = 0; i < C; i++) {
                dwte_token[i] += d_embedding_vector[i];
            }
        }
    }
}

// --------------------------------------------------------- //
// LSTM Model Parameters
#define NUM_PARAMETER_TENSORS 6
typedef struct {
    float* wte;    // Token embeddings (V, C)
    float* W_x;    // Input weights (C, 4*C)
    float* W_h;    // Hidden weights (C, 4*C)
    float* b;      // Biases (4*C)
    float* W_out;  // Output weights (C, V)
    float* b_out;  // Output biases (V)
} ParameterTensors;

float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }

    float* params_memory = (float*)malloc(num_parameters * sizeof(float));
    float** ptrs[] = {
        &params->wte, &params->W_x, &params->W_h, &params->b, &params->W_out, &params->b_out
    };

    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }

    return params_memory;
}

// Activation Tensors
#define NUM_ACTIVATION_TENSORS 7
typedef struct {
    float* embeddings;  // (B, T, C)
    float* h;           // (B, T, C)
    float* c;           // (B, T, C)
    float* gates;       // (B, T, 4*C)
    float* logits;      // (B, T, V)
    float* probs;       // (B, T, V)
    float* losses;      // (B, T)
} ActivationTensors;

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }

    float* acts_memory = (float*)malloc(num_activations * sizeof(float));
    float** ptrs[] = {
        &acts->embeddings, &acts->h, &acts->c, &acts->gates, &acts->logits, &acts->probs, &acts->losses
    };

    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }

    return acts_memory;
}

// LSTM Configurations
typedef struct {
    int max_seq_len;
    int vocab_size;
    int num_layers;
    int channels;
} LSTMConfig;

typedef struct {
    LSTMConfig config;

    // The weights (Parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;

    // Gradients of the weights
    ParameterTensors grads;
    float* grads_memory;

    // Buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;

    // The activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;

    // Gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;

    // Other state configurations
    int batch_size;
    int seq_len;
    int* inputs;
    int* targets;
    float mean_loss;

} LSTM;

// Initialize weights with small random values
void initialize_weights(float* weights, size_t size) {
    for (size_t i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }
}

void lstm_init(LSTM* model) {
    // Hyperparameters
    model->config.max_seq_len = 1024;
    model->config.vocab_size = 50257; // GPT-2 tokenizer vocab size
    model->config.num_layers = 1;
    model->config.channels = 768; // Hidden size

    int V = model->config.vocab_size;
    int C = model->config.channels;

    // Allocate space for parameters
    model->param_sizes[0] = V * C;        // wte
    model->param_sizes[1] = C * 4 * C;    // W_x
    model->param_sizes[2] = C * 4 * C;    // W_h
    model->param_sizes[3] = 4 * C;        // b
    model->param_sizes[4] = C * V;        // W_out
    model->param_sizes[5] = V;            // b_out

    // Count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    cout << "Number of parameters: " << num_parameters << endl;
    model->num_parameters = num_parameters;

    // Allocate memory for parameters
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    // Initialize parameters
    initialize_weights(model->params.wte, model->param_sizes[0]);
    initialize_weights(model->params.W_x, model->param_sizes[1]);
    initialize_weights(model->params.W_h, model->param_sizes[2]);
    initialize_weights(model->params.b, model->param_sizes[3]);
    initialize_weights(model->params.W_out, model->param_sizes[4]);
    initialize_weights(model->params.b_out, model->param_sizes[5]);

    // Initialize gradients to zero
    model->grads_memory = (float*)calloc(num_parameters, sizeof(float));
    malloc_and_point_parameters(&model->grads, model->param_sizes);

    // Other initializations
    model->acts_memory = NULL;
    model->grads_acts_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f;
}

void matmul_forward(float* out, float* inp, float* weight, float* bias, int B, int T, int C, int OC) {
    // out: (B, T, OC)
    // inp: (B, T, C)
    // weight: (C, OC)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                float* w_col = weight + o * C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * w_col[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

void softmax_forward(float* probs, float* logits, int B, int T, int V) {
    // output: probs are (B,T,V) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,V) of the unnormalized log probabilities
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // probs <- softmax(logits)
            float* logits_bt = logits + b * T * V + t * V;
            float* probs_bt = probs + b * T * V + t * V;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -INFINITY;
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
        }
    }
}

void lstm_forward(LSTM* model, int* inputs, int* targets, int B, int T) {
    if (model->params_memory == NULL) {
        cout << "Error: Model was not initialized." << endl;
        exit(1);
    }

    // Params
    int V = model->config.vocab_size;
    int C = model->config.channels;

    // Validating all indices are in the range [0,V)
    for (int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // Allocating space for all the activations
    if (model->acts_memory == NULL) {
        // Current B, T
        model->batch_size = B;
        model->seq_len = T;

        // Allocating space
        model->act_sizes[0] = B * T * C;       // embeddings
        model->act_sizes[1] = B * T * C;       // h
        model->act_sizes[2] = B * T * C;       // c
        model->act_sizes[3] = B * T * 4 * C;   // gates
        model->act_sizes[4] = B * T * V;       // logits
        model->act_sizes[5] = B * T * V;       // probs
        model->act_sizes[6] = B * T;           // losses

        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        cout << "Number of activations: " << num_activations << endl;

        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);

        // Allocate gradients of activations
        model->grads_acts_memory = (float*)calloc(num_activations, sizeof(float));
        malloc_and_point_activations(&model->grads_acts, model->act_sizes);

        // Memory for caching inputs and targets
        model->inputs = (int*)malloc(B * T * sizeof(int));
        model->targets = (int*)malloc(B * T * sizeof(int));
    }
    else {
        // Validating B, T memory allocation
        if (B != model->batch_size || T != model->seq_len) {
            cout << "Model: B = " << model->batch_size << " T = " << model->seq_len << " Desired: B = " << B << " T = " << T << endl;
            exit(EXIT_FAILURE);
        }
    }

    // Cache the inputs/targets
    memcpy(model->inputs, inputs, B * T * sizeof(int));
    if (targets != NULL) {
        memcpy(model->targets, targets, B * T * sizeof(int));
    }

    // Forward Pass
    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;

    // Embedding Lookup
    embedding_forward(acts.embeddings, inputs, params.wte, B, T, C);

    // Initialize h_prev and c_prev to zeros
    float* h_prev = (float*)calloc(B * C, sizeof(float));
    float* c_prev = (float*)calloc(B * C, sizeof(float));

    // Loop over time steps
    for (int t = 0; t < T; t++) {
        float* x_t = acts.embeddings + t * B * C; // (B, C)
        float* h_t = acts.h + t * B * C;
        float* c_t = acts.c + t * B * C;
        float* gates_t = acts.gates + t * B * 4 * C;

        // Compute gates: x_t @ W_x + h_prev @ W_h + b
        // gates_t shape: (B, 4*C)
        // x_t @ W_x: (B, C) @ (C, 4*C) -> (B, 4*C)
        // h_prev @ W_h: (B, C) @ (C, 4*C) -> (B, 4*C)

        // Allocate temporary storage
        float* x_Wx = (float*)calloc(B * 4 * C, sizeof(float));
        float* h_Wh = (float*)calloc(B * 4 * C, sizeof(float));

        // Matmul x_t @ W_x
        matmul_forward(x_Wx, x_t, params.W_x, NULL, B, 1, C, 4 * C);
        // Matmul h_prev @ W_h
        matmul_forward(h_Wh, h_prev, params.W_h, NULL, B, 1, C, 4 * C);

        // Compute gates_t = x_Wx + h_Wh + b
        for (int i = 0; i < B * 4 * C; i++) {
            gates_t[i] = x_Wx[i] + h_Wh[i] + params.b[i % (4 * C)];
        }

        // Split gates_t into i_t, f_t, o_t, g_t and apply activations
        float* i_t = (float*)malloc(B * C * sizeof(float));
        float* f_t = (float*)malloc(B * C * sizeof(float));
        float* o_t = (float*)malloc(B * C * sizeof(float));
        float* g_t = (float*)malloc(B * C * sizeof(float));

        for (int b = 0; b < B; b++) {
            float* gates_b = gates_t + b * 4 * C;
            float* i_b = i_t + b * C;
            float* f_b = f_t + b * C;
            float* o_b = o_t + b * C;
            float* g_b = g_t + b * C;

            // Input gate
            sigmoid_forward(i_b, gates_b + 0 * C, C);
            // Forget gate
            sigmoid_forward(f_b, gates_b + 1 * C, C);
            // Output gate
            sigmoid_forward(o_b, gates_b + 2 * C, C);
            // Cell gate
            tanh_forward(g_b, gates_b + 3 * C, C);
        }

        // Compute c_t = f_t * c_prev + i_t * g_t
        for (int i = 0; i < B * C; i++) {
            c_t[i] = f_t[i] * c_prev[i] + i_t[i] * g_t[i];
        }

        // Compute h_t = o_t * tanh(c_t)
        float* tanh_c_t = (float*)malloc(B * C * sizeof(float));
        tanh_forward(tanh_c_t, c_t, B * C);
        for (int i = 0; i < B * C; i++) {
            h_t[i] = o_t[i] * tanh_c_t[i];
        }

        // Update h_prev and c_prev
        memcpy(h_prev, h_t, B * C * sizeof(float));
        memcpy(c_prev, c_t, B * C * sizeof(float));

        // Free temporary storage
        free(x_Wx);
        free(h_Wh);
        free(i_t);
        free(f_t);
        free(o_t);
        free(g_t);
        free(tanh_c_t);
    }

    free(h_prev);
    free(c_prev);

    // Output layer
    // h_t @ W_out + b_out
    // h_t shape: (B*T, C)
    // logits shape: (B*T, V)
    matmul_forward(acts.logits, acts.h, params.W_out, params.b_out, B, T, C, V);

    // Softmax
    softmax_forward(acts.probs, acts.logits, B, T, V);

    // Forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        crossentropy_forward(acts.losses, acts.probs, targets, B, T, V);

        // Evaluate the mean loss
        float mean_loss = 0.0f;
        for (int i = 0; i < B * T; i++) {
            mean_loss += acts.losses[i];
        }
        mean_loss /= B * T;
        model->mean_loss = mean_loss;
    }
    else {
        model->mean_loss = -1.0f;
    }
}

void lstm_zero_grad(LSTM* model) {
    if (model->grads_memory != NULL) {
        memset(model->grads_memory, 0, model->num_parameters * sizeof(float));
    }
    if (model->grads_acts_memory != NULL) {
        memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float));
    }
}


void matmul_backward(float* dinp, float* dweight, float* dbias, float* dout, float* inp, float* weight, int B, int T, int C, int OC) {
    // Backward into inp
    if (inp != NULL) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* dout_bt = dout + b * T * OC + t * OC;
                float* dinp_bt = dinp + b * T * C + t * C;
                for (int o = 0; o < OC; o++) {
                    float* w_col = weight + o * C;
                    float d = dout_bt[o];
                    for (int i = 0; i < C; i++) {
                        dinp_bt[i] += w_col[i] * d;
                    }
                }
            }
        }
    }

    // Backward into weight and bias (always needed even if inp is NULL)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * OC + t * OC;
            float* inp_bt = (inp != NULL) ? inp + b * T * C + t * C : NULL;
            for (int o = 0; o < OC; o++) {
                float d = dout_bt[o];
                float* dw_col = dweight + o * C;
                if (dbias != NULL) { dbias[o] += d; }
                if (inp_bt != NULL) {
                    for (int i = 0; i < C; i++) {
                        dw_col[i] += inp_bt[i] * d;
                    }
                }
            }
        }
    }
}


void lstm_backward(LSTM* model) {
    int B = model->batch_size;
    int T = model->seq_len;
    int V = model->config.vocab_size;
    int C = model->config.channels;

    ParameterTensors params = model->params;
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // Initialize gradients to zero
    memset(grads_acts.embeddings, 0, B * T * C * sizeof(float));
    memset(grads_acts.h, 0, B * T * C * sizeof(float));
    memset(grads_acts.c, 0, B * T * C * sizeof(float));
    memset(grads_acts.gates, 0, B * T * 4 * C * sizeof(float));

    // Compute gradient of loss w.r.t logits
    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V);

    // Backprop through output layer
    matmul_backward(grads_acts.h, grads.W_out, grads.b_out, grads_acts.logits, acts.h, params.W_out, B, T, C, V);

    // Initialize dh_next and dc_next to zero
    float* dh_next = (float*)calloc(B * C, sizeof(float));
    float* dc_next = (float*)calloc(B * C, sizeof(float));

    // Backprop through time
    for (int t = T - 1; t >= 0; t--) {
        float* dh = grads_acts.h + t * B * C;
        float* dc = grads_acts.c + t * B * C;
        float* gates_t = acts.gates + t * B * 4 * C;
        float* dgates_t = grads_acts.gates + t * B * 4 * C;
        float* c_t = acts.c + t * B * C;
        float* c_prev = (t > 0) ? acts.c + (t - 1) * B * C : NULL;
        float* h_prev = (t > 0) ? acts.h + (t - 1) * B * C : NULL;
        float* x_t = acts.embeddings + t * B * C;

        // Add dh from next time step
        for (int i = 0; i < B * C; i++) {
            dh[i] += dh_next[i];
        }

        // Compute gradients w.r.t o_t and c_t
        float* o_t = (float*)malloc(B * C * sizeof(float));
        float* tanh_c_t = (float*)malloc(B * C * sizeof(float));
        float* do_t = (float*)malloc(B * C * sizeof(float));
        float* dc_t = (float*)malloc(B * C * sizeof(float));

        // Extract gates
        float* gates_i = gates_t + 0 * B * C;
        float* gates_f = gates_t + 1 * B * C;
        float* gates_o = gates_t + 2 * B * C;
        float* gates_g = gates_t + 3 * B * C;

        // Compute o_t and tanh_c_t
        sigmoid_forward(o_t, gates_o, B * C);
        tanh_forward(tanh_c_t, c_t, B * C);

        // do_t = dh * tanh(c_t) * o_t * (1 - o_t)
        for (int i = 0; i < B * C; i++) {
            do_t[i] = dh[i] * tanh_c_t[i];
        }
        sigmoid_backward(dgates_t + 2 * B * C, o_t, do_t, B * C);

        // dc = dh * o_t * (1 - tanh^2(c_t)) + dc_next
        for (int i = 0; i < B * C; i++) {
            dc_t[i] = dh[i] * o_t[i] * (1.0f - tanh_c_t[i] * tanh_c_t[i]) + dc_next[i];
        }

        // Compute gradients w.r.t gates
        float* di_t = (float*)malloc(B * C * sizeof(float));
        float* df_t = (float*)malloc(B * C * sizeof(float));
        float* dg_t = (float*)malloc(B * C * sizeof(float));

        float* i_t = (float*)malloc(B * C * sizeof(float));
        float* f_t = (float*)malloc(B * C * sizeof(float));
        float* g_t = (float*)malloc(B * C * sizeof(float));

        sigmoid_forward(i_t, gates_i, B * C);
        sigmoid_forward(f_t, gates_f, B * C);
        tanh_forward(g_t, gates_g, B * C);

        // di_t = dc * g_t * i_t * (1 - i_t)
        for (int i = 0; i < B * C; i++) {
            di_t[i] = dc_t[i] * g_t[i];
        }
        sigmoid_backward(dgates_t + 0 * B * C, i_t, di_t, B * C);

        // In lstm_backward, before accessing c_prev and h_prev
        if (t > 0) {
            c_prev = acts.c + (t - 1) * B * C;
            h_prev = acts.h + (t - 1) * B * C;
        } else {
            c_prev = NULL;
            h_prev = NULL;
        }

        // df_t = dc * c_prev * f_t * (1 - f_t)
        if (c_prev != NULL) {
            for (int i = 0; i < B * C; i++) {
                df_t[i] = dc_t[i] * c_prev[i];
            }
            sigmoid_backward(dgates_t + 1 * B * C, f_t, df_t, B * C);
        } else {
            // Set df_t and corresponding gradients to zero
            memset(df_t, 0, B * C * sizeof(float));
            memset(dgates_t + 1 * B * C, 0, B * C * sizeof(float));
        }

        // dg_t = dc * i_t * (1 - g_t^2)
        for (int i = 0; i < B * C; i++) {
            dg_t[i] = dc_t[i] * i_t[i];
        }
        tanh_backward(dgates_t + 3 * B * C, g_t, dg_t, B * C);

        // Update dc_next
        if (c_prev != NULL) {
            for (int i = 0; i < B * C; i++) {
                dc_next[i] = dc_t[i] * f_t[i];
            }
        }

        // Compute gradients w.r.t inputs and weights
        // Combine gate gradients
        float* dgates_combined = dgates_t;

        // Gradients w.r.t x_t
        matmul_backward(grads_acts.embeddings + t * B * C, grads.W_x, NULL, dgates_combined, x_t, params.W_x, B, 1, C, 4 * C);

        // Gradients w.r.t h_prev
        if (h_prev != NULL) {
            matmul_backward(dh_next, grads.W_h, NULL, dgates_combined, h_prev, params.W_h, B, 1, C, 4 * C);
        } else {
            // If h_prev is zero, only accumulate gradients w.r.t weights
            matmul_backward(NULL, grads.W_h, NULL, dgates_combined, NULL, params.W_h, B, 1, C, 4 * C);
            memset(dh_next, 0, B * C * sizeof(float));
        }

        // Accumulate biases
        for (int i = 0; i < 4 * C; i++) {
            for (int b = 0; b < B; b++) {
                grads.b[i] += dgates_combined[b * 4 * C + i];
            }
        }

        // Free temporary variables
        free(o_t);
        free(tanh_c_t);
        free(do_t);
        free(dc_t);
        free(di_t);
        free(df_t);
        free(dg_t);
        free(i_t);
        free(f_t);
        free(g_t);
    }

    // Backprop through embeddings
    embedding_backward(grads.wte, grads_acts.embeddings, model->inputs, B, T, C);

    // Free temporary variables
    free(dh_next);
    free(dc_next);
}

// AdamW 
void lstm_update(LSTM* model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // Initialize optimizer buffers if not already done
    if (model->m_memory == NULL) {
        model->m_memory = (float*)calloc(model->num_parameters, sizeof(float));
        model->v_memory = (float*)calloc(model->num_parameters, sizeof(float));
    }

    for (int i = 0; i < model->num_parameters; i++) {
        float param = model->params_memory[i];
        float grad = model->grads_memory[i];

        // Apply weight decay
        grad += weight_decay * param;

        // Update first moment
        model->m_memory[i] = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;

        // Update second moment
        model->v_memory[i] = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;

        // Compute bias-corrected moments
        float m_hat = model->m_memory[i] / (1.0f - powf(beta1, t));
        float v_hat = model->v_memory[i] / (1.0f - powf(beta2, t));

        // Update parameter
        model->params_memory[i] -= learning_rate * m_hat / (sqrtf(v_hat) + eps);
    }
}

void lstm_free(LSTM* model) {
    free(model->params_memory);
    free(model->grads_memory);
    free(model->m_memory);
    free(model->v_memory);
    free(model->acts_memory);
    free(model->grads_acts_memory);
    free(model->inputs);
    free(model->targets);
}

// --------------------------------------------------------- //
// Main function

#ifndef TESTING

int main() {

    // Seed the random number generator
    srand(time(NULL));

    // Initialize the LSTM model
    LSTM model;
    lstm_init(&model);

    // Building the Dataloader
    const char* tiny_shakespeare_train = "../datasets/tiny_shakespeare/tiny_shakespeare_train.bin";
    const char* tiny_shakespeare_val = "../datasets/tiny_shakespeare/tiny_shakespeare_val.bin";

    const char* train_tokens = tiny_shakespeare_train;
    const char* val_tokens = tiny_shakespeare_val;

    int B = 1;
    int T = 16;

    DataLoader train_loader;
    dataloader_init(&train_loader, train_tokens, B, T);
    cout << "Train dataset num_batches: " << train_loader.num_batches << endl;

    DataLoader val_loader;
    dataloader_init(&val_loader, val_tokens, B, T);
    cout << "Val dataset num_batches: " << val_loader.num_batches << endl;

    int val_num_batches = 5;

    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "../datasets/gpt2_tokenizer.bin");

    // Allocating memory to generate samples from the model
    unsigned long long rng_state = 1337;
    int* gen_tokens = (int*)malloc(B * T * sizeof(int));
    const int genT = 64;

    // Train
    for (int step = 0; step <= 50; step++) {

        // Estimating the validation loss
        if (step % 10 == 0) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);

            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                lstm_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            cout << "Val loss: " << val_loss << endl;
        }

        // Print generated text
        if (step > 0 && step % 20 == 0) {
            for (int i = 0; i < B * T; i++) {
                gen_tokens[i] = EOT;
            }
            cout << "Generating:\n-----------\n";
            for (int t = 1; t < genT; t++) {

                // Inference
                lstm_forward(&model, gen_tokens, NULL, B, T);

                float* probs = model.acts.probs + (t - 1) * model.config.vocab_size;
                float coin = random_f32(&rng_state);

                int next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;

                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                }
                else {
                    // Printing the token id
                    cout << next_token << " ";
                }
            }
            cout << "\n-----------\n";
        }

        // Training step
        auto start = std::chrono::steady_clock::now();
        dataloader_next_batch(&train_loader);
        lstm_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        lstm_zero_grad(&model);
        lstm_backward(&model);
        lstm_update(&model, 1e-3f, 0.9f, 0.999f, 1e-8f, 1e-5f, step + 1);
        auto end = std::chrono::steady_clock::now();

        double time_elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

        cout << "Step: " << step << " train loss: " << model.mean_loss << " took: " << time_elapsed_s * 1000 << " ms" << endl;
    }

    // Free Memory
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    lstm_free(&model);
    free(gen_tokens);

    return 0;

}

#endif

