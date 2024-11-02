// Standard C++ libraries
#include<bits/stdc++.h>
#include<iostream>
#include<math.h>
#include<string.h>
#include<fstream>
#include<vector>
#include<assert.h>
#include<time.h>


using namespace std;


// Dataloader
typedef struct{

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

void dataloader_init(DataLoader *loader, const char* filename, int B, int T){
  loader->B = B;
  loader->T = T;

  loader->tokens_file = fopen(filename, "rb");
  if(loader->tokens_file == NULL){
    cout<<"Error opening tokens file\n";
    exit(1);
  }

  fseek(loader->tokens_file, 0, SEEK_END);
  loader->file_size = ftell(loader->tokens_file);
  fseek(loader->tokens_file, 0, SEEK_SET);
  if(loader->file_size < (B * T + 1) * sizeof(int)){
    cout<<"Error: File size is too small for the batch size and sequence length\n";
    exit(1);
  }
  loader->current_position = 0;

  loader->batch = (int*) malloc((B * T +1) * sizeof(int));
  loader->inputs = loader->batch;
  loader->targets = loader->batch + 1;
  loader->num_batches = loader->file_size / (B * T * sizeof(int));
}

void dataloader_reset(DataLoader *loader){
  loader->current_position = 0;
}

void dataloader_next_batch(DataLoader *loader){
  int B = loader->B;
  int T = loader->T;

  if(loader->current_position + (B * T +1) * sizeof(int) > loader->file_size){
    loader->current_position = 0;
  }

  fseek(loader->tokens_file, loader->current_position, SEEK_SET);
  fread(loader->batch, sizeof(int), B*T+1, loader->tokens_file);
  loader->current_position += B * T * sizeof(int);
}

void dataloader_free(DataLoader *loader){
  fclose(loader->tokens_file);
  free(loader->batch);
}

// --------------------------------------------------------- //

// end-of-text token id
#define EOT 50256

unsigned int random_u32(unsigned long long *state){

  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) { // random float32 in [0,1)
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

// Tokenizer (Decoding)
typedef struct{
  uint32_t vocab_size;
  char **token_table;
  int init_ok;
} Tokenizer;

void safe_printf(const char *piece){
  if(piece == NULL) {return;}
  if(piece[0] == '\0') {return;}

  if(piece[1] == '\0'){
    unsigned char byte_val = piece[0];
    if(!(isprint(byte_val) || isspace(byte_val))){
      return;
    }
  }
  cout<<piece;
}

void tokenizer_init(Tokenizer *tokenizer, const char *filename){
  FILE *file = fopen(filename, "rb");
  if(file == NULL){
    cout<<"-------"<<endl;
    cout<<"Failed to open tokenizer file"<<endl;
    cout<<"-------"<<endl;
    tokenizer->init_ok = 0;
    return; 
  }

  uint32_t header[256];
  fread(header, sizeof(uint32_t), 256, file);
  assert(header[0] == 20240328);
  assert(header[1] == 1);
  tokenizer->vocab_size = header[2];

  unsigned char length;
  tokenizer->token_table = (char **)malloc(tokenizer->vocab_size * sizeof(char *));
  for(uint32_t i = 0; i<tokenizer->vocab_size;i++){
    fread(&length, sizeof(unsigned char), 1, file);
    assert(length > 0);
    char *token_bytes = (char *)malloc(length + 1);
    fread(token_bytes, sizeof(char), length, file);
    token_bytes[length] = '\0';
    tokenizer->token_table[i] = token_bytes;
  }

  fclose(file);
  tokenizer->init_ok = 1;
}

const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id){
  if(tokenizer->init_ok == 0){
    return NULL;
  }

  if(token_id < tokenizer->vocab_size){
    return tokenizer->token_table[token_id];
  } else{
    cout<<"Invalid token id:"<< token_id<<endl;
    return NULL;
  }
}

void tokenizer_free(Tokenizer *tokenizer){
  if(tokenizer->init_ok){
    for(uint32_t i=0;i<tokenizer->vocab_size;i++){
      free(tokenizer->token_table[i]);
    }
    free(tokenizer->token_table);
  }
}


// --------------------------------------------------------- //

// LSTM Model
void lstm_cell_forward(float* out, float* inp, float* weight, float* bias, int B, int T, int C){

    for(int b=0;b<B;b++){
      for(int t=0;t<T;t++){

          // Input gate
          // float* wxi = 0;
          // float* whi = 0;
          // float* bi = 0;

          // Forget gate
          // float* wxf = 0;
          // float* whf = 0;
          // float* bf = 0;

          // Output gate
          // float* wxo = 0;
          // float* who = 0;
          // float* bo = 0;

          // Input Node
          // float* wxc = 0;
          // float* whc = 0;
          // float* bc = 0;
        
      }
    }

}

void lstm_cell_backward(float* dout, float* inp, float* dweight, float* dbias, int B, int T, int C){

    for(int b=0;b<B;b++){
      for(int t=0;t<T;t++){

          // Input gate
          // float* dwxi = ;
          // float* dwhi = ;
          // float* dbi = ;

          // Forget gate
          // float* dwxf = ;
          // float* dwhf = ;
          // float* bf = ;

          // Output gate
          // float* dwxo = ;
          // float* dwho = ;
          // float* dbo = ;

          // Input Node
          // float* dwxc = ;
          // float* dwhc = ;
          // float* dbc = ;
        
      }
    }

}

void hadamard();

void sigmoid();
void tanh_forward();

void crossentropy_forward(float* losses, float* probs, int* targets, int B, int T, int V){

    for(int b=0;b<B;b++){
      for(int t=0;t<T;t++){
        float* probs_bt = probs + b * T * V + t * V;
        int ix = targets[b * T + t];
        losses[b * T + t] = -logf(probs_bt[ix]);
      }
    }
}
void crossentropy_tanh_backward(float* dlogits, float* dlosses, float* probs, int* targets, int B, int T, int V){

  // Backward through both tanh and crossentropy
  for(int b=0;b<B;b++){
    for(int t=0;t<T;t++){
      float* dlogits_bt = dlogits + b * T * V + t * V;
      float* probs_bt = probs + b * T * V + t * V;
      float dloss = dlosses[b * T + t];

      int ix = targets[b * T + t];
      for(int i=0;i<V;i++){
        float p = probs_bt[i];
        float indicator = i == ix ? 1.0f : 0.0f;
        dlogits_bt[i] += (p - indicator) * dloss;
      }
    }
  }
}

void initalize_weights();
void gradient_clipping();

// --------------------------------------------------------- //

// TODO: Add all the Parameter tensors and change the number
// Parameters of the model
#define NUM_PARAMETER_TENSORS 4
typedef struct{
  float* I; // Input gate
  float* F; // Forget gate
  float* O; // Output gate
  float* C_tilde; // Input Node

} ParameterTensors;

float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_size){

  size_t num_parameters = 0;
  for(size_t i=0;i < NUM_PARAMETER_TENSORS;i++){
    num_parameters += param_sizes[i];
  }

  float* params_memory = (float*)malloc(num_parameters * sizeof(float));
  float** ptrs[] = {
    &params->I, &params->F, &params->O, &params->C_tilde
  };

  float* params_memory_iterator = params_memory;
  for(size_t i=0;i < NUM_PARAMETER_TENSORS; i++){
    *(ptrs[i]) = params_memory_iterator;
    params_memory_iterator += param_sizes[i];
  }

  return params_memory;

}

// TODO: Add all the Activation tensors and change the number
#define NUM_ACTIVATION_TENSORS 4
typedef struct{


  float* logits;
  float* losses;

} ActivationTensors;

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes){

  size_t num_activations = 0;
  for(size_t i=0;i<NUM_ACTIVATION_TENSORS;i++){
    num_activations += act_sizes[i];
  }

  float* acts_memory = (float*)malloc(num_activations * sizeof(float));
  float** ptrs[] = {
    &acts->logits, &acts->losses
  };

  float* acts_memory_iterator = acts_memory;
  for(size_t i = 0;i<NUM_ACTIVATION_TENSORS;i++){
    *(ptrs[i]) = acts_memory_iterator;
    acts_memory_iterator += act_sizes[i];
  }

  return acts_memory;
}

typedef struct {
  int max_seq_len;
  int vocab_size;
  int padded_vocab_size;
  int num_layers;
  int num_heads;
  int channels;
} LSTMConfig;

typedef struct{
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

  // Other state configutations
  int batch_size;
  int seq_len;
  int* inputs;
  int* targets;
  float mean_loss;

} LSTM;

void lstm_init(LSTM *model){

  // Hyperparameters
  // model->config.max_seq_len = ;
  // model->config.vocab_size = ;
  // model->config.num_layers = ;
  // model->config.num_heads = ;
  // model->config.channels = ;


  // Allocate Space for the parameters and initialize them
  // model->param_sizes[0] = ;
  // ....

  // Count the number of parameters
  size_t num_parameters = 0;
  for(size_t i=0;i<NUM_PARAMETER_TENSORS;i++){
    num_parameters += model->param_sizes[i];
  }
  cout<<"Number of parameters: "<<num_parameters<<endl;
  model->num_parameters = num_parameters;

  // Allocating memory for parameters
  model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);

  // Other initializations
  model->acts_memory = NULL;
  model->grads_memory = NULL;
  model->m_memory = NULL;
  model->v_memory = NULL;
  model->grads_acts_memory = NULL;
  model->inputs = NULL;
  model->targets = NULL;
  model->batch_size = 0;
  model->seq_len = 0;
  model->mean_loss = -1.0f;
}

void lstm_forward(LSTM *model, int* inputs, int* targets, int B, int T){

  if(model->params_memory == NULL){
    cout<<"Error: Model was not initialized."<<endl;
    exit(1);
  }

  // Params
  int V = model->config.vocab_size;
  int L = model->config.num_layers;
  int NH = model->config.num_heads;
  int C = model->config.channels;

  // Validating all indices are in the range [0,V)
  for(int i=0;i<B*T;i++){
    assert(0 <= inputs[i] && inputs[i] < V);
    if(targets != NULL){
      assert(0 <= targets[i] && targets[i] < V);
    }
  }

  // Allocating space for ll the activations
  if(model->acts_memory == NULL){
    // Current B, T
    model->batch_size = B;
    model->seq_len = T;

    // Allocating space
    // model->act_sizes[0] = ;
    // ...

    size_t num_activations = 0;
    for(size_t i=0;i<NUM_ACTIVATION_TENSORS;i++){
      num_activations += model->act_sizes[i];
    }
    cout<<"Num of activation: "<<num_activations<<endl;
    
    model->num_activations = num_activations;
    model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);

    // Memory for caching inputs and targets
    model->inputs = (int*)malloc(B*T*size(int));
    model->targets = (int*)malloc(B*T*sizeof(int));
  } else{
    // Validating B, T memory allocation
    if(B != model->batch_size || T != model->seq_len){
      cout<<"Model: B = "<< model->batch_size << "T = "<<model->seq_len<<"Desired: B = "<<B<<" T = "<<T;
      exit(EXIT_FAILURE);
    }
  }

  // cache the inputs/targets
  memcpy(model->inputs, inputs, B * T * sizeof(int));
  if(targets != NULL){
    memcpy(model->targets, targets, B * T * sizeof(int));
  } 

  // Forward Pass
  ParameterTensors params = model->params;
  ActivationTensors acts = model->acts;
  float* residual;

  for(int l=0;l<L;l++){

    // Accessing the pointers of the weights for this layer
    // ...

    // Accessing the pointers of the weights for this layer
    // ...

    // Forward Pass
    lstm_cell_forward();    
  }

  tanh_forward();

  // Forward the cross-entropy loss function is we have the targets
  if(targets != NULL){
    crossentropy_forward(model->acts.losses, model->probs, targets, B, T, V);

    // Evaluate the mean loss
    float mean_loss = 0.0f;
    for(int i=0;i<B*T;i++){
      mean_loss += model->acts.losses[i];
    }
    mean_loss /= B*T;
    model->mean_loss = mean_loss;
  } else{
    model->mean_loss = -1.0f;
  }
}

void lstm_zero_grad(GPT2 *model){
  if(model->grads_memory != NULL){
    memset(model->grads_memory, 0, model->num_parameters * sizeof(float));
  }
  if(model->grads_acts_memory != NULL){
    memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float));
  }
}

void lstm_backward(LSTM *model){

  // Checking forward pass done with targets
  if(model->mean_loss == -1.0f){
    cout<<"Error: Forward pass not done before backward pass"<<endl;
    exit(1);
  }

  // Lazily allocate the memory for gradients of the weights and activations, if not done
  if(model->grads_memory == NULL){
    model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
    model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
    lstm_zero_grad(model);
  }

  // Params
  int B = model->batch_size;
  int T = model->seq_len;
  int V = model->config.vocab_size;
  int L = model->config.num_layers;
  int NH = model->config.num_heads;
  int C = model->config.channels;

  // Backward Pass, going in reverse order of forward pass and calling backward functions
  ParameterTensors params = model->params;
  ParameterTensors grads = model->grads;
  ActivationTensors acts = model->acts;
  ActivationTensors grads_acts = model->grads_acts;

  // Starting Chain rule
  float dloss_mean = 1.0f / (B*T);
  for(int i=0;i<B*T;i++){
    grads_acts.losses[i] = dloss_mean;
  }

  // Start here:
  crossentropy_tanh_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V);

  for(int l=L-1;l>=0;l--){

    // Get pointers of the weights for this layer
    // ...

    // Get the pointers of the gradients of the weights for this layer
    // ...

    // Get pointers of the activations for this layer
    // ... 

    // Get pointers of the gradients of the activations for this layer
    // ...

    // Backprop Layer
    lstm_cell_backward();
  }

}

// Update using AdamW
void lstm_update(LSTM *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t){

  // Allocate memory for m_memory and v_memory
  if(model->m_memory == NULL){
    model->m_memory = (float*)calloc(model->num_parameters, sizeof(float));
    model->v_memory = (float*)calloc(model->num_parameters, sizeof(float));
  }

  for(int i=0;i<model->num_parameters;i++){
    float param = model->params_memory[i];
    float grad = model->grads_memory[i];

    // Updating the first moment (momentum)
    float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
    // Updating the secong moment (RMSprop)
    float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
    // Bias-correct both moments
    float m_hat = m / (1.0f - powf(beta1, t));
    float v_hat = v / (1.0f - powf(beta2, t));

    // Update
    model->m_memory[i] = m;
    model->v_memory[i] = v;
    model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);

  }

}

void lstm_free(LSTM *model){
  free(model->params_memory);
  free(model->grads_memory);
  free(model->m_memory);
  free(model->v_memory);
  free(model->acts_memory);
  free(model->grads_acts_memory);
  free(model->inputs);
  free(model->targets);
}


#ifndef TESTING
// Skip main below if testing



int main() { 
  

  // Building the Dataloader
  const char* tiny_shakespeare_train = "../datasets/tiny_shakespeare/tiny_shakespeare_train.bin";
  const char* tiny_shakespeare_val = "../datasets/tiny_shakespeare/tiny_shakespeare_val.bin";

  const char* train_tokens = tiny_shakespeare_train;
  const char* val_tokens = tiny_shakespeare_val;

  int B = 4;
  int T = 64;

  DataLoader train_loader;
  dataloader_init(&train_loader, train_tokens, B, T);
  cout<<"Train dataset num_batches:"<<train_loader.num_batches<<endl;

  DataLoader val_loader;
  dataloader_init(&val_loader, val_tokens, B, T);
  cout<<"Val dataset num_batches: "<<val_loader.num_batches<<endl;

  int val_num_batches = 5;

  Tokenizer tokenizer;
  tokenizer_init(&tokenizer, "../datasets/gpt2_tokenizer.bin");
  

  // Allocating memory to generate smaples from the model
  unsigned long long rng_state = 1337;
  int* gen_tokens = (int*)malloc(B*T*sizeof(int));
  const inst gentT = 64;

  // Train
  struct timespec start, end;
  for(int step=0;step<=50;step++){

    // Estimating the validation loss
    if(step % 10 == 0){
      float val_loss = 0.0f;
      dataloader_reset(&val_loader);

      for(int i=0;i<val_num_batches;i++){
        dataloader_next_batch(&val_loader);
        lstm_forward(&model, val_loader.inputs, val_loader.targets, B, T);
        val_loss += model.mean_loss;
      }
      val_loss /= val_num_batches;
      cout<<"Val loss: "<<val_loss<<endl;
    }

    // Print generated text
    if(step > 0 && step % 20 == 0){
      for(int i=0;i<B*T;i++){
        gen_tokens[i] = TEXT_EOT;
      }
      cout<<"Generating:---------"<<endl;
      for(int t=1;t<genT;t++){

        // Wasteful inference
        lstm_forward(&model, gen_tokens, NULL, B, T);

        float* probs = model.acts.probs + (t-1) * model.config.vocab_size;
        float coin = random_32(&rng_state);

        int next_token = sample_mult(probs, model.config.vocab_size, coin);
        gen_tokens[t] = next_token;

        if(tokenizer.init_ok){
          const char* token_str = tokenizer_decode(&tokenizer, next_token);
          safe_printf(token_str);
        } else{
          // Printing the token id
          cout<<next_token;
        }
        fflush(stdout);
      }
      cout<<"\n-----------\n";
    }

    // Training step
    auto start = std::chrono::steady_clock::now();
    dataloader_next_batch(&train_loader);
    lstm_forward(&model);
    lstm_zero_grad(&model);
    lstm_backward(&model);
    lstm_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
    auto end = std::chrono::steady_clock::now();

    double time_elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count();

    cout<<"Step: "<<step<<" train loss: "<< model.mean_loss<<" took: "<< time_elapsed_s * 1000;
  }


  //Free Memory
  dataloader_free(&train_loader);
  dataloader_free(&val_loader);
  tokenizer_free(&tokenizer);
  lstm_free(&model);
  free(gen_tokens);  

  return 0; 

  }
