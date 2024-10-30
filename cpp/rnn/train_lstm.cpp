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

void rnn_forward();
void rnn_backward();

void hadamard();

void sigmoid();
void tanh();


void crossentropy_forward();
void crossentropy_backward();

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

}

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
typdef struct{


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

  // Start HERE:

}


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

  // Train


  //Free Memory
  dataloader_free(&train_loader);
  dataloader_free(&val_loader);
  tokenizer_free(&tokenizer);
  

  return 0; 

  }
