#include<iostream>
#include<cstdint>
#include<cctype>

#include "utils.hpp"

using namespace tokenizer {
  
  typedef struct{

    unit32_t vocab_size;
    char **token_table;
    int init_ok;
    int eot_token;
    
  } Tokenizer;

  void safe_cout(const char *piece){

    if(piece == NULL) { return; }
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

      cout<<"Failed to open tokenizer file: "<<filename;
      tokenizer->init_ok = 0;
      return;
    }

  

    uint32_t header[256];
    freadCheck(header, sizeof(uint32_t), 256, file);
    assert(header[0] == 20240328)

    int version = header[1];

    tokenizer->vocab_size = header[2];

    if(version == 1){
      
      assert(tokenizer->vocab_size == 50257);
      tokenizer->eot_token =50256;
      
    } else if (version == 2 ) {
      tokenizer->eot_token = header[3];
    } else{
      cerr<<"Tokenizer model file: "<< filename<<" has bad version "<<version;
      exit(EXIT_FAILURE);
    }

    unsigned char length;
    tokenizer->token_table = (char **)mallocCheck(tokenizer->vocab_size * sizeof(char *));
    
    for(uint32_t i=0;i < tokenizer->vocab_size;i++){
      
      freadCheck(&length, sizeof(unsigned char), 1, file);
      assert(length > 0);
      char *token_bytes = (char *)mallocCheck(length + 1);
      freadCheck(token_bytes, sizeof(char), length, file);
      token_bytes[length] = '\0';
      tokenizer->token_table[i] = token_bytes;

    }

    fcloseCheck(file);
    tokenizer->init_ok = 1;

  }

  const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id){

    if(tokenizer->init_ok == 0){
      return NULL;
    }

    if(token_id < tokenizer->vocab_size){
      return tokenizer->token_table[token_id];
    } else {
      cout<<"Invalid token id: "<< token_id;

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
  
}
