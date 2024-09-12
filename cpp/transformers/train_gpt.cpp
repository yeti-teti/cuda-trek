/*
 * GPT2 Implementation using CPP
 */

#include <math.h>
#include <string.h>
#include <time.h>

#ifdef OMP
#include <omp.h>
#endif

// Import self defined libs
#include "libTran/dataloader.h"
#include "libTran/tokenizer.h"
#include "libTran/utils.h"

using namespace std;

// Each Layers forward and Backward passes
void encoder_forward();
void encoder_backward();
void layernorm_forward();
void layernorm_backward();
void matmul_forward_naive();
void matmul_forward();
void matmul_backward();
void attention_forward();
void attention_backward();
void gelu_forward();
void gelu_backward();
void residual_forward();
void residual_backward();
void softmax_forward();
void crossentropy_forward();
void crossentropy_softmax_backward();

// GPT 2 Definition

int main() { return 0; }
