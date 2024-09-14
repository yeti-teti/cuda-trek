#ifndef LAYERPASS_H
#define LAYERPASS_H

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

#endif
