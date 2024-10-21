/*
 * GPT2 Implementation using CPP
 */

// CPP libraries

#include <math.h>
#include <string.h>
#include <time.h>

// OMP
#ifdef OMP
#include <omp.h>
#endif

// Import self defined libs
#include "libTran/dataloader.h"
#include "libTran/tokenizer.h"
#include "libTran/utils.h"

using namespace std;

// Each Layers forward and Backward passes

// Forward pass
void encoder_forward(float* out, int* inp, float* wte, float* wpe, int B, int T, int C){
    for(int b=0;b<B;b++){
      for(int t=0;t<T;t++){
        float* out_bt = out + b * T * C + t * C;
        int ix = inp[b * T + t];
        float* wte_ix = wte + ix * C;
        float* wpe_t = wpe + t * C;

        for(int i=0;i<C;i++){
            out_bt[i] = wte_ix[i] + wpe_t[i];
        }
    }
  }
}

// Backward Pass
void encoder_backward(float* dwte, float* dwpe, float* dout, int* inp, int B, int T, int C){
  
  for(int b=0;b<B;b++){
    for(int t=0;t<T;t++){
      float* dout_bt = dout + b * T * C + t * C;
      int ix = inp[b * T + t];
      float* dwte_ix = dwte + ix * C;
      float* dwpe_t = dwpe + t * C;

      for(int i=0;i<C;i++){
        float d = dout_bt[i];
        dwte_ix[i] += d;
        dwpe_t[i] += d;
      }
    }
  }
    
}


// LayerNorm
void layernorm_forward(float* out, float* mean, float* rstd, float* inp, float* weight, float* bias, int B, int T, int C){

  float eps = 1e-5f;
  for(int b=0;b<B;b++){
    for(int t=0;t<T;t++){

        float* x = inp + b * T * C + t * C;

        float m = 0.0f;
        for(int i=0;i<C;i++){
          m += x[i];
        }

        m = m / C;
        
        float v = 0.0f;
        for(int i=0;i < C;i++){
          float xshift = x[i] - m;
          v += xshift * xshift;
        }
        v = v / C;

        float s = 1.0f / sqrtf(v + eps);

        float* out_bt = out + b * T * C + t * C;
        for(int i=0;i<C;i++){
          float n = (s * (x[i] - m));
          float o = n * weights[i] + bias[i];
          out_bt[i] = o;
        }
        mean[b * T + t] = m;
        rstd[b * T + t] = s;
    }
  }
}

void layernorm_backward(float* dinp, float* dweight, float* dbias, float* dout, float* inp, float* weight, float* mean, float* rstd, int B, int T, int C){

  for(int b = 0;b < B;b++){
    for(int t=0;t<T;t++){
      float* dout_bt = dout + b * T * C + t * C;
      float* inp_bt = inp + b * T * C + t * C;
      float* dinp_bt = dinp + b * T *C + t * C;
      float mean_bt = mean[b * T + t];
      float rstd_bt = rstd[b * T + t];

      float dnorm_mean = 0.0f;
      float dnorm_norm_mean = 0.0f;
      for(int i=0;i<C;i++){
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt; 
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i; 
        dnorm_norm_mean += dnorm_i * norm_bti;
      }     
     
      dnorm_mean = dnorm_mean / C;
      dnorm_norm_mean = dnorm_norm_mean / C;

      for(int i=0;i < C;i++){
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];

        dbias[i] += dout_bt[i];
        dweight[i] += norm_bti * dout_bt[i];

        float dval = 0.0f;
        dval += dnorm_i;
        dval -= dnorm_mean;
        dval -= norm_bti * dnorm_norm_mean; 
        dval *= rstd_bt;
        dinp_bt += dval; 
      }
    }
  }
}

void matmul_forward_naive(float* out, const float* inp, const float* weight, const float* bias, int B, int T, int C, int OC){

  #pragma omp parallel for collapse(2)
  for(int b=0;b<B;b++){
    for(int t=0;t<T;t++){
      int bt = b * T + t;
      for(int o=0;o<OC;o++){
        float val = (bias != NULL) ? bias[o] : 0.0f;
        for(int i=0;i<C;i++){
          val += inp[bt * C + i] * weight[o*C + i];
        }
        out[bt * OC + o] = val;
      }
    }
  }
}


void matmul_forward(float* out, const float* inp, const float* weight, const float* bias, int B, int T, int C, int OC){

  const int LOOP_UNROLL = 8;
  if(B * T % LOOP_UNROLL != 0){
    matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
    return;
  }


  #pragma omp prallel for
  for(int obt = 0; obt < B * T;obt += LOOP_UNROLL){
    for(int o=0;o<OC;o++){

        float result[LOOP_UNROLL];

        for(int ibt = 0;ibt<LOOP_UNROLL;ibt++){
          result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
        }

        for(int i=0;i<C;i++){
          float w = weight[i + o * C];

          for(int ibt = 0;ibt < LOOP_UNROLL; ibt++){
            int bt = obt + ibt;
            result[ibt] += inp[bt * C + i] * w;
          }
        }

        for(int ibt=0;ibt<LOOP_UNROLL;ibt++){
          int bt = obt + ibt;
          out[bt * OC + o] = result[ibt];
        }
    }
  }

}

void matmul_backward(float* dinp, float* dweight, float* dbias, const float* dout, const float* inp, const float* weight, int B, int T, int C, int OC){

  #pragma omp parallel for collapse(2)
  for(int b=0;b<B;b++){
    for(int t=0;t<T;t++){
      const float* dout_bt = dout + b * T * OC + t * OC;
      float* dinp_bt = dinp + b * T * C + t * C;
      
      for(int o=0;o<OC;o++){
        const float* wrow = weight + o*C;
        float d = dout_bt[o];
        for(int i=0;i<C;i++){
          dinp_bt[i] += wrow[i] * d;
        }
      }
    }
  }

  #pragma omp parallel for
  for(int o=0;o<OC;o++){
    for(int b=0;b<B;b++){
      for(int t=0;t<T;t++){
        const float* dout_bt = dout + b * T * OC + t * OC;
        const float* inp_bt = inp + b * T * C + t * C;

        float* dwrow = dweight + o*C;
        float d = dout_bt[o];

        if(dbias != NULL) { dbias[o] += d; }
      
        for(int i=0;i<C;i++){
          dwrow[i] += inp_bt[i] * d;
        }
      }
    }
  }
}

//GPT 2 Definition




int main() { 
  

  return 0; 
}
