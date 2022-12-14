#include <stdio.h>
#include <x86intrin.h>
#include <sleef.h>
#include <iostream>
#include <chrono>
#include <cblas.h>                /* C BLAS          BLAS  */
#include "MlpIF.hpp"
#include "defs.hpp"

/////////////////////////////////////////////////////
//////////////// Static Assertions //////////////////
/////////////////////////////////////////////////////
static_assert(h1 % 4 == 0, "We only support h1=4*k currently.");
static_assert(h2 % 4 == 0, "We only support h2=4*k currently.");
static_assert(h1_div_4 == (h1 / 4), "h1_div_4 set incorrectly");
static_assert(h2_div_4 == (h2 / 4), "h2_div_4 set incorrectly");
static_assert(fc1_size == (obs_dim * h1), "fc1_size set incorrectly");
static_assert(fc2_size == (h1      * h2), "fc2_size set incorrectly");
static_assert(fc3_size == (h2 * act_dim), "fc3_size set incorrectly");

#if do_mlp_output_tanh == True
  static_assert((act_dim == 1) || (act_dim % 2 == 0), "We only support act_dim=2*k currently.");
  static_assert((act_dim == 1) || (actdim_div_2 == (act_dim / 2)), "actdim_div_2 set incorrectly");
#endif


MLP3::MLP3(){
  // no need to do anything!
}

void MLP3::set_weights(double* fc1_ptr, double* fc2_ptr, double* fc3_ptr,
                       double* fc1_bias_ptr, double* fc2_bias_ptr,
                       double* fc3_bias_ptr){
  fc1 = fc1_ptr;
  fc2 = fc2_ptr;
  fc3 = fc3_ptr;
  fc1_bias = fc1_bias_ptr;
  fc2_bias = fc2_bias_ptr;
  fc3_bias = fc3_bias_ptr;
}

void MLP3::base_forward(double* obs, double* out, double* o1, double* o2) {
  // cblas_dcopy(size,    source,      src_stride,   destination,   dst_stride);
  // cblas_dgemv :: y <- alpha  *  A  *  xT +  beta  *  y
  // cblas_dgemv(row_order, transform, Arows, Acols, alpha, A, Arows, X, incX, beta, Y, incY);
  cblas_dcopy(h1, fc1_bias, 1, z1, 1);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, h1, obs_dim, 1, fc1, obs_dim, obs, 1, 1, z1, 1);
  #if activation == tanh_activation
    for (i=0; i < h1_div_4; i++){
      vz = _mm256_loadu_pd(z1 + 4 * i);
      vo = Sleef_tanhd4_u10(vz);
      _mm256_storeu_pd(o1 + 4 * i, vo);
    };
  #elif activation == relu_activation
    for (i=0; i < h1; i++){
      o1[i] = z1[i] > 0 ? z1[i] : 0;
    };
  #else
    #error "activation function not implemented"
  #endif
  cblas_dcopy(h2, fc2_bias, 1, z2, 1);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, h2, h1, 1, fc2, h1, o1, 1, 1, z2, 1);
  #if activation == tanh_activation
    for (i=0; i < h2_div_4; i++){
      vz = _mm256_loadu_pd(z2 + 4 * i);
      vo = Sleef_tanhd4_u10(vz);
      _mm256_storeu_pd(o2 + 4 * i, vo);
    };
  #elif activation == relu_activation
    for (i=0; i < h2; i++){
      o2[i] = z2[i] > 0 ? z2[i] : 0;
    };
  #else
    #error "activation function not implemented"
  #endif
  cblas_dcopy(act_dim, fc3_bias, 1, out, 1);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, act_dim, h2, 1, fc3, h2, o2, 1, 1, out, 1);

  #if do_mlp_output_tanh == True
    #if act_dim == 1
      double tmp1[2];
      tmp1[0] = out[0];
      tmp1[1] = out[0];
      vz2 = _mm_loadu_pd(tmp1);
      vo2 = Sleef_tanhd2_u10(vz2);
      _mm_storeu_pd(tmp1, vo2);
      out[0] = tmp1[0];
    #else
      for (i=0; i < actdim_div_2; i++){
        vz2 = _mm_loadu_pd(out + 2 * i);
        vo2 = Sleef_tanhd2_u10(vz2);
        _mm_storeu_pd(out + 2 * i, vo2);
      };
    #endif
  #endif

  #if mlp_output_scaling != 1
    for (i=0; i < act_dim; i++){
      out[i] *= mlp_output_scaling;
    };
  #endif
}

#if __MAINPROG__ == MlpIF_CPP
  std::chrono::system_clock::time_point tm_start;
  double gettm(void)
  {
      std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - tm_start;
      return elapsed.count();
  }

  int main() {
    double fc1[obs_dim * h1] = {0};
    double fc2[h1      * h2] = {0};
    double fc3[h2 * act_dim] = {0};
    double fc1_bias[h1] = {0};
    double fc2_bias[h2] = {0};
    double fc3_bias[act_dim] = {0};

    double obs[obs_dim] = {0};
    double* act;
    double sim_time;
    int i;
    MLP3 net;

    std::cout << "FeedForward Neural Network Using CBLAS and SLEEF" << std::endl << std::endl;

    std::cout << "Step 1) Creating the Neural Network!" << std::endl;
    net = MLP3();
    std::cout << "  -->   Done Instantiating the Neural Network!" << std::endl;
    std::cout << "--------------------" << std::endl;

    std::cout << "Step 2) Setting the Neural Weights!" << std::endl;
    net.set_weights(fc1, fc2, fc3, fc1_bias, fc2_bias, fc3_bias);
    std::cout << "  -->   Done Setting the Neural Weights!" << std::endl;
    std::cout << "--------------------" << std::endl;

    std::cout << "Step 3) Applying Many Forwards!" << std::endl;
    sim_time = gettm();
    for (i=0; i<8000; i++)
      act = net.forward(obs);
    sim_time = gettm() - sim_time;

    std::cout << "  -->   Done Applying Many Forwards!" << std::endl;
    std::cout << "  -->   Inference Time: " << sim_time << std::endl;
    std::cout << "--------------------" << std::endl;
    std::cout << " The output actions are: " << std::endl;
    for (i=0; i<act_dim; i++)
      std::cout << act[i] << ", ";
    std::cout << std::endl;
  }

#endif
