#pragma once

#include <x86intrin.h>
#include "defs.hpp"

class MLP3 {
  public:
    MLP3();                                 // This is the constructor
    //~MLP3();                              // This is the destructor
    void set_weights(double* fc1_i, double* fc2_i, double* fc3_i,
                     double* fc1_bias_i, double* fc2_bias_i,
                     double* fc3_bias_i);

    double* forward(double* obs) {
      base_forward(obs, act, z1, z2);
      return act;
    }

    void forward(double* obs, double* out) {
      base_forward(obs, out, z1, z2);
    }

    void base_forward(double* obs, double* out, double* o1, double* o2);

  private:
    double* fc1;
    double* fc2;
    double* fc3;
    double* fc1_bias;
    double* fc2_bias;
    double* fc3_bias;

    //intermediate variables
    double z1[h1];
    double z2[h2];
    double act[act_dim];

    int i, j;
    __m256d vz, vo;
    #if do_mlp_output_tanh == True
      __m128d vz2, vo2;
    #endif
};
