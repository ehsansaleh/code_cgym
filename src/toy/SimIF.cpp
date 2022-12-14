#include "iostream"
#include <sstream>
#include "mujoco.h"
#include "assert.h"
#include <cmath>
#include <chrono>
#include <string>
#include <cstdlib>    // std::getenv
#include <stdlib.h>   // realpath
#include <limits.h>   // PATH_MAX
#include <stdio.h>
#include <string.h>   // strcpy, strcat
#include <iomanip>    // std::setprecision, std::setw
#include <iostream>   // std::cout, std::fixed
#include <stdexcept>  // std::runtime_error

#include <bits/stdc++.h>

#if xml_type == content_type
  #include "mj_xml.hpp"    // xml_bytes, xml_content
#endif

#include "defs.hpp"
#include "SimIF.hpp"

/////////////////////////////////////////////////////
//////////////// Static Assertions //////////////////
/////////////////////////////////////////////////////

static_assert(max_steps > 1, "max_steps set incorrectly");

/////////////////////////////////////////////////////
////////// SimInterface Class Definition ////////////
/////////////////////////////////////////////////////

// Member functions definitions including constructor
SimInterface::SimInterface(void) {
  #if defined(stdout_pipe_file)
    freopen(stdout_pipe_file,"w",stdout);
  #endif
}

double* SimInterface::reset(double* init_args_double, int* init_args_int) {
  step_count = 0;
  state = init_args_double[0];
  return &state;
}

void SimInterface::step(double action_raw[act_dim],
                        double** next_obs,
                        double* reward,
                        bool* done) {
  #ifdef Debug_step
    std::cout << "Step outer " << step_count << ":" << std::endl;
  #endif

  if (action_raw[0] > 1.0) {
    state += 0.01;
  } else if (action_raw[0] < -1.0) {
    state -= 0.01;
  } else {
    state += action_raw[0] * 0.01;
  }
  if (state > 1.0)
    state = 1.0;
  else if (state < -1.0)
    state = -1.0;

  *reward = - (state * state);
  *next_obs = &state;
  step_count++;

  *done = (step_count >=  max_steps);

  #ifdef Debug_step
    std::cout << "Reward: " << *reward << std::endl;
    std::cout << "Done:   " << *done << std::endl;
    std::cout << "--------------------" << std::endl;
  #endif
}

SimInterface::~SimInterface(void) {
  // delete everything
}

void write_opt(const std::string key_str, double val, char** key_write_ptr,
               double** val_write_ptr, char* key_write_ptr_max,
               double* val_write_ptr_max) {
  int n = key_str.length() + 1;
  if ((*key_write_ptr + n) >= key_write_ptr_max)
    throw std::runtime_error("the destination char array is too small");
  if ((*val_write_ptr + 1) >= val_write_ptr_max)
    throw std::runtime_error("the destination char array is too small");
  strcpy(*key_write_ptr, key_str.c_str());
  **val_write_ptr = val;
  *key_write_ptr += n;
  *val_write_ptr = *val_write_ptr + 1;
}

void get_build_options(char* keys, double* vals, int keys_len, int vals_len) {
  char* key_p;
  double* val_p;
  char* a = keys + keys_len;
  double* b = vals + vals_len;
  key_p = keys;
  val_p = vals;
  // enumerations and constants
  write_opt("True", True, &key_p, &val_p, a, b);
  write_opt("False", False, &key_p, &val_p, a, b);
  write_opt("tanh_activation", tanh_activation, &key_p, &val_p, a, b);
  write_opt("relu_activation", relu_activation, &key_p, &val_p, a, b);
  write_opt("PI", PI, &key_p, &val_p, a, b);
  // settings
  write_opt("max_steps", max_steps, &key_p, &val_p, a, b);
  write_opt("h1", h1, &key_p, &val_p, a, b);
  write_opt("h2", h2, &key_p, &val_p, a, b);
  write_opt("activation", activation, &key_p, &val_p, a, b);
  write_opt("do_mlp_output_tanh", do_mlp_output_tanh, &key_p, &val_p, a, b);
  write_opt("mlp_output_scaling", mlp_output_scaling, &key_p, &val_p, a, b);

  write_opt("obs_dim", obs_dim, &key_p, &val_p, a, b);
  write_opt("act_dim", act_dim, &key_p, &val_p, a, b);
  write_opt("n_init_args_double", n_init_args_double, &key_p, &val_p, a, b);
  write_opt("n_init_args_int", n_init_args_int, &key_p, &val_p, a, b);

  write_opt("fc1_size", fc1_size, &key_p, &val_p, a, b);
  write_opt("fc2_size", fc2_size, &key_p, &val_p, a, b);
  write_opt("fc3_size", fc3_size, &key_p, &val_p, a, b);
  write_opt("h1_div_4", h1_div_4, &key_p, &val_p, a, b);
  write_opt("h2_div_4", h2_div_4, &key_p, &val_p, a, b);
  write_opt("actdim_div_2", actdim_div_2, &key_p, &val_p, a, b);

  write_opt("Shared_Obj", Shared_Obj, &key_p, &val_p, a, b);
  write_opt("SimIF_CPP", SimIF_CPP, &key_p, &val_p, a, b);
  write_opt("MlpIF_CPP", MlpIF_CPP, &key_p, &val_p, a, b);
  write_opt("Rollout_CPP", Rollout_CPP, &key_p, &val_p, a, b);
  write_opt("__MAINPROG__", __MAINPROG__, &key_p, &val_p, a, b);
}

// This will be used by python's ctypes library for binding
extern "C"
{
  void rollout_get_build_options(char* keys, double* vals, char* xml_var,
    int keys_len, int vals_len, int xml_var_len) {
    get_build_options(keys, vals, keys_len, vals_len);
    #if xml_type == file_path_type
      strcpy(xml_var, xml_path);
    #elif xml_type == content_type
      for (int i=0; i<(xml_bytes-5); i++) {
        xml_var[i] = xml_content[i];
        if (i >= xml_var_len) {
          throw std::runtime_error("xml_var_len is too small");
          break;
        }
      }
    #endif
  }
}

#if __MAINPROG__ == SimIF_CPP

  std::chrono::system_clock::time_point tm_start;
  mjtNum gettm(void)
  {
      std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - tm_start;
      return elapsed.count();
  }

  int main(int argc, char *argv[]){
    #ifdef Debug_main
      std::cout << std::fixed << std::setprecision(8);
      std::cout << "Step 1) Creating the SimInterface" << std::endl;
      std::cout << "  --> Started Creating a SimInterface instance!" << std::endl;
    #endif
    SimInterface simiface;
    #ifdef Debug_main
      std::cout << "  --> Done Creating a SimInterface instance!" << std::endl;
      std::cout << "--------------------" << std::endl;
      std::cout << "Step 2) Resetting the SimInterface" << std::endl;
      std::cout << "  --> Started Resetting the SimInterface!" << std::endl;
    #endif

    double init_args_double[n_init_args_double];
    int init_args_int[n_init_args_int];
    init_args_double[0] = 0.5; // theta_hip

    double* init_state;
    init_state = simiface.reset(init_args_double, init_args_int);

    #ifdef Debug_main
      std::cout << "  -> Done Resetting the SimInterface!" << std::endl;
      std::cout << "init_state[0] = " << init_state[0] << std::endl;
      std::cout << "--------------------" << std::endl;
    #endif

    #ifdef Debug_main
      std::cout << "Step 3) Stepping the SimInterface" << std::endl;
      std::cout << "  --> Started Stepping the SimInterface!" << std::endl;
    #endif

    double action_raw[act_dim] = {-0.25};
    double* next_state;
    double reward;
    bool done;
    double sim_time = gettm();
    for (int i=0; i < max_steps; i++)
      simiface.step(action_raw, &next_state, &reward, &done);
    sim_time = gettm() - sim_time;
    #ifdef Debug_main
      std::cout << "  --> Done Stepping the SimInterface!" << std::endl;
      std::cout << "  --> Simulation Time: " << sim_time << std::endl;
      std::cout << "next_state[0] = " << next_state[0] << std::endl;
      std::cout << "--------------------" << std::endl;
    #endif

    return 0;
  }

#endif
