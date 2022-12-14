/////////////////////////////////////////////////////
////////////////// Debug Options ////////////////////
/////////////////////////////////////////////////////

#define Debug_main
#undef Debug_step
#undef Debug_reward

/////////////////////////////////////////////////////
/////////////// Argument Definitions ////////////////
/////////////////////////////////////////////////////

// Add any general definitions here
#define True true
#define False false
#define PI 3.14159265358979323846

#define file_path_type 0
#define content_type 1

#define tanh_activation 0
#define relu_activation 1

/////////////////////////////////////////////////////
/////////////// SimIF Options Defs //////////////////
/////////////////////////////////////////////////////
#define max_steps 200

/////////////////////////////////////////////////////
////////////// MLP Module Definitions ///////////////
/////////////////////////////////////////////////////

#define h1 64  // Hidden Units in the MLP's 1st Layer
#define h2 64  // Hidden Units in the MLP's 2nd Layer
#define activation tanh_activation
#define do_mlp_output_tanh False
#define mlp_output_scaling 1

// For TD3 architecture, use the following settigns
// #define activation relu_activation
// #define do_mlp_output_tanh True
// #define mlp_output_scaling 10

/////////////////////////////////////////////////////
///////////  Checking Multi-choice options //////////
/////////////////////////////////////////////////////

#if !defined(activation) || !((activation == tanh_activation)    || \
                              (activation == relu_activation))
  #error "Undefined activation."
#endif

#if !defined(do_mlp_output_tanh)                          || \
    !((do_mlp_output_tanh == True)                        || \
      (do_mlp_output_tanh == False))
  #error "Undefined do_mlp_output_tanh."
#endif

/////////////////////////////////////////////////////
//////////////  Functional definitions //////////////
/////////////////////////////////////////////////////

// NOTE: After some investigation into the compiled code, I found out that new GCC
//       versions are smart enough to replace these values with constant numbers.

#define obs_dim 1
#define act_dim 1
#define n_init_args_double 1
#define n_init_args_int 0

#define fc1_size (obs_dim * h1)
#define fc2_size (h1      * h2)
#define fc3_size (h2 * act_dim)
#define h1_div_4 (h1/4)
#define h2_div_4 (h2/4)
#define actdim_div_2 (act_dim/2)

/////////////////////////////////////////////////////
////////////// Main Program Definitions /////////////
/////////////////////////////////////////////////////

#define Shared_Obj 0
#define SimIF_CPP 1
#define MlpIF_CPP 2
#define Rollout_CPP 3

#ifndef __MAINPROG__
  #define __MAINPROG__ SimIF_CPP
#endif

#if !defined(__MAINPROG__) || !((__MAINPROG__ == SimIF_CPP)                     || \
                                (__MAINPROG__ == MlpIF_CPP)                     || \
                                (__MAINPROG__ == Rollout_CPP)                   || \
                                (__MAINPROG__ == Shared_Obj))
  #error "Undefined __MAINPROG__."
#endif

// #define stdout_pipe_file "../cpp_output.txt"
// Piping stdout happens in the SimIneterface constructor in SimIF.cpp
