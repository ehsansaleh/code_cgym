/////////////////////////////////////////////////////
////////////////// Debug Options ////////////////////
/////////////////////////////////////////////////////

#define Debug_main
#undef Debug_step_inner
#undef Debug_step_outer
#undef Debug_reward

#ifndef ENVNAME
  #error "ENVNAME must be defined"
#endif

/////////////////////////////////////////////////////
/////////////// Argument Definitions ////////////////
/////////////////////////////////////////////////////

#define True true
#define False false

#define torque 0
#define jointspace_pd 1
#define workspace_pd 2

#define file_path_type 0
#define content_type 1

#define tanh_activation 0
#define relu_activation 1

#define PI 3.14159265358979323846

#define Pendulum 0
#define NLRPendulum 1

#define NOFFT 0
#define FFTWLIB 1

/////////////////////////////////////////////////////
/////////////// SimIF Options Defs //////////////////
/////////////////////////////////////////////////////

#define inner_loop_rate 20
#define outer_loop_rate 20
#define physics_timestep 0.05
#define time_before_reset 10.0
#define act_dim 1
#define obs_dim 3

#if ENVNAME == Pendulum
  #define min_torque -2
  #define max_torque 2
#elif ENVNAME == NLRPendulum
  #define min_torque -80
  #define max_torque 80
#else
  #error "ENVNAME not implemented"
#endif

#define min_speed -8
#define max_speed 8
#define gravity_acc 10
#define link_mass 1
#define link_length 1

#define torque_delay 0.00
#define observation_delay 0.00
#define action_type torque
#define kP 1.25 // only relavant when action_type == jointspace_pd
#define kD 0.05 // only relavant when action_type == jointspace_pd
#define check_mj_unstability True
#define do_obs_noise False

#if ENVNAME == Pendulum
  #define FFTLIB NOFFT
#elif ENVNAME == NLRPendulum
  #define FFTLIB FFTWLIB
#else
  #error "FFTLIB not specified"
#endif

#if ENVNAME == NLRPendulum
  #define RewardGiver NLRRewardGiver
#elif ENVNAME == Pendulum
  #define RewardGiver StandupRewardGiver
#else
  #error "ENVNAME RewardGiver not implemented"
#endif

// reward-related hyper-parameters
#if ENVNAME == Pendulum
  // nothing!
#elif ENVNAME == NLRPendulum
  // NLRPendulum-v1 spec
  #define des_period 0.5
  #define des_offset (PI * 2. / 12.)
  #define des_amp 0.2
#endif

// stft related hyper-parameters
#define stft_inner_steps 200 // done_inner_steps
#define stft_gamma 0.99

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

#if !defined(action_type) || !((action_type == torque)           || \
                               (action_type == jointspace_pd)    || \
                               (action_type == workspace_pd))
  #error "Undefined action_type."
#endif

#if !defined(do_obs_noise) || !((do_obs_noise == True)           || \
                                (do_obs_noise == False))
  #error "Undefined do_obs_noise."
#endif

#if !defined(check_mj_unstability)                               || \
    !((check_mj_unstability == True)                             || \
      (check_mj_unstability == False))
  #error "Undefined check_mj_unstability."
#endif

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
/////////  Disabled/not-implemented options /////////
/////////////////////////////////////////////////////

#if action_type == workspace_pd
  #error "workspace_pd is not implemented in this C++ wrapper."
#endif

// Functional definitions
// NOTE: After some investigation into the compiled code, I found out that new GCC
//       versions are smart enough to replace these values with constant numbers.

#define phys_coeff1 (3*gravity_acc/(2*link_length))
#define phys_coeff2 (3./(link_mass*link_length*link_length))

#define inner_step_time ((double) (1.0 / inner_loop_rate))
#define outer_step_time ((double) (1.0 / outer_loop_rate))
#define inner_loops_per_outer_loop (inner_loop_rate / outer_loop_rate)
#define physics_loops_per_inner_loop ((int) (inner_step_time / physics_timestep))
#define extra_physics_loops_per_inner_loop (physics_loops_per_inner_loop - 1)
#define torque_delay_steps ((int) round(torque_delay / inner_step_time))
#define observation_delay_steps ((int) round(observation_delay / inner_step_time))
#define trq_dly_buflen (torque_delay_steps + 1)
#define obs_dly_buflen (observation_delay_steps + 1)
#define done_inner_steps ((int) round(time_before_reset / inner_step_time))
#define done_outer_steps ((int) round(time_before_reset / outer_step_time))
#define rew_dly_buflen (obs_dly_buflen)
#define done_dly_buflen (obs_dly_buflen)
#define nlr_buflen (stft_inner_steps + 1 - observation_delay_steps)
#define gamma_pow_buf (pow(stft_gamma, stft_inner_steps))

#define fc1_size (obs_dim * h1)
#define fc2_size (h1      * h2)
#define fc3_size (h2 * act_dim)
#define h1_div_4 (h1/4)
#define h2_div_4 (h2/4)
#define actdim_div_2 (act_dim/2)
#define n_init_args_double 2
#define n_init_args_int 0

#if ENVNAME == NLRPendulum
  // gamma, max_ep_len = 0.99, 200
  // r_coeff = 20. * (1. - gamma ** max_ep_len) /
  //           ((1.-gamma) * (gamma**max_ep_len))
  // #define r_coeff 12927.637360818304
  #define r_coeff (20.0 * (1.0 - gamma_pow_buf) / ((1.0 - stft_gamma) * (gamma_pow_buf)))
  #define dc_r_c1 (r_coeff / (nlr_buflen * PI))
  #define dc_r_c2 (des_offset * r_coeff / PI)

  #define des_freq_low (1.0 / des_period)
  #define des_freq_high (1.2 / des_period)
  #define good_ac_r_c1 (-0.1 * r_coeff)
  #define bad_ac_r_c1 (nlr_buflen * des_amp)
  #define bad_ac_r_c2 (10 * r_coeff)
  #define bad_ac_r_c3 (0.00000001 * r_coeff)

  #define des_freq_k_low (des_freq_low * nlr_buflen * physics_timestep)
  #define des_freq_k_high (des_freq_high * nlr_buflen * physics_timestep)

  #define fftout_len ( 1 + ((int) (nlr_buflen / 2)) )
#endif

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

// #define stdout_pipe_file "cpp_output.txt"
// Piping stdout happens in the SimIneterface constructor in SimIF.cpp
