/////////////////////////////////////////////////////
////////////////// Debug Options ////////////////////
/////////////////////////////////////////////////////

#define Debug_main
#undef Debug_step_inner
#undef Debug_step_outer
#undef Debug_reward

/////////////////////////////////////////////////////
/////////////// Argument Definitions ////////////////
/////////////////////////////////////////////////////

#define True true
#define False false

#define compliant_v1 0
#define compliant_v2 1
#define noncompliant 2

#define torque 0
#define jointspace_pd 1
#define workspace_pd 2

#define file_path_type 0
#define content_type 1

#define tanh_activation 0
#define relu_activation 1

#define dense 0
#define sparse 1
#define simplified 2

#define dyno 0
#define naive 1
#define legacy 2

#define PI 3.14159265358979323846

/////////////////////////////////////////////////////
/////////////// SimIF Options Defs //////////////////
/////////////////////////////////////////////////////
#define inner_loop_rate 4000
#define outer_loop_rate 4000
#define ground_contact_type compliant_v2
#define do_obs_noise True
#define jumping_obs False
#define do_jump False
#define physics_timestep 0.00025
#define torque_delay 0.001
#define observation_delay 0.001
#define time_before_reset 2.0
#define action_type torque
#define use_motor_model True
#define motor_saturation_model dyno
#define stand_reward dense
#define add_act_sm_r False

#ifndef xml_type
  #define xml_type file_path_type
#endif
#ifndef xml_file
  #define xml_file "./leg.xml"
#endif

#define motor_root_real -1606.059046034205
#define motor_root_imag 4190.778464532961
#define hip_kP 1.25
#define hip_kD 0.05
#define knee_kP 1.25
#define knee_kD 0.05
#define maxabs_tau 10
#define output_torque_scale 1
#define check_mj_unstability True
#define timed_jump True

#define maxabs_omega_for_tau 21.66
#define tau_omega_slope -0.397

#define omega_noise_scale 1
#define noise_rows 401

//original dense reward
#define jump_push_time 0.2
#define jump_fly_time 0.2
#define torque_smoothness_coeff 2000
#define posture_height 0.1
#define max_leg_vel 100
#define jump_vel_coeff 10000
#define max_reach 0.28
#define jump_vel_reward False
#define turn_off_constraints False
#define omega_hip_maxabs 100
#define omega_knee_maxabs 100
#define constraint_penalty 100
#define theta_hip_bounds_low -3.9269908169872414
#define theta_hip_bounds_high 1.5707963267948966
#define theta_knee_bounds_low -2.705260340591211
#define theta_knee_bounds_high -0.6108652381980153
#define hip_minimum_z 0.05
#define knee_minimum_z 0.02

//dyno model parameters
#define rpm_per_omega 223.06583865964922
#define fcn_k50_co_0 0.3518
#define fcn_k50_co_1 -1.677
#define fcn_k50_co_2 186.0537
#define max_cmd_dyno 2000
#define rpm_for_max_cmd_dyno 4830
#define cmd_per_rpm_slope_dyno -0.368
#define model_coeff_dyno_0 (-2.48090029e-01)
#define model_coeff_dyno_1 (5.95753139e-03)
#define model_coeff_dyno_2 (1.11332764e-04)
#define model_coeff_dyno_3 (-5.66257261e-07)
#define model_coeff_dyno_4 (-1.71874239e-07)
#define model_coeff_dyno_5 (-1.17675521e-08)

//disabled
#define contact_obs False
#define extra_obs_dim False
#define filter_state False
#define filter_action False
#define action_is_delta False
//disabled reward stuff
#define add_torque_penalty_on_air False
#define add_omega_smoothness_reward False

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

#if !defined(ground_contact_type)                                || \
    !((ground_contact_type == compliant_v1)                      || \
      (ground_contact_type == compliant_v2)                      || \
      (ground_contact_type == noncompliant))
  #error "Undefined ground_contact_type."
#endif

#if !defined(action_type) || !((action_type == torque)           || \
                               (action_type == jointspace_pd)    || \
                               (action_type == workspace_pd))
  #error "Undefined action_type."
#endif

#if !defined(do_obs_noise) || !((do_obs_noise == True)           || \
                                (do_obs_noise == False))
  #error "Undefined do_obs_noise."
#endif

#if !defined(do_jump) || !((do_jump == True)                     || \
                           (do_jump == False))
  #error "Undefined do_jump."
#endif

#if !defined(use_motor_model) || !((use_motor_model == True)     || \
                                   (use_motor_model == False))
  #error "Undefined use_motor_model."
#endif

#if !defined(contact_obs) || !((contact_obs == True)             || \
                               (contact_obs == False))
  #error "Undefined contact_obs."
#endif

#if !defined(jumping_obs) || !((jumping_obs == True)             || \
                               (jumping_obs == False))
  #error "Undefined jumping_obs."
#endif

#if !defined(extra_obs_dim) || !((extra_obs_dim == True)         || \
                                 (extra_obs_dim == False))
  #error "Undefined extra_obs_dim."
#endif

#if !defined(filter_state) || !((filter_state == True)           || \
                                (filter_state == False))
  #error "Undefined filter_state."
#endif

#if !defined(filter_action) || !((filter_action == True)         || \
                                 (filter_action == False))
  #error "Undefined filter_action."
#endif

#if !defined(action_is_delta) || !((action_is_delta == True)     || \
                                   (action_is_delta == False))
  #error "Undefined action_is_delta."
#endif

#if !defined(stand_reward) || !((stand_reward == simplified)     || \
                                (stand_reward == dense)          || \
                                (stand_reward == sparse))
  #error "Undefined stand_reward."
#endif

#if !defined(timed_jump) || !((timed_jump == True)               || \
                              (timed_jump == False))
  #error "Undefined timed_jump."
#endif



#if !defined(jump_vel_reward) || !((jump_vel_reward == True)     || \
                                   (jump_vel_reward == False))
  #error "Undefined jump_vel_reward."
#endif

#if !defined(check_mj_unstability)                               || \
    !((check_mj_unstability == True)                             || \
      (check_mj_unstability == False))
  #error "Undefined timed_jump."
#endif

#if !defined(turn_off_constraints)                               || \
      !((turn_off_constraints == True)                           || \
        (turn_off_constraints == False))
  #error "Undefined turn_off_constraints."
#endif

#if !defined(motor_saturation_model)                             || \
      !((motor_saturation_model == dyno)                         || \
        (motor_saturation_model == naive)                        || \
        (motor_saturation_model == legacy))
  #error "Undefined motor_saturation_model."
#endif


#if !defined(add_torque_penalty_on_air)                          || \
    !((add_torque_penalty_on_air == True)                        || \
      (add_torque_penalty_on_air == False))
  #error "Undefined add_torque_penalty_on_air."
#endif

#if !defined(add_omega_smoothness_reward)                        || \
    !((add_omega_smoothness_reward == True)                      || \
      (add_omega_smoothness_reward == False))
  #error "Undefined add_omega_smoothness_reward."
#endif

#if !defined(add_act_sm_r) || !((add_act_sm_r == True)           || \
                                (add_act_sm_r == False))
  #error "Undefined add_act_sm_r."
#endif

#if jump_vel_coeff < 0
  #error "jump_vel_coeff should be positive"
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

#ifdef obs_history_taps
  #error "obs_history_taps is not implemented in this C++ wrapper."
#endif

#ifdef obs_history_len
  #error "obs_history_len is not implemented in this C++ wrapper."
#else
  #define obs_history_len_int 1
#endif

#if contact_obs == True
  #error "contact_obs is not implemented in this C++ wrapper."
#else
  #define contact_obs_int 0
#endif

#if jumping_obs == True
  #define jumping_obs_int 1
#else
  #define jumping_obs_int 0
#endif

#if extra_obs_dim == True
  #error "extra_obs_dim is not implemented in this C++ wrapper."
#else
  #define extra_obs_dim_int 0
#endif

#if filter_state == True
  #error "filter_state is not implemented in this C++ wrapper."
#endif

#if filter_action == True
  #error "filter_action is not implemented in this C++ wrapper."
#endif

#if timed_jump == False
  #error "timed_jump == False is not implemented in this C++ wrapper."
#endif

#if action_type == workspace_pd
  #error "workspace_pd is not implemented in this C++ wrapper."
#endif

#if action_is_delta == True
  #error "action_is_delta is not implemented in this C++ wrapper."
#endif

#if stand_reward == dense
  #define RewardGiver OriginalRewardGiver
#elif stand_reward == sparse
  #error "sparse stand_reward is not implemented in this C++ wrapper."
#else
  #define RewardGiver SimplifiedRewardGiver
#endif

#if jump_vel_reward == True
  #error "jump_vel_reward is not implemented in this C++ wrapper."
#endif

#if add_torque_penalty_on_air == True
  #error "add_torque_penalty_on_air is not implemented in this C++ wrapper."
#endif

#if add_omega_smoothness_reward == True
  #error "add_omega_smoothness_reward is not implemented in this C++ wrapper."
#endif

#if motor_saturation_model == dyno
  #define use_dyno_model True
  #define use_naive_model False
  #define use_legacy_model False
#elif motor_saturation_model == naive
  #define use_dyno_model False
  #define use_naive_model True
  #define use_legacy_model False
#elif motor_saturation_model == legacy
  #define use_dyno_model False
  #define use_naive_model False
  #define use_legacy_model True
#else
  #error "unknown motor_saturation_model"
#endif

// Functional definitions
// NOTE: After some investigation into the compiled code, I found out that new GCC
//       versions are smart enough to replace these values with constant numbers.

#define inner_step_time ((double) (1.0 / inner_loop_rate))
#define inner_loops_per_outer_loop (inner_loop_rate / outer_loop_rate)
#define physics_loops_per_inner_loop ((int) (inner_step_time / physics_timestep))
#define extra_physics_loops_per_inner_loop (physics_loops_per_inner_loop - 1)
#define obs_dim (4 * obs_history_len_int + contact_obs_int + jumping_obs_int + extra_obs_dim_int)
#define act_dim 2
#define torque_delay_steps ((int) round(torque_delay / inner_step_time))
#define observation_delay_steps ((int) round(observation_delay / inner_step_time))
#define trq_dly_buflen (torque_delay_steps + 1)
#define obs_dly_buflen (observation_delay_steps + 1)
#define done_inner_steps ((int) round(time_before_reset / inner_step_time))
#define neg_omega_hip_maxabs (-1 * omega_hip_maxabs)
#define neg_omega_knee_maxabs (-1 * omega_knee_maxabs)
#define neg_fcn_k50_co_0 (-1 * fcn_k50_co_0)

#define fc1_size (obs_dim * h1)
#define fc2_size (h1      * h2)
#define fc3_size (h2 * act_dim)
#define h1_div_4 (h1/4)
#define h2_div_4 (h2/4)
#define actdim_div_2 (act_dim/2)
#define n_init_args_double 7
#define n_init_args_int 1

/////////////////////////////////////////////////////
///////// C++ Wrapper Specific Definitions //////////
/////////////////////////////////////////////////////

#define mjstep1_after_mjstep     0
#define separate_mjstep1_mjstep2 1
#define delay_valid_obs          2

#define mjstep_order separate_mjstep1_mjstep2

#if mjstep_order == mjstep1_after_mjstep
  #define rew_dly_buflen (obs_dly_buflen)
#elif mjstep_order == separate_mjstep1_mjstep2
  #define rew_dly_buflen (obs_dly_buflen - 1)
#elif mjstep_order == delay_valid_obs
  #define rew_dly_buflen (obs_dly_buflen - 1)
#else
  #error "Unknown mjstep_order"
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

// #define stdout_pipe_file "../cpp_output.txt"
// Piping stdout happens in the SimIneterface constructor in SimIF.cpp
