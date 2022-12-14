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

#define take_average 0
#define take_last 1

#define PI 3.14159265358979323846

#define InvertedPendulum 0
#define InvertedDoublePendulum 1
#define HalfCheetah 2
#define Swimmer 3
#define Hopper 4
#define Walker2d 5
#define Ant 6
#define Humanoid 7
#define HumanoidStandup 8
#define Reacher 9

/////////////////////////////////////////////////////
/////////////// SimIF Options Defs //////////////////
/////////////////////////////////////////////////////

// environment-specific settings
#if ENVNAME == InvertedPendulum
  #define inner_loop_rate 50
  #define outer_loop_rate 25
  #define physics_timestep 0.02
  #define time_before_reset 40.0
  #define qpos_dim 2
  #define qvel_dim 2
  #define ctrl_dim 1
  #define obs_dim 4
#elif ENVNAME == InvertedDoublePendulum
  #define inner_loop_rate 100
  #define outer_loop_rate 20
  #define physics_timestep 0.01
  #define time_before_reset 50
  #define qpos_dim 3
  #define qvel_dim 3
  #define ctrl_dim 1
  #define obs_dim 11
#elif ENVNAME == HalfCheetah
  #define inner_loop_rate 100
  #define outer_loop_rate 20
  #define physics_timestep 0.01
  #define time_before_reset 50
  #define qpos_dim 9
  #define qvel_dim 9
  #define ctrl_dim 6
  #define obs_dim 17
#elif ENVNAME == Swimmer
  #define inner_loop_rate 100
  #define outer_loop_rate 25
  #define physics_timestep 0.01
  #define time_before_reset 40
  #define qpos_dim 5
  #define qvel_dim 5
  #define ctrl_dim 2
  #define obs_dim 8
#elif ENVNAME == Hopper
  #define inner_loop_rate 500
  #define outer_loop_rate 125
  #define physics_timestep 0.002
  #define time_before_reset 8
  #define qpos_dim 6
  #define qvel_dim 6
  #define ctrl_dim 3
  #define obs_dim 11
#elif ENVNAME == Walker2d
  #define inner_loop_rate 500
  #define outer_loop_rate 125
  #define physics_timestep 0.002
  #define time_before_reset 8
  #define qpos_dim 9
  #define qvel_dim 9
  #define ctrl_dim 6
  #define obs_dim 17
#elif ENVNAME == Ant
  #define inner_loop_rate 100
  #define outer_loop_rate 20
  #define physics_timestep 0.01
  #define time_before_reset 50
  #define qpos_dim 15
  #define qvel_dim 14
  #define ctrl_dim 8
  #define use_contact_forces True
  #define body_dim 14
  #define obs_cfrc_idx (qpos_dim + qvel_dim - 2)
  #define obs_cfrc_len (body_dim * 6)
  #if use_contact_forces == True
    #define obs_dim (27 + obs_cfrc_len)
  #elif use_contact_forces == False
    #define obs_dim 27
  #else
    #error "use_contact_forces undefined"
  #endif
#elif (ENVNAME == Humanoid) || (ENVNAME == HumanoidStandup)
  #define inner_loop_rate 333.3333333333333
  #define outer_loop_rate 66.66666666666666
  #define inner_loops_per_outer_loop 5
  #define physics_timestep 0.003
  #define time_before_reset 15
  #define qpos_dim 24
  #define qvel_dim 23
  #define ctrl_dim 17
  #define use_contact_forces True
  #define body_dim 14
  #if use_contact_forces == True
    #define obs_dim 376
  #elif use_contact_forces == False
    #error "you need to compute the obs_dim"
  #else
    #error "use_contact_forces undefined"
  #endif
  #define obs_cinert_len (body_dim * 10) // 140
  #define obs_cinert_idx (qpos_dim + qvel_dim - 2) // 45
  #define obs_cvel_len (body_dim * 6) // 84
  #define obs_cvel_idx (qpos_dim + qvel_dim + obs_cinert_len - 2) // 185
  #define obs_qfrcact_len (qvel_dim) // 23
  #define obs_qfrcact_idx (qpos_dim + qvel_dim + obs_cinert_len + obs_cvel_len - 2) // 208
  #define obs_cfrc_len (body_dim * 6) // 84
  #define obs_cfrc_idx (qpos_dim + qvel_dim + obs_cinert_len + obs_cvel_len + obs_qfrcact_len - 2) // 292
  #if (obs_dim - obs_cfrc_len) != obs_cfrc_idx
    #error "something went wrong!"
  #endif
#elif ENVNAME == Reacher
  #define inner_loop_rate 100
  #define outer_loop_rate 50
  #define physics_timestep 0.01
  #define time_before_reset 1
  #define qpos_dim 4
  #define qvel_dim 4
  #define ctrl_dim 2
  #define obs_dim 11
#else
  #error "ENVNAME not implemented"
#endif


#define torque_delay 0.00
#define observation_delay 0.00
#define action_type torque
#define kP 1.25 // only relavant when action_type == jointspace_pd
#define kD 0.05 // only relavant when action_type == jointspace_pd
#define check_mj_unstability True
#define agg_inner_rewards take_last
#define do_obs_noise False

#ifndef xml_type
  #define xml_type file_path_type
#endif
#ifndef xml_file
  #define xml_file "./leg.xml"
#endif

// reward related parameters
#if (ENVNAME == InvertedPendulum) || (ENVNAME == InvertedDoublePendulum)
  // nothing needed!
#elif ENVNAME == HalfCheetah
  #define forward_reward_weight 1   // used in HalfCheetah
  #define ctrl_cost_weight 0.1      // used in HalfCheetah
#elif ENVNAME == Swimmer
  #define forward_reward_weight 1   // used in Swimmer
  #define ctrl_cost_weight 0.0001   // used in Swimmer
#elif ENVNAME == Hopper
  #define forward_reward_weight 1   // used in Hopper
  #define ctrl_cost_weight 0.001    // used in Hopper
  #define healthy_reward_weight 1.0
  #define terminate_when_unhealthy True
  #define healthy_state_range_low -100.0
  #define healthy_state_range_high 100.0
  #define healthy_z_range_low 0.7
  #define healthy_angle_range_low -0.2
  #define healthy_angle_range_high 0.2
#elif ENVNAME == Walker2d
  #define forward_reward_weight 1   // used in Walker
  #define ctrl_cost_weight 0.001    // used in Walker
  #define healthy_reward_weight 1.0
  #define terminate_when_unhealthy True
  #define healthy_z_range_low 0.8
  #define healthy_z_range_high 2
  #define healthy_angle_range_low -1.0
  #define healthy_angle_range_high 1.0
#elif ENVNAME == Ant
  #define forward_reward_weight 1   // used in Ant
  #define ctrl_cost_weight 0.5
  #define contact_cost_weight 0.0005
  #define healthy_reward_weight 1.0
  #define terminate_when_unhealthy True
  #define healthy_z_range_low 0.2
  #define healthy_z_range_high 1.0
  #define contact_force_range_low -1.0
  #define contact_force_range_high 1.0
#elif ENVNAME == Humanoid
  #define forward_reward_weight 1.25
  #define ctrl_cost_weight 0.1
  #define contact_cost_weight 0.0000005
  #define contact_cost_range_high 10.0
  #define healthy_reward_weight 5.0
  #define terminate_when_unhealthy True
  #define healthy_z_range_low 1.0
  #define healthy_z_range_high 2.0
#elif ENVNAME == HumanoidStandup
  #define forward_reward_weight 1.0
  #define ctrl_cost_weight 0.1
  #define contact_cost_weight 0.0000005
  #define contact_cost_range_high 10.0
#elif ENVNAME == Reacher
  // nothing needed!
#else
  #error "ENVNAME not implemented"
#endif

// reward related hyper-parameters
#define sparse_rew_outer_steps 1
#define sparse_rew_outer_discrate 1.0

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

#if !defined(agg_inner_rewards)                                  || \
    !((agg_inner_rewards == take_average)                        || \
      (agg_inner_rewards == take_last))
  #error "Undefined agg_inner_rewards."
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

#if action_type == jointspace_pd
  #if act_dim != qpos_dim
    #error "cannot have meaningful jointspace_pd when act_dim != qpos_dim"
  #endif
  #if qvel_dim != qpos_dim
    #error "cannot have meaningful jointspace_pd when qvel_dim != qpos_dim"
  #endif
#endif

#if (ENVNAME == Ant) || (ENVNAME == Humanoid) || (ENVNAME == HumanoidStandup)
  #if !defined(use_contact_forces)                        || \
    !((use_contact_forces == True)                        || \
      (use_contact_forces == False))
    #error "Undefined use_contact_forces."
  #endif
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

#define act_dim (ctrl_dim)

#define inner_step_time ((double) (1.0 / inner_loop_rate))
#define outer_step_time ((double) (1.0 / outer_loop_rate))
#if !defined(inner_loops_per_outer_loop)
  #define inner_loops_per_outer_loop (inner_loop_rate / outer_loop_rate)
#endif
#define physics_loops_per_inner_loop ((int) (inner_step_time / physics_timestep))
#define extra_physics_loops_per_inner_loop (physics_loops_per_inner_loop - 1)
#define torque_delay_steps ((int) round(torque_delay / inner_step_time))
#define observation_delay_steps ((int) round(observation_delay / inner_step_time))
#define trq_dly_buflen (torque_delay_steps + 1)
#define obs_dly_buflen (observation_delay_steps + 1)
#define done_inner_steps ((int) round(time_before_reset / inner_step_time))

#define fc1_size (obs_dim * h1)
#define fc2_size (h1      * h2)
#define fc3_size (h2 * act_dim)
#define h1_div_4 (h1/4)
#define h2_div_4 (h2/4)
#define actdim_div_2 (act_dim/2)
#define n_init_args_double (qpos_dim + qvel_dim)
#define n_init_args_int (0)

/////////////////////////////////////////////////////
///////// C++ Wrapper Specific Definitions //////////
/////////////////////////////////////////////////////

#define mjstep1_after_mjstep     0
#define separate_mjstep1_mjstep2 1
#define delay_valid_obs          2
#define only_mjstep              3

#define mjstep_order only_mjstep

#if (mjstep_order == mjstep1_after_mjstep) || (mjstep_order == only_mjstep)
  #define rew_dly_buflen (obs_dly_buflen)
  #define done_dly_buflen (obs_dly_buflen)
#elif mjstep_order == separate_mjstep1_mjstep2
  #define rew_dly_buflen (obs_dly_buflen - 1)
  #define done_dly_buflen (obs_dly_buflen - 1)
#elif mjstep_order == delay_valid_obs
  #define rew_dly_buflen (obs_dly_buflen - 1)
  #define done_dly_buflen (obs_dly_buflen - 1)
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

// #define stdout_pipe_file "cpp_output.txt"
// Piping stdout happens in the SimIneterface constructor in SimIF.cpp
