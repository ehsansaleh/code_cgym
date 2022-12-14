#include "iostream"
#include <sstream>
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

#include "defs.hpp"
#include "SimIF.hpp"

#if (FFTLIB == FFTWLIB)
  #include <fftw3.h>
#endif


/////////////////////////////////////////////////////
//////////////// Static Assertions //////////////////
/////////////////////////////////////////////////////

static_assert((inner_loop_rate % outer_loop_rate) == 0,
    "inner loop rate must be an integer multiple of outer loop rate");

static_assert(inner_loops_per_outer_loop == (inner_loop_rate / outer_loop_rate),
  "inner_loops_per_outer_loop set incorrectly");

static_assert(inner_step_time == (1.0 / inner_loop_rate),
  "inner_step_time set incorrectly");

static_assert(physics_loops_per_inner_loop == (inner_step_time / physics_timestep),
  "physics_loops_per_inner_loop set incorrectly");

static_assert(n_init_args_double == 2, "n_init_args_double set incorrectly");

static_assert(n_init_args_int == 0, "n_init_args_int set incorrectly");

static_assert(torque_delay_steps == (int) round(torque_delay / inner_step_time),
  "torque_delay_steps set incorrectly");

static_assert(observation_delay_steps == (int) round(observation_delay / inner_step_time),
  "observation_delay_steps set incorrectly");

static_assert(trq_dly_buflen == (torque_delay_steps + 1),
  "trq_dly_buflen set incorrectly");

static_assert(obs_dly_buflen == (observation_delay_steps + 1),
  "obs_dly_buflen set incorrectly");

static_assert(done_inner_steps == (int) round(time_before_reset / inner_step_time),
  "done_inner_steps set incorrectly");

/////////////////////////////////////////////////////
//////////////// Utility Functions //////////////////
/////////////////////////////////////////////////////

// high-performance copy and addition utility functions
#if act_dim == 1
  inline void cp_act(double* src, double* dst){
    dst[0] = src[0];
  }
#else
  #error "Please define cp_act"
#endif

#if obs_dim == 3
  inline void cp_obs(double* src, double* dst){
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
  }
#else
  #error "Please define cp_obs"
#endif

inline bool mj_isStable(double* state, double* actuation){
  #if check_mj_unstability == True
    return !(isnan(state[0]) || isnan(state[1]) || isnan(actuation[0]));
  #else
    return true;
  #endif
}

inline bool mj_isStable_state(double* state){
  #if check_mj_unstability == True
    return !(isnan(state[0]) || isnan(state[1]));
  #else
    return true;
  #endif
}

inline void mj_step(double* state, double* actuation){
  state[1] += (phys_coeff1 * sin(state[0]) + phys_coeff2 * actuation[0]) * physics_timestep;
  state[0] += state[1] * physics_timestep;
  if (state[1] > max_speed)
    state[1] = max_speed;
  if (state[1] < min_speed)
    state[1] = min_speed;
}

/////////////////////////////////////////////////////
////// StandupRewardGiver Class Definitions /////////
/////////////////////////////////////////////////////
StandupRewardGiver::StandupRewardGiver(){
  // nothing needed here!
}

inline void StandupRewardGiver::reset(double obs_current[obs_dim],
  bool is_mj_stable, SimInterface* simintf) {
  simif = simintf;
}

inline double f_mod(double x, double y){
  return x - y * floor(x / y);
}

inline void StandupRewardGiver::update_reward(
  double obs_current[obs_dim], // This should be current, since we'll be applying a
                               // reward delay (i.e., the same as observation delay)
  double act_current[act_dim],
  bool is_mj_stable,
  double* reward) {

  angle_norm = f_mod(simif->state[0]+PI, (2*PI)) - PI;
  *reward -= angle_norm * angle_norm;
  *reward -= 0.001 * (act_current[0] * act_current[0]);
  *reward -= 0.1 * obs_current[2] * obs_current[2];
}

inline void StandupRewardGiver::update_done(
  double obs_current[obs_dim], // This should be current, since we'll be applying a
                               // reward delay (i.e., the same as observation delay)
  double act_current[act_dim],
  bool is_mj_stable,
  bool* done) {
  *done = *done || !is_mj_stable;
}

/////////////////////////////////////////////////////
////// StandupRewardGiver Class Definitions /////////
/////////////////////////////////////////////////////
NLRRewardGiver::NLRRewardGiver(){
  // nothing needed here!
  #if FFTLIB == NOFFT
    //
  #elif FFTLIB == FFTWLIB
    nlr_fftw_plan = fftw_plan_dft_r2c_1d(nlr_buflen, theta_time_sig,
      theta_freq_sig, FFTW_ESTIMATE);
  #else
    #error "FFTLIB not implemented"
  #endif
}

inline void NLRRewardGiver::reset(double obs_current[obs_dim],
  bool is_mj_stable, SimInterface* simintf) {
  simif = simintf;
  theta_time_sig[0] = simif->state[0] - PI;
  nlrbuffidx = 1;
}

inline void NLRRewardGiver::update_reward(
  double obs_current[obs_dim], // This should be current, since we'll be applying a
                               // reward delay (i.e., the same as observation delay)
  double act_current[act_dim],
  bool is_mj_stable,
  double* reward) {
    if (nlrbuffidx <= (nlr_buflen - 1))
      theta_time_sig[nlrbuffidx] = simif->state[0] - PI;

    if (nlrbuffidx == (nlr_buflen - 1)) {
      #if FFTLIB == NOFFT
        // nothing needed!
      #elif FFTLIB == FFTWLIB
        fftw_execute(nlr_fftw_plan);
        double* theta_freq_sig_d = (double*) theta_freq_sig;

        sp2ac_sum = 0.0;
        sp2ac_bad_sum = 0.0;
        for (int k=1; k < fftout_len; k++) {
          freq_ampsq = 2 * (theta_freq_sig_d[2*k] * theta_freq_sig_d[2*k] +
            theta_freq_sig_d[2*k+1] * theta_freq_sig_d[2*k+1]);
          theta_freqabssq_sig[k] = freq_ampsq;
          sp2ac_sum += freq_ampsq;

          if ((k <= des_freq_k_low) || (k >= des_freq_k_high))
            sp2ac_bad_sum += freq_ampsq;
        }

        // good_ac_reward
        *reward += sp2ac_bad_sum * (good_ac_r_c1 / (sp2ac_sum + 0.000001));

        // dc_reward
        dc_diff = theta_freq_sig_d[0] * dc_r_c1 - dc_r_c2;
        *reward += (dc_diff < 0) ? dc_diff : -dc_diff;

        // bad_ac_reward
        ac_size_err = sqrt(sp2ac_sum) / bad_ac_r_c1 - 1.;
        if (ac_size_err > 10)
          *reward -= bad_ac_r_c2;
        else if (ac_size_err > 0)
          *reward -= ac_size_err * r_coeff;

        // Not really necessary, but for sake of unique optimality
        if (ac_size_err < -1)
          *reward -= bad_ac_r_c3;
        else if (ac_size_err < 0)
          *reward += bad_ac_r_c3 * ac_size_err;

      #else
        #error "FFTLIB not implemented"
      #endif
    }

    nlrbuffidx++;
    if (nlrbuffidx == nlr_buflen){
      theta_time_sig[0] = simif->state[0] - PI;
      nlrbuffidx = 1;
    }
}

inline void NLRRewardGiver::update_done(
  double obs_current[obs_dim], // This should be current, since we'll be applying a
                               // reward delay (i.e., the same as observation delay)
  double act_current[act_dim],
  bool is_mj_stable,
  bool* done) {
  *done = *done || (nlrbuffidx >= nlr_buflen) || !is_mj_stable;
}

NLRRewardGiver::~NLRRewardGiver() {
  #if FFTLIB == FFTWLIB
    fftw_destroy_plan(nlr_fftw_plan);
  #endif
}
/////////////////////////////////////////////////////
////////// SimInterface Class Definition ////////////
/////////////////////////////////////////////////////

// Member functions definitions including constructor
SimInterface::SimInterface(void) {
}

double* SimInterface::add_trq_buff(double new_trqs[]) {
  int i, j;
  double (*out)[act_dim];
  // Populating the whole buffer with the new items
  if (! trq_dlybuf_ever_pushed)
    for (i = 0; i < trq_dly_buflen; i++)
      for (j = 0; j < act_dim; j++)
        trq_delay_buff[i][j] = new_trqs[j];

  trq_dlybuf_ever_pushed = true;

  for (j = 0; j < act_dim; j++)
    trq_delay_buff[trq_dlybuf_push_idx][j] = new_trqs[j];

  trq_dlybuf_push_idx++;
  if (trq_dlybuf_push_idx >= trq_dly_buflen)
    trq_dlybuf_push_idx = 0;

  out = &(trq_delay_buff[trq_dlybuf_push_idx]);
  return (double*) out;
}

double* SimInterface::add_obs_buff(double new_obs[]) {
  int i, j;
  double (*out)[obs_dim];
  // Populating the whole buffer with the new items
  if (! obs_dlybuf_ever_pushed)
    for (i = 0; i < obs_dly_buflen; i++)
      for (j = 0; j < obs_dim; j++)
        obs_delay_buff[i][j] = new_obs[j];

  obs_dlybuf_ever_pushed = true;

  for (j = 0; j < obs_dim; j++)
    obs_delay_buff[obs_dlybuf_push_idx][j] = new_obs[j];

  obs_dlybuf_push_idx++;
  if (obs_dlybuf_push_idx >= obs_dly_buflen)
    obs_dlybuf_push_idx = 0;

  out = &(obs_delay_buff[obs_dlybuf_push_idx]);
  return (double*) out;
}

double SimInterface::add_rew_buff(double new_rew) {
  int i;
  // Populating the whole buffer with the new items
  if (! rew_dlybuf_ever_pushed)
    for (i = 0; i < rew_dly_buflen; i++)
        rew_delay_buff[i] = new_rew;

  rew_dlybuf_ever_pushed = true;
  rew_delay_buff[rew_dlybuf_push_idx] = new_rew;
  rew_dlybuf_push_idx++;
  if (rew_dlybuf_push_idx >= rew_dly_buflen)
    rew_dlybuf_push_idx = 0;
  return rew_delay_buff[rew_dlybuf_push_idx];
}

bool SimInterface::add_done_buff(bool new_done) {
  int i;
  // Populating the whole buffer with the new items
  if (! done_dlybuf_ever_pushed)
    for (i = 0; i < done_dly_buflen; i++)
        done_delay_buff[i] = new_done;

  done_dlybuf_ever_pushed = true;
  done_delay_buff[done_dlybuf_push_idx] = new_done;
  done_dlybuf_push_idx++;
  if (done_dlybuf_push_idx >= done_dly_buflen)
    done_dlybuf_push_idx = 0;
  return done_delay_buff[done_dlybuf_push_idx];
}

void SimInterface::update_mj_obs() {
  mj_obs[0] = cos(state[0]);
  mj_obs[1] = sin(state[0]);
  mj_obs[2] = state[1];

  #if do_obs_noise == True
    cp_obs(mj_obs, non_noisy_observation, obs_dim);
    #error "you should implement the noise injection here"
  #endif
}

double* SimInterface::reset(double* init_args_double, int* init_args_int) {
  inner_step_count = 0;
  outer_step_count = 0;

  // Resetting the observation delay buffer
  obs_dlybuf_ever_pushed = false;
  obs_dlybuf_push_idx = 0;

  // Resetting the torque delay buffer
  trq_dlybuf_ever_pushed = false;
  trq_dlybuf_push_idx = 0;

  // Resetting the reward delay buffer
  rew_dlybuf_ever_pushed = false;
  rew_dlybuf_push_idx = 0;

  // Resetting the done delay buffer
  done_dlybuf_ever_pushed = false;
  done_dlybuf_push_idx = 0;

  #if do_obs_noise == True
    #error "you should initialize the noise related variables here."
  #endif

  // Setting qpos/qvel elements
  state[0] = init_args_double[0];
  state[1] = init_args_double[1];

  is_mj_stable = mj_isStable_state(state);
  if (!is_mj_stable)
    throw std::runtime_error("SimInterface: The environment is unstable even after resetting!");

  // updating the observation
  update_mj_obs();

  // Resetting the reward object
  rew_giver.reset(mj_obs, is_mj_stable, this);

  #if defined(sparse_rew_outer_steps) && (sparse_rew_outer_steps > 1)
    sparse_rew_accumlator = 0.0;
    sparse_rew_discount = 1.0;
  #endif

  return add_obs_buff(mj_obs);
}

void SimInterface::step_inner(double action_raw[act_dim]) {
  double* joint_torque_current;
  int i;

  #ifdef Debug_step_inner
    std::cout << "  Step inner " << inner_step_count << ":" << std::endl;
  #endif

  #define obs_dlyd obs_delay_buff[obs_dlybuf_push_idx]
  #if action_type == torque
    #define joint_torque_command action_raw
  #elif action_type == jointspace_pd
    joint_torque_command[0] = -kP * (obs_dlyd[0] - action_raw[0]) - kD * obs_dlyd[2];
  #elif action_type == workspace_pd
    #error "workspace_pd needs some implementation here "
           "(translating workspace_pd to joint_torque_command)."
  #else
    #error "action_type not implemented."
  #endif

  #ifdef Debug_step_inner
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "    0) obs_dlyd             = ";
    for(i=0; i<obs_dim; i++)
      std::cout << obs_dlyd[i] << ", ";
    std::cout  << std::endl;

    std::cout << "    1) tau                  = ";
    for(i=0; i<act_dim; i++)
      std::cout << joint_torque_command[i] << ", ";
    std::cout  << std::endl;
  #endif

  // Applying actuator delay
  joint_torque_current = add_trq_buff(joint_torque_command);

  #ifdef Debug_step_inner
    std::cout << "    2) joint_torque_current = ";
    for(i=0; i<act_dim; i++)
      std::cout << joint_torque_current[i] << ", ";
    std::cout  << std::endl;
  #endif

  for(i=0; i<act_dim; i++){
    if (joint_torque_current[i] < min_torque)
      joint_torque_capped[i] = min_torque;
    else if (joint_torque_current[i] > max_torque)
      joint_torque_capped[i] = max_torque;
    else
      joint_torque_capped[i] = joint_torque_current[i];
  }

  #ifdef Debug_step_inner
    std::cout << "    2) joint_torque_capped  = ";
    for(i=0; i<act_dim; i++)
      std::cout << joint_torque_capped[i] << ", ";
    std::cout  << std::endl;
  #endif
}

void SimInterface::step(double action_raw[act_dim],
                        double** next_obs,
                        double* reward,
                        bool* done) {
  // Note: You can find the Non-delayed Observation in mj_obs

  #ifdef Debug_step_outer
    std::cout << "Step outer " << outer_step_count << ":" << std::endl;
  #endif

  *reward = 0;
  *done = false;

  // Notes: 1) When the mujoco simulation is stable,
  //           we can safely update mj_obs.
  //       2) When mujoco simulation gets unstable
  //          -> 1) No mj_step calls whatsoever!
  //          -> 2) We shouldn't call update_mj_obs!

  #if inner_loops_per_outer_loop == 1
    if (is_mj_stable) {
      // Phase 1: Preparing and Setting the Control variables
      //    Note: This does not apply mj_step and only sets
      //          the controls (i.e., only does pre-step processing)
      step_inner(action_raw);
    }

    #if ENVNAME != NLRPendulum
      // anything other than the NLR pendulum must define r = R(s,a)
      rew_giver.update_reward(mj_obs, joint_torque_capped, is_mj_stable, reward);
      *reward = add_rew_buff(*reward);
    #endif

    if (is_mj_stable) {
      // Phase 2: Calling `mj_step`
      //    Note: This call applies the actuation obtained from `step_inner`
      //          However, some `mj_data` attributes such as the foot position
      //          will not be updated, while theta and omega values get updated.
      for (int pl=0; pl < physics_loops_per_inner_loop; pl++)
        mj_step(state, joint_torque_capped);

      // Phase 4: Updating Mujoco's simulation stability Status
      is_mj_stable = mj_isStable(state, joint_torque_capped);
    }

    // Phase 5: Updating Output Variables
    //     Note: Depending on whether the simulation is stable or not,
    //           we should take different actions.
    if (is_mj_stable) {
      update_mj_obs();
    }
    *next_obs = add_obs_buff(mj_obs);

    #if ENVNAME == NLRPendulum
      // Only the NLR pendulum defines r = R(a,s')
      rew_giver.update_reward(mj_obs, joint_torque_capped, is_mj_stable, reward);
      *reward = add_rew_buff(*reward);
    #endif

    // Updating the done variable with the next observation
    rew_giver.update_done(mj_obs, joint_torque_capped, is_mj_stable, done);
    *done = add_done_buff(*done);

    inner_step_count++;          // Incrementing the inner step counter
  #else
    double single_step_rew;
    for (int i=0; i<inner_loops_per_outer_loop; i++){
      single_step_rew = 0;
      // NOTE: I left a lot of comments in the `#if inner_loops_per_outer_loop == 1` case,
      //       and I avoided repeating them here to make the code less cluttered. These two
      //       cases should be identical except for the `for` loop needed in the second case.
      if (is_mj_stable) {
        step_inner(action_raw);
      }

      #if ENVNAME != NLRPendulum
        // anything other than the NLR pendulum must define r = R(s,a)
        rew_giver.update_reward(mj_obs, joint_torque_capped, is_mj_stable, &single_step_rew);
        *reward += add_rew_buff(single_step_rew);
      #endif

      if (is_mj_stable) {
        for (int pl=0; pl < physics_loops_per_inner_loop; pl++)
          mj_step(state, joint_torque_capped);

        is_mj_stable = mj_isStable(state, joint_torque_capped);
      }
      if (is_mj_stable)
        update_mj_obs();
      *next_obs = add_obs_buff(mj_obs);

      #if ENVNAME == NLRPendulum
        // Only the NLR pendulum defines r = R(a,s')
        rew_giver.update_reward(mj_obs, joint_torque_capped, is_mj_stable, &single_step_rew);
        *reward += add_rew_buff(single_step_rew);
      #endif

      // Updating the done variable with the next observation
      rew_giver.update_done(mj_obs, joint_torque_capped, is_mj_stable, done);
      *done = add_done_buff(*done);

      inner_step_count++;
    };

    *reward /= inner_loops_per_outer_loop;

  #endif

  outer_step_count++;

  // Immediately finishing everything before we hit done_inner_steps
  *done = (*done) || (inner_step_count >=  done_inner_steps);
  //if (*done) *reward=0; // May be a good idea, but I am not sure

  #if defined(sparse_rew_outer_steps) && (sparse_rew_outer_steps > 1)
    sparse_rew_accumlator += ((*reward) * sparse_rew_discount);
    sparse_rew_discount *= sparse_rew_outer_discrate;

    if (((outer_step_count % sparse_rew_outer_steps) == 0) ||
         (*done && (inner_step_count <= done_inner_steps))
       ){
      *reward = sparse_rew_accumlator / sparse_rew_discount;
      sparse_rew_accumlator = 0.0;
      sparse_rew_discount = 1.0;
    } else
      *reward = 0;
  #endif

  #ifdef Debug_step_outer
    std::cout << "Reward: " << *reward << std::endl;
    std::cout << "Done:   " << *done << std::endl;
    std::cout << "--------------------" << std::endl;
  #endif
}

SimInterface::~SimInterface(void) {
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
  write_opt("torque", torque, &key_p, &val_p, a, b);
  write_opt("jointspace_pd", jointspace_pd, &key_p, &val_p, a, b);
  write_opt("workspace_pd", workspace_pd, &key_p, &val_p, a, b);
  write_opt("file_path_type", file_path_type, &key_p, &val_p, a, b);
  write_opt("content_type", content_type, &key_p, &val_p, a, b);
  write_opt("tanh_activation", tanh_activation, &key_p, &val_p, a, b);
  write_opt("relu_activation", relu_activation, &key_p, &val_p, a, b);
  write_opt("PI", PI, &key_p, &val_p, a, b);
  write_opt("inner_loop_rate", inner_loop_rate, &key_p, &val_p, a, b);
  write_opt("outer_loop_rate", outer_loop_rate, &key_p, &val_p, a, b);
  write_opt("physics_timestep", physics_timestep, &key_p, &val_p, a, b);
  write_opt("time_before_reset", time_before_reset, &key_p, &val_p, a, b);
  write_opt("act_dim", act_dim, &key_p, &val_p, a, b);
  write_opt("obs_dim", obs_dim, &key_p, &val_p, a, b);
  write_opt("min_torque", min_torque, &key_p, &val_p, a, b);
  write_opt("max_torque", max_torque, &key_p, &val_p, a, b);
  write_opt("min_speed", min_speed, &key_p, &val_p, a, b);
  write_opt("max_speed", max_speed, &key_p, &val_p, a, b);
  write_opt("gravity_acc", gravity_acc, &key_p, &val_p, a, b);
  write_opt("torque_delay", torque_delay, &key_p, &val_p, a, b);
  write_opt("observation_delay", observation_delay, &key_p, &val_p, a, b);
  write_opt("action_type", action_type, &key_p, &val_p, a, b);
  write_opt("kP", kP, &key_p, &val_p, a, b);
  write_opt("kD", kD, &key_p, &val_p, a, b);
  write_opt("check_mj_unstability", check_mj_unstability, &key_p, &val_p, a, b);
  write_opt("do_obs_noise", do_obs_noise, &key_p, &val_p, a, b);
  #ifdef FFTLIB
    write_opt("FFTLIB", FFTLIB, &key_p, &val_p, a, b);
  #endif
  #if ENVNAME == NLRPendulum
    write_opt("des_period", des_period, &key_p, &val_p, a, b);
    write_opt("des_offset", des_offset, &key_p, &val_p, a, b);
    write_opt("des_amp", des_amp, &key_p, &val_p, a, b);
  #endif
  #if defined(sparse_rew_outer_steps) && (sparse_rew_outer_steps > 1)
    write_opt("sparse_rew_outer_steps", sparse_rew_outer_steps, &key_p, &val_p, a, b);
    write_opt("sparse_rew_outer_discrate", sparse_rew_outer_discrate, &key_p, &val_p, a, b);
  #endif
  #if (ENVNAME == NLRPendulum) && defined(stft_inner_steps)
    write_opt("stft_inner_steps", stft_inner_steps, &key_p, &val_p, a, b);
    write_opt("stft_gamma", stft_gamma, &key_p, &val_p, a, b);
    write_opt("gamma_pow_buf", gamma_pow_buf, &key_p, &val_p, a, b);
  #endif
  write_opt("h1", h1, &key_p, &val_p, a, b);
  write_opt("h2", h2, &key_p, &val_p, a, b);
  write_opt("activation", activation, &key_p, &val_p, a, b);
  write_opt("do_mlp_output_tanh", do_mlp_output_tanh, &key_p, &val_p, a, b);
  write_opt("mlp_output_scaling", mlp_output_scaling, &key_p, &val_p, a, b);
  write_opt("phys_coeff1", phys_coeff1, &key_p, &val_p, a, b);
  write_opt("phys_coeff2", phys_coeff2, &key_p, &val_p, a, b);
  write_opt("inner_step_time", inner_step_time, &key_p, &val_p, a, b);
  write_opt("outer_step_time", outer_step_time, &key_p, &val_p, a, b);
  write_opt("inner_loops_per_outer_loop", inner_loops_per_outer_loop, &key_p, &val_p, a, b);
  write_opt("physics_loops_per_inner_loop", physics_loops_per_inner_loop, &key_p, &val_p, a, b);
  write_opt("extra_physics_loops_per_inner_loop", extra_physics_loops_per_inner_loop, &key_p, &val_p, a, b);
  write_opt("torque_delay_steps", torque_delay_steps, &key_p, &val_p, a, b);
  write_opt("observation_delay_steps", observation_delay_steps, &key_p, &val_p, a, b);
  write_opt("trq_dly_buflen", trq_dly_buflen, &key_p, &val_p, a, b);
  write_opt("obs_dly_buflen", obs_dly_buflen, &key_p, &val_p, a, b);
  write_opt("done_inner_steps", done_inner_steps, &key_p, &val_p, a, b);
  write_opt("fc1_size", fc1_size, &key_p, &val_p, a, b);
  write_opt("fc2_size", fc2_size, &key_p, &val_p, a, b);
  write_opt("fc3_size", fc3_size, &key_p, &val_p, a, b);
  write_opt("h1_div_4", h1_div_4, &key_p, &val_p, a, b);
  write_opt("h2_div_4", h2_div_4, &key_p, &val_p, a, b);
  write_opt("actdim_div_2", actdim_div_2, &key_p, &val_p, a, b);
  write_opt("n_init_args_double", n_init_args_double, &key_p, &val_p, a, b);
  write_opt("n_init_args_int", n_init_args_int, &key_p, &val_p, a, b);
  #if ENVNAME == NLRPendulum
    write_opt("r_coeff", r_coeff, &key_p, &val_p, a, b);
    write_opt("dc_r_c1", dc_r_c1, &key_p, &val_p, a, b);
    write_opt("dc_r_c2", dc_r_c2, &key_p, &val_p, a, b);
    write_opt("des_freq_low", des_freq_low, &key_p, &val_p, a, b);
    write_opt("des_freq_high", des_freq_high, &key_p, &val_p, a, b);
    write_opt("good_ac_r_c1", good_ac_r_c1, &key_p, &val_p, a, b);
    write_opt("bad_ac_r_c1", bad_ac_r_c1, &key_p, &val_p, a, b);
    write_opt("bad_ac_r_c2", bad_ac_r_c2, &key_p, &val_p, a, b);
    write_opt("bad_ac_r_c3", bad_ac_r_c3, &key_p, &val_p, a, b);
    write_opt("des_freq_k_low", des_freq_k_low, &key_p, &val_p, a, b);
    write_opt("des_freq_k_high", des_freq_k_high, &key_p, &val_p, a, b);
    write_opt("fftout_len", fftout_len, &key_p, &val_p, a, b);
  #endif
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
  }
}

#if __MAINPROG__ == SimIF_CPP

  std::chrono::system_clock::time_point tm_start;
  double gettm(void)
  {
      std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - tm_start;
      return elapsed.count();
  }

  int main(int argc, char *argv[]){
    double* init_state;
    int i;

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

    init_args_double[0] = PI - 0.2; // theta
    init_args_double[1] = 0.0; // thetadot

    init_state = simiface.reset(init_args_double, init_args_int);

    #ifdef Debug_main
      std::cout << "  -> Done Resetting the SimInterface!" << std::endl;
      for(i=0; i<obs_dim; i++)
        std::cout << "init_state[" << i << "] = " << init_state[i] << std::endl;
      std::cout << "--------------------" << std::endl;
    #endif

    #ifdef Debug_main
      std::cout << "Step 3) Stepping the SimInterface" << std::endl;
      std::cout << "  --> Started Stepping the SimInterface!" << std::endl;
    #endif

    double action_raw[act_dim] = {0.0};
    double* next_state;
    double reward;
    bool done = false;
    double sim_time = gettm();
    for (i=0; i < (outer_loop_rate*time_before_reset); i++){
      simiface.step(action_raw, &next_state, &reward, &done);
      if (done) break;
    }
    sim_time = gettm() - sim_time;
    #ifdef Debug_main
      std::cout << "  --> Done Stepping the SimInterface!" << std::endl;
      std::cout << "  --> Simulation Time: " << sim_time << std::endl;
      std::cout << "  --> Simulation Steps: " << i + ((int) done) << std::endl;
      for(i=0; i<obs_dim; i++)
        std::cout << "next_state[" << i << "] = " << next_state[i] << std::endl;
      std::cout << "--------------------" << std::endl;
    #endif

    return 0;
  }

#endif
