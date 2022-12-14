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

#if (ENVNAME == Humanoid) || (ENVNAME == HumanoidStandup)
  static_assert(inner_loops_per_outer_loop == (int) round(inner_loop_rate / outer_loop_rate),
    "inner loop rate must be an integer multiple of outer loop rate");
#else
  static_assert((inner_loop_rate % outer_loop_rate) == 0,
      "inner loop rate must be an integer multiple of outer loop rate");
#endif

static_assert(inner_loops_per_outer_loop == (inner_loop_rate / outer_loop_rate),
  "inner_loops_per_outer_loop set incorrectly");

static_assert(inner_step_time == (1.0 / inner_loop_rate),
  "inner_step_time set incorrectly");

static_assert(physics_loops_per_inner_loop == (inner_step_time / physics_timestep),
  "physics_loops_per_inner_loop set incorrectly");

static_assert(act_dim == ctrl_dim, "act_dim set incorrectly");

static_assert(n_init_args_double == (qpos_dim + qvel_dim), "n_init_args_double set incorrectly");

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

#if mjstep_order == separate_mjstep1_mjstep2
  static_assert(observation_delay_steps >= 1, \
    "You cannot logically use (mjstep_order = separate_mjstep1_mjstep2) " \
    "when observation_delay_steps is less than 1. You may want to use the " \
    "inefficient mjstep1_after_mjstep mode if you insist!");
#elif mjstep_order == delay_valid_obs
  static_assert(observation_delay_steps >= 1, \
    "You cannot logically use (mjstep_order = delay_valid_obs) when" \
    "observation_delay_steps is less than 1. You may want to use the " \
    "inefficient mjstep1_after_mjstep mode if you insist!");
#endif

/////////////////////////////////////////////////////
//////////////// Utility Functions //////////////////
/////////////////////////////////////////////////////

// high-performance copy and addition utility functions
inline void cp_arr1(double* src, double* dst){
  dst[0] = src[0];
}
inline void cp_arr2(double* src, double* dst){
  dst[0] = src[0];
  dst[1] = src[1];
}
inline void cp_arr3(double* src, double* dst){
  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];
}
inline void cp_arr4(double* src, double* dst){
  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];
  dst[3] = src[3];
}
inline void cp_arr5(double* src, double* dst){
  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];
  dst[3] = src[3];
  dst[4] = src[4];
}
inline void cp_arr6(double* src, double* dst){
  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];
  dst[3] = src[3];
  dst[4] = src[4];
  dst[5] = src[5];
}

inline void addTo_arr1(double* src, double* dst){
  dst[0] += src[0];
}
inline void addTo_arr2(double* src, double* dst){
  dst[0] += src[0];
  dst[1] += src[1];
}
inline void addTo_arr3(double* src, double* dst){
  dst[0] += src[0];
  dst[1] += src[1];
  dst[2] += src[2];
}
inline void addTo_arr4(double* src, double* dst){
  dst[0] += src[0];
  dst[1] += src[1];
  dst[2] += src[2];
  dst[3] += src[3];
}
inline void addTo_arr5(double* src, double* dst){
  dst[0] += src[0];
  dst[1] += src[1];
  dst[2] += src[2];
  dst[3] += src[3];
  dst[4] += src[4];
}
inline void addTo_arr6(double* src, double* dst){
  dst[0] += src[0];
  dst[1] += src[1];
  dst[2] += src[2];
  dst[3] += src[3];
  dst[4] += src[4];
  dst[5] += src[5];
}

#if act_dim == 1
  #define cp_act cp_arr1
#elif act_dim == 2
  #define cp_act cp_arr2
#elif act_dim == 3
  #define cp_act cp_arr3
#elif act_dim == 4
  #define cp_act cp_arr4
#elif act_dim == 5
  #define cp_act cp_arr5
#elif act_dim == 6
  #define cp_act cp_arr6
#else
  #define cp_act(src, dst) mju_copy(dst, src, act_dim)
#endif

#if obs_dim == 1
  #define cp_obs cp_arr1
#elif obs_dim == 2
  #define cp_obs cp_arr2
#elif obs_dim == 3
  #define cp_obs cp_arr3
#elif obs_dim == 4
  #define cp_obs cp_arr4
#elif obs_dim == 5
  #define cp_obs cp_arr5
#elif obs_dim == 6
  #define cp_obs cp_arr6
#else
  #define cp_obs(src, dst) mju_copy(dst, src, obs_dim)
#endif

#if qpos_dim == 1
  #define cp_qpos cp_arr1
  #define addTo_qpos addTo_arr1
#elif qpos_dim == 2
  #define cp_qpos cp_arr2
  #define addTo_qpos addTo_arr2
#elif qpos_dim == 3
  #define cp_qpos cp_arr3
  #define addTo_qpos addTo_arr3
#elif qpos_dim == 4
  #define cp_qpos cp_arr4
  #define addTo_qpos addTo_arr4
#elif qpos_dim == 5
  #define cp_qpos cp_arr5
  #define addTo_qpos addTo_arr5
#elif qpos_dim == 6
  #define cp_qpos cp_arr6
  #define addTo_qpos addTo_arr6
#else
  #define cp_qpos(src, dst) mju_copy(dst, src, qpos_dim)
  #define addTo_qpos(src, dst) mju_addTo(dst, src, qpos_dim)
#endif

#if qvel_dim == 1
  #define cp_qvel cp_arr1
  #define addTo_qvel addTo_arr1
#elif qvel_dim == 2
  #define cp_qvel cp_arr2
  #define addTo_qvel addTo_arr2
#elif qvel_dim == 3
  #define cp_qvel cp_arr3
  #define addTo_qvel addTo_arr3
#elif qvel_dim == 4
  #define cp_qvel cp_arr4
  #define addTo_qvel addTo_arr4
#elif qvel_dim == 5
  #define cp_qvel cp_arr5
  #define addTo_qvel addTo_arr5
#elif qvel_dim == 6
  #define cp_qvel cp_arr6
  #define addTo_qvel addTo_arr6
#else
  #define cp_qvel(src, dst) mju_copy(dst, src, qvel_dim)
  #define addTo_qvel(src, dst) mju_addTo(dst, src, qvel_dim)
#endif

inline bool mj_isStable(mjModel* mj_model, mjData* mj_data){
  #if check_mj_unstability == True
    int i;
    for(i=0; i<qpos_dim; i++ )
      if (mju_isBad(mj_data->qpos[i]))
        return false;
    for (i=0; i<qvel_dim; i++)
      if (mju_isBad(mj_data->qvel[i]))
        return false;
    for (i=0; i<qvel_dim; i++)
      if (mju_isBad(mj_data->qacc[i]))
        return false;
    return true;
  #else
    return true;
  #endif
}

/////////////////////////////////////////////////////
////////// RewardGiver Class Definitions ////////////
/////////////////////////////////////////////////////
RewardGiver::RewardGiver(){
  #if (ENVNAME == Ant)
    do_compute_mjids = true;
  #endif
}

#if (ENVNAME == Humanoid)
  inline double mass_center_x(mjModel* mj_model, mjData* mj_data){
    double a=0;
    double b=0;
    int i;
    for (i=0; i < body_dim; i++){
      a += mj_model->body_mass[i] * mj_data->xipos[3*i];
      b += mj_model->body_mass[i];
    }
    return a / b;
  }
#endif

#if (ENVNAME == Reacher)
  inline void fingertip_minus_target(mjData* mj_data, int fingertip_body_mjid,
    int target_body_mjid, double* out){
      out[0] = (mj_data->xipos[3*fingertip_body_mjid] -
        mj_data->xipos[3*target_body_mjid]);
      out[1] = (mj_data->xipos[3*fingertip_body_mjid+1] -
        mj_data->xipos[3*target_body_mjid+1]);
      out[2] = (mj_data->xipos[3*fingertip_body_mjid+2] -
        mj_data->xipos[3*target_body_mjid+2]);
  }

  inline double pnt3d_norm2(double point[3]){
    return sqrt((point[0] * point[0]) +
                (point[1] * point[1]) +
                (point[2] * point[2]));
  }
#endif

inline void RewardGiver::reset(double obs_current[obs_dim], mjModel* mj_model,
  mjData* mj_data, bool is_mj_stable, SimInterface* simintf) {

  simif = simintf;
  #if ((ENVNAME == HalfCheetah) || (ENVNAME == Swimmer) || \
    (ENVNAME == Hopper) || (ENVNAME == Walker2d))
    x_position_before = mj_data->qpos[0];
  #elif ((ENVNAME == InvertedPendulum) || (ENVNAME == InvertedDoublePendulum))
    // nothing is needed
  #elif (ENVNAME == Ant)
    if (do_compute_mjids){
      do_compute_mjids = false;
      torso_body_mjid = mj_name2id(mj_model, mjOBJ_BODY, "torso");
      if (torso_body_mjid < 0)
        mju_error("Body 'torso' not found");
    }
    x_position_before = mj_data->xipos[3*torso_body_mjid];
  #elif (ENVNAME == Reacher)
    vec_norm_before = pnt3d_norm2(obs_current + (obs_dim - 3));
  #elif (ENVNAME == Humanoid)
    x_position_before = mass_center_x(mj_model, mj_data);
  #elif (ENVNAME == HumanoidStandup)
    // nothing is needed for standup!
  #else
    #error "ENVNAME not implemented"
  #endif
}

#define act_current mj_data->ctrl
// // Should not be different from the previous
// #define act_current simif->joint_torque_capped

#if ((ENVNAME == InvertedPendulum) || (ENVNAME == InvertedDoublePendulum)  || \
  (ENVNAME == HalfCheetah) || (ENVNAME == Swimmer) || (ENVNAME == Reacher) || \
  (ENVNAME == HumanoidStandup))
  // no need for definign is_healthy!
#elif ENVNAME == Hopper
  inline bool is_healthy(mjData* mj_data, bool is_mj_stable){
      int i;
      if (mj_data->qpos[1] < healthy_z_range_low)
        return false;

      if ((mj_data->qpos[2] < healthy_angle_range_low) ||
          (mj_data->qpos[2] > healthy_angle_range_high))
        return false;

      for (i=2; i<qpos_dim; i++)
        if ((mj_data->qpos[i] < healthy_state_range_low) ||
            (mj_data->qpos[i] > healthy_state_range_high))
          return false;

      for (i=0; i<qvel_dim; i++)
        if ((mj_data->qvel[i] < healthy_state_range_low) ||
            (mj_data->qvel[i] > healthy_state_range_high))
          return false;

      return true;
  }
#elif ENVNAME == Walker2d
  inline bool is_healthy(mjData* mj_data, bool is_mj_stable){
    return ((mj_data->qpos[1] > healthy_z_range_low)      &&
            (mj_data->qpos[1] < healthy_z_range_high)     &&
            (mj_data->qpos[2] > healthy_angle_range_low)  &&
            (mj_data->qpos[2] < healthy_angle_range_high));
  }
#elif ENVNAME == Ant
  inline bool is_healthy(mjData* mj_data, bool is_mj_stable){
    return ((mj_data->qpos[2] > healthy_z_range_low)      &&
            (mj_data->qpos[2] < healthy_z_range_high)     &&
            is_mj_stable);
  }
#elif ENVNAME == Humanoid
  inline bool is_healthy(mjData* mj_data, bool is_mj_stable){
    return ((mj_data->qpos[2] > healthy_z_range_low)      &&
            (mj_data->qpos[2] < healthy_z_range_high));
  }
#else
  #error "ENVNAME not implemented"
#endif

inline void RewardGiver::update_reward(
  double obs_current[obs_dim], // This should be current, since we'll be applying a
                               // reward delay (i.e., the same as observation delay)
  mjModel* mj_model,
  mjData* mj_data,
  bool is_mj_stable,
  double* reward) {
  #if ENVNAME == InvertedPendulum
    *reward += 1.0;
  #elif ENVNAME == InvertedDoublePendulum
    #define cart_center_site_mjid 0
    #define cart_pos_x mj_data->site_xpos[3*cart_center_site_mjid]
    #define cart_pos_y mj_data->site_xpos[3*cart_center_site_mjid+2]
    #define v1 mj_data->qvel[1]
    #define v2 mj_data->qvel[2]

    #ifdef Debug_reward
      std::cout << std::fixed << std::setprecision(8);
      std::cout << "x=" << cart_pos_x << ", y=" << cart_pos_y << \
                   ", v1=" << v1 << ", v2=" << v2 << std::endl;
    #endif

    double cart_pos_y_neg2;
    cart_pos_y_neg2 = (cart_pos_y - 2);
    *reward -= (0.01 * cart_pos_x * cart_pos_x) + \
               (cart_pos_y_neg2 * cart_pos_y_neg2);
    *reward -= (0.001 * v1 * v1) + (0.005 * v2 * v2);
    *reward += 10;
  #elif ((ENVNAME == HalfCheetah) || (ENVNAME == Swimmer)            || \
    (ENVNAME == Hopper) || (ENVNAME == Walker2d) || (ENVNAME == Ant) || \
    (ENVNAME == Humanoid) || (ENVNAME == HumanoidStandup))

    #if (ENVNAME == Ant)
      #define x_position_after mj_data->xipos[3*torso_body_mjid]
    #elif ((ENVNAME == HalfCheetah) || (ENVNAME == Swimmer) || \
      (ENVNAME == Hopper) || (ENVNAME == Walker2d))
      #define x_position_after mj_data->qpos[0]
    #elif (ENVNAME == Humanoid)
      x_position_after = mass_center_x(mj_model, mj_data);
    #elif (ENVNAME == HumanoidStandup)
      #define x_position_after mj_data->qpos[2]
    #else
      #error "x_position_after not defined"
    #endif

    #if agg_inner_rewards == take_last
      #define r_dt outer_step_time
    #else
      #error "r_dt needs specification"
    #endif

    #if (ENVNAME == HumanoidStandup)
      #define x_velocity (x_position_after / physics_timestep)
    #else
      #define x_velocity ((x_position_after - x_position_before) / r_dt)
    #endif

    int k;

    *reward += forward_reward_weight * x_velocity;
    for (k=0; k<act_dim; k++)
      *reward -= ctrl_cost_weight * act_current[k] * act_current[k];

    #if ((ENVNAME == Hopper) || (ENVNAME == Walker2d) || (ENVNAME == Ant) || \
         (ENVNAME == Humanoid))
      #if terminate_when_unhealthy == True
        *reward += healthy_reward_weight;
      #else
        if is_healthy(mj_data, is_mj_stable)
          *reward += healthy_reward_weight;
      #endif
    #elif (ENVNAME == HumanoidStandup)
      *reward += 1;
    #endif

    #if (ENVNAME == Ant)
      #if use_contact_forces == True
        for (k = obs_cfrc_idx; k < (obs_cfrc_idx + obs_cfrc_len); k++)
          *reward -= obs_current[k] * obs_current[k] * contact_cost_weight;
      #endif
    #elif (ENVNAME == Humanoid) || (ENVNAME == HumanoidStandup)
      #if use_contact_forces == True
        contact_cost = 0;
        for (k = obs_cfrc_idx; k < (obs_cfrc_idx + obs_cfrc_len); k++)
          contact_cost += obs_current[k] * obs_current[k];
        contact_cost *= contact_cost_weight;
        #ifdef contact_cost_range_high
          if (contact_cost > contact_cost_range_high)
            contact_cost = contact_cost_range_high;
        #endif
        *reward -= contact_cost;
      #endif
    #endif

    #if ENVNAME != HumanoidStandup
      x_position_before = x_position_after; // for next round!
    #endif
  #elif (ENVNAME == Reacher)
    *reward -= vec_norm_before;
    for (int k=0; k<act_dim; k++)
      *reward -= act_current[k] * act_current[k];
    vec_norm_before = pnt3d_norm2(obs_current + (obs_dim - 3));
  #else
    #error "ENVNAME not implemented"
  #endif
}

inline void RewardGiver::update_done(
  double obs_current[obs_dim], // This should be current, since we'll be applying a
                               // reward delay (i.e., the same as observation delay)
  mjModel* mj_model,
  mjData* mj_data,
  bool is_mj_stable,
  bool* done) {
  #if ENVNAME == InvertedPendulum
    *done = *done || (obs_current[1] > 0.2) || (obs_current[1] < -0.2);
  #elif ENVNAME == InvertedDoublePendulum
    *done = *done || (cart_pos_y <= 1);
  #elif ((ENVNAME == HalfCheetah) || (ENVNAME == Swimmer)|| \
    (ENVNAME == HumanoidStandup) || (ENVNAME == Reacher))
    // nothing needs to be done here!
  #elif ((ENVNAME == Hopper) || (ENVNAME == Walker2d) || (ENVNAME == Ant) || \
    (ENVNAME == Humanoid))
    #if terminate_when_unhealthy == True
      *done = *done || !is_healthy(mj_data, is_mj_stable);
    #endif
  #else
    #error "ENVNAME not implemented"
  #endif

  *done = *done || !is_mj_stable;
}

#undef act_current

/////////////////////////////////////////////////////
////////// SimInterface Class Definition ////////////
/////////////////////////////////////////////////////

// Member functions definitions including constructor
SimInterface::SimInterface(void) {
  #if defined(stdout_pipe_file)
    freopen(stdout_pipe_file,"w",stdout);
  #endif

  // Activating Mujoco and looking for a key file
  const char* mjkey_file = std::getenv("MJKEY_PATH");
  if (mjkey_file)
    mj_activate(mjkey_file);
  else {
    char* home_envvar = std::getenv("HOME");
    char pathbuf[PATH_MAX];
    char *mjkey_abspath;

    char* mjkey_path_tmp = new char[PATH_MAX];
    mjkey_path_tmp = strcpy(mjkey_path_tmp, home_envvar);
    mjkey_path_tmp = strcat(mjkey_path_tmp, "/.mujoco/mjkey.txt");
    mjkey_abspath = realpath(mjkey_path_tmp, pathbuf);
    if (mjkey_abspath)
      mj_activate(mjkey_abspath);
    else {
      perror("~/.mujoco/mjkey.txt not found!");
      exit(EXIT_FAILURE);
    };
  }

  mj_model = NULL;                  // MuJoCo model
  mj_data = NULL;                   // MuJoCo data

  // load and compile model
  char error[100] = "Could not load binary model";
  #if xml_type == file_path_type
    mj_model = mj_loadXML(xml_file, 0, error, 100);
    if( !mj_model )
      mju_error_s("Load model error: %s", error);
  #elif xml_type == content_type
    mjVFS* mj_vfs = (mjVFS*) malloc(sizeof(mjVFS));
    int mjret_code, vfile_id, i;
    char* vfile;

    // Initializing the Virtual File System
    mj_defaultVFS(mj_vfs);

    // Creating an Empty File
    mjret_code = mj_makeEmptyFileVFS(mj_vfs, "mj_model.xml", xml_bytes);
    if (mjret_code == 1)
      mju_error("Mujoco's VFS is full.");
    if (mjret_code == 2)
      mju_error("Mujoco's VFS has an identical file name.");

    // Finding the Virtual File ID
    vfile_id = mj_findFileVFS(mj_vfs, "mj_model.xml");
    if (vfile_id < 0)
      mju_error("virtual file not found!");

    // Populating the Virtual File
    vfile = (char*) mj_vfs->filedata[vfile_id];
    for (i=0; i<(xml_bytes-5); i++)
      vfile[i] = xml_content[i];

    // Loading the mj_model from the VFS
    mj_model = mj_loadXML("mj_model.xml", mj_vfs, error, 100);
    if( !mj_model )
      mju_error_s("Load model error: %s", error);
  #else
    #error "xml loading mode not implemented."
  #endif

  // Setting the mujoco timestep option and making other assertions
  if (mj_model->opt.timestep != physics_timestep){
    mju_error("non-matching physics_timestep and mj_model->opt.timestep");
    mj_model->opt.timestep = physics_timestep;
  }

  if (mj_model->nq != qpos_dim){
    mju_error("non-matching qpos_dim and mj_model->nq");
  }

  if (mj_model->nv != qvel_dim){
    mju_error("non-matching qvel_dim and mj_model->nv");
  }

  if (mj_model->nu != ctrl_dim){
    mju_error("non-matching ctrl_dim and mj_model->nu");
  }

  #ifdef body_dim
    if (mj_model->nbody != body_dim){
      mju_error("non-matching body_dim and mj_model->nbody");
    }
  #endif

  // make data
  mj_data = mj_makeData(mj_model);

  #if (ENVNAME == Reacher)
    fingertip_body_mjid = mj_name2id(mj_model, mjOBJ_BODY, "fingertip");
    target_body_mjid = mj_name2id(mj_model, mjOBJ_BODY, "target");
    if (fingertip_body_mjid < 0)
      mju_error("Body 'fingertip' not found");
    if (target_body_mjid < 0)
      mju_error("Body 'target' not found");
  #endif
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
  #if ENVNAME == InvertedPendulum
    cp_qpos(mj_data->qpos, mj_obs);
    cp_qvel(mj_data->qvel, mj_obs+qpos_dim);
  #elif ENVNAME == InvertedDoublePendulum
      //cp_qpos(mj_data->qpos, mj_obs);
      //cp_qvel(mj_data->qvel, mj_obs+qpos_dim);
      double tmpvar;
      int i;
      mj_obs[0] = mj_data->qpos[0];  // cart x pos
      for (i=1; i < qpos_dim; i++){ // link angles
        tmpvar = mj_data->qpos[i];
        mj_obs[i] = sin(tmpvar);
        mj_obs[i + qpos_dim - 1] = cos(tmpvar);
      }
      for (i=0; i < qvel_dim; i++){
        tmpvar = mj_data->qvel[i];
        if (tmpvar > 10)
          tmpvar = 10;
        else if (tmpvar < -10)
          tmpvar = -10;
        mj_obs[i + (2*qpos_dim-1)] = tmpvar;
      }
      for (i=0; i < qvel_dim; i++){
        tmpvar = mj_data->qfrc_constraint[i];
        if (tmpvar > 10)
          tmpvar = 10;
        else if (tmpvar < -10)
          tmpvar = -10;
        mj_obs[i + (2*qpos_dim+qvel_dim-1)] = tmpvar;
      }
  #elif ENVNAME == HalfCheetah
    mju_copy(mj_obs, mj_data->qpos + 1, qpos_dim - 1); // excluding the position
    cp_qvel(mj_data->qvel, mj_obs + qpos_dim - 1);
  #elif ENVNAME == Swimmer
    mju_copy(mj_obs, mj_data->qpos + 2, qpos_dim - 2); // excluding the position
    cp_qvel(mj_data->qvel, mj_obs + qpos_dim - 2);
  #elif (ENVNAME == Hopper) || (ENVNAME == Walker2d)
    mju_copy(mj_obs, mj_data->qpos + 1, qpos_dim - 1); // excluding the position
    cp_qvel(mj_data->qvel, mj_obs + qpos_dim - 1);
    for (int k = (qpos_dim - 1); k < (qpos_dim + qvel_dim - 1); k++){
      if (mj_obs[k] < -10) mj_obs[k] = -10;
      if (mj_obs[k] > +10) mj_obs[k] = +10;
    }
  #elif ENVNAME == Ant
    mju_copy(mj_obs, mj_data->qpos + 2, qpos_dim - 2); // excluding the position
    cp_qvel(mj_data->qvel, mj_obs + qpos_dim - 2);
    #if use_contact_forces == True
      mju_copy(mj_obs + obs_cfrc_idx, mj_data->cfrc_ext, obs_cfrc_len);
      for (int k = obs_cfrc_idx; k < (obs_cfrc_len + obs_cfrc_idx); k++){
        if (mj_obs[k] < contact_force_range_low) mj_obs[k] = contact_force_range_low;
        if (mj_obs[k] > contact_force_range_high) mj_obs[k] = contact_force_range_high;
      }
    #endif
  #elif (ENVNAME == Humanoid) || (ENVNAME == HumanoidStandup)
    mju_copy(mj_obs, mj_data->qpos + 2, qpos_dim - 2); // excluding the position
    cp_qvel(mj_data->qvel, mj_obs + qpos_dim - 2);
    mju_copy(mj_obs + obs_cinert_idx, mj_data->cinert, obs_cinert_len);
    mju_copy(mj_obs + obs_cvel_idx, mj_data->cvel, obs_cvel_len);
    cp_qvel(mj_data->qfrc_actuator, mj_obs + obs_qfrcact_idx);
    #if use_contact_forces == True
      mju_copy(mj_obs + obs_cfrc_idx, mj_data->cfrc_ext, obs_cfrc_len);
    #endif
  #elif ENVNAME == Reacher
    mj_obs[0] = cos(mj_data->qpos[0]);
    mj_obs[1] = cos(mj_data->qpos[1]);
    mj_obs[2] = sin(mj_data->qpos[0]);
    mj_obs[3] = sin(mj_data->qpos[1]);
    #if qpos_dim == 4
      cp_arr2(mj_data->qpos+2, mj_obs+4);
    #else
      mju_copy(mj_obs+4, mj_data->qpos+2, qpos_dim-2);
    #endif
    mj_obs[qpos_dim+2] = mj_data->qvel[0];
    mj_obs[qpos_dim+3] = mj_data->qvel[1];
    fingertip_minus_target(mj_data, fingertip_body_mjid, target_body_mjid,
      mj_obs + qpos_dim + 4);
  #else
    #error "ENVNAME not implemented"
  #endif

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

  // Erasing all mj_data
  mj_resetData(mj_model, mj_data);

  // Setting qpos/qvel elements
  addTo_qpos(init_args_double, mj_data->qpos);
  addTo_qvel(init_args_double+qpos_dim, mj_data->qvel);

  // Calling mj_forward to validate all other mj_data variables without
  // integrating through time.
  mj_forward(mj_model, mj_data);

  is_mj_stable = mj_isStable(mj_model, mj_data);
  if (!is_mj_stable)
    mju_error("SimInterface: The environment is unstable even after resetting!");

  // updating the observation
  update_mj_obs();

  // Resetting the reward object
  rew_giver.reset(mj_obs, mj_model, mj_data, is_mj_stable, this);

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
    for(i=0; i<qpos_dim; i++)
      joint_torque_command[i] = -kP * (obs_dlyd[i] - action_raw[i]) - kD * obs_dlyd[i+qpos_dim];
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

  #if act_dim != ctrl_dim
    #error "The following capping makes no sense and can result in a segfault"
  #endif
  for(i=0; i<act_dim; i++){
    if (joint_torque_current[i] < mj_model->actuator_ctrlrange[2*i])
      joint_torque_capped[i] = mj_model->actuator_ctrlrange[2*i];
    else if (joint_torque_current[i] > mj_model->actuator_ctrlrange[2*i+1])
      joint_torque_capped[i] = mj_model->actuator_ctrlrange[2*i+1];
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
    // NOTE: Please read the `The Invalid State Problem of Mujoco`
    //       section of the `Readme.md` file to understand the issue behind
    //       `mjstep_order` and how to set it properly.
    #if (mjstep_order == mjstep1_after_mjstep) || (mjstep_order == only_mjstep)
      // Pro: The states are validated always
      // Con: Inefficiency due to calling `mj_step1` after `mj_step`
      //      (`mj_step1` is the first part of `mj_step` and gets
      //       repeated in the next iteration).
      if (is_mj_stable) {
        // Phase 1: Preparing and Setting the Control variables
        //    Note: This does not apply mj_step and only sets
        //          the controls in mj_data.control
        //          (i.e., only does pre-step processing)
        step_inner(action_raw);

        // Phase 2: Calling `mj_step`
        //    Note: This call applies the actuation obtained from `step_inner`
        //          However, some `mj_data` attributes such as the foot position
        //          will not be updated, while theta and omega values get updated.
        cp_act(joint_torque_capped, mj_data->ctrl);
        for (int pl=0; pl < physics_loops_per_inner_loop; pl++)
          mj_step(mj_model, mj_data);

        // Phase 3: Calling `mj_step1`
        //    Note: This call validates all mj_data attributes
        #if mjstep_order == mjstep1_after_mjstep
          mj_step1(mj_model, mj_data);
        #endif

        // Phase 4: Updating Mujoco's simulation stability Status
        is_mj_stable = mj_isStable(mj_model, mj_data);
      }

      // Phase 5: Updating Output Variables
      //     Note: Depending on whether the simulation is stable or not,
      //           we should take different actions.
      if (is_mj_stable) {
        update_mj_obs();
      }
      *next_obs = add_obs_buff(mj_obs);
      rew_giver.update_reward(mj_obs, mj_model, mj_data, is_mj_stable, reward);
      *reward = add_rew_buff(*reward);
    #elif mjstep_order == separate_mjstep1_mjstep2
      // Pro: The states are validated always
      // Con: Mujoco's documentation advices against separately calling `mj_step1`
      //      and `mj_step2` for the RK4 integrator for some unknown reason. (Look for
      //      "Keep in mind though that the RK4 solver does not work with mj_step1/2."
      //      at http://www.mujoco.org/book/reference.html)
      if (is_mj_stable) {
        // Phase 1: Preparing and Setting the Control variables
        //    Note: This does not apply mj_step and only sets
        //          the controls in mj_data.control
        //          (i.e., only does pre-step processing)
        step_inner(action_raw);

        // Phase 2: Calling `mj_step1`
        //    Note: This call only validates all the attributes in mj_data
        //          i.e., `mj_step1` only does pre-actutation processing
        //          and updating states.
        cp_act(joint_torque_capped, mj_data->ctrl);
        for (int pl=0; pl < extra_physics_loops_per_inner_loop; pl++)
          mj_step(mj_model, mj_data);
        mj_step1(mj_model, mj_data);

        is_mj_stable = mj_isStable(mj_model, mj_data);
      }
      // Phase 3: Updating the reward
      //    Note: Now that we have all the states validated and synchoronized,
      //          we can call `update_reward`.
      // Comment: You should be able to call `update_mj_obs()`without having
      //          to worry about state validation issues. If this doesn't seem
      //          to be the case, this is possibly indicitave of a bug.
      rew_giver.update_reward(mj_obs, mj_model, mj_data, is_mj_stable, reward);
      *reward = add_rew_buff(*reward);

      // Phase 4: Applying external forces and controls
      if (is_mj_stable) {
        mj_step2(mj_model, mj_data);
        is_mj_stable = mj_isStable(mj_model, mj_data);
      }

      // Phase 5: Applying the observation delay and updating mj_obs
      if (is_mj_stable)
        update_mj_obs();
      *next_obs = add_obs_buff(mj_obs);
    #elif mjstep_order == delay_valid_obs
      // Pro: This is a safe and efficient call
      // Con: Since we will not validate all mj_data attributes, some attributes
      //      may be outdated or out-of-sync with other. For example, qpos and qvels
      //      values (e.g., theta and omega) seem to be updated after mj_step, while
      //      foot_x and foot_z values are not updated (i.e., mj_data is not in a
      //      valid state). You should be very careful about how to extract different
      //      observations.
      //
      // Note: We'll preserve syncing between physical variables by delaying theta and
      //       omega through storing them in the mj_obs variable of the sim interface.
      //       The rest of the non-updated attributes will be extracted from mj_data.
      //       Since we're delaying the input observations to the reward function, the
      //       reward delay is set to be one step smaller than the observation delay.
      if (is_mj_stable) {
        // Phase 1: Preparing and Setting the Control variables
        //    Note: This does not apply mj_step and only sets
        //          the controls in mj_data.control
        //          (i.e., only does pre-step processing)
        step_inner(action_raw);

        // Phase 2: Calling `mj_step`
        //    Note: This call applies the actuation obtained from `step_inner`
        //          However, some `mj_data` attributes such as the foot position
        //          will not be updated, while theta and omega values get updated.
        cp_act(joint_torque_capped, mj_data->ctrl);
        for (int pl=0; pl < physics_loops_per_inner_loop; pl++)
          mj_step(mj_model, mj_data);
        is_mj_stable = mj_isStable(mj_model, mj_data);
      }

      // Phase 3: Updating the reward
      //    Note: Now that we have all the states validated and synchoronized,
      //          we can call `update_reward`.
      // Comment: Do not call `update_mj_obs()`. update_mj_obs will update
      //          mj_obs, which will make the physical variables out of sync.

      rew_giver.update_reward(mj_obs, mj_model, mj_data, is_mj_stable, reward);
      *reward = add_rew_buff(*reward);

      if (is_mj_stable)
        update_mj_obs();

      *next_obs = add_obs_buff(mj_obs);
    #else
      #error "mjstep_order not implemented"
    #endif
    // Updating the done variable with the next observation
    rew_giver.update_done(mj_obs, mj_model, mj_data, is_mj_stable, done);
    *done = add_done_buff(*done);

    inner_step_count++;          // Incrementing the inner step counter
  #else
    double single_step_rew;
    for (int i=0; i<inner_loops_per_outer_loop; i++){
      single_step_rew = 0;
      // NOTE: I left a lot of comments in the `#if inner_loops_per_outer_loop == 1` case,
      //       and I avoided repeating them here to make the code less cluttered. These two
      //       cases should be identical except for the `for` loop needed in the second case.
      #if (mjstep_order == mjstep1_after_mjstep) || (mjstep_order == only_mjstep)
        if (is_mj_stable) {
          step_inner(action_raw);

          cp_act(joint_torque_capped, mj_data->ctrl);
          for (int pl=0; pl < physics_loops_per_inner_loop; pl++)
            mj_step(mj_model, mj_data);

          #if mjstep_order == mjstep1_after_mjstep
            mj_step1(mj_model, mj_data);
          #endif

          //mj_step1(mj_model, mj_data);
          is_mj_stable = mj_isStable(mj_model, mj_data);
        }
        if (is_mj_stable)
          update_mj_obs();
        *next_obs = add_obs_buff(mj_obs);

        // accumulating the inner rewards
        #if agg_inner_rewards == take_average
          rew_giver.update_reward(mj_obs, mj_model, mj_data, is_mj_stable, &single_step_rew);
          *reward += add_rew_buff(single_step_rew);
        #elif agg_inner_rewards == take_last
          if (i == inner_loops_per_outer_loop - 1){
            rew_giver.update_reward(mj_obs, mj_model, mj_data, is_mj_stable, &single_step_rew);
            *reward += add_rew_buff(single_step_rew);
          }
          else
            add_rew_buff(single_step_rew);
        #else
          #error "agg_inner_rewards not implemented"
        #endif

      #elif mjstep_order == separate_mjstep1_mjstep2
        if (is_mj_stable) {
          step_inner(action_raw);
          cp_act(joint_torque_capped, mj_data->ctrl);
          for (int pl=0; pl < extra_physics_loops_per_inner_loop; pl++)
            mj_step(mj_model, mj_data);
          mj_step1(mj_model, mj_data);
          is_mj_stable = mj_isStable(mj_model, mj_data);
        }

        // accumulating the inner rewards
        #if agg_inner_rewards == take_average
          rew_giver.update_reward(mj_obs, mj_model, mj_data, is_mj_stable, &single_step_rew);
          *reward += add_rew_buff(single_step_rew);
        #elif agg_inner_rewards == take_last
          if (i == inner_loops_per_outer_loop - 1){
            rew_giver.update_reward(mj_obs, mj_model, mj_data, is_mj_stable, &single_step_rew);
            *reward += add_rew_buff(single_step_rew);
          }
          else
            add_rew_buff(single_step_rew);
        #else
          #error "agg_inner_rewards not implemented"
        #endif

        if (is_mj_stable) {
          mj_step2(mj_model, mj_data);
          is_mj_stable = mj_isStable(mj_model, mj_data);
        }

        if (is_mj_stable)
          update_mj_obs();

        *next_obs = add_obs_buff(mj_obs);
      #elif mjstep_order == delay_valid_obs
        if (is_mj_stable) {
          step_inner(action_raw);
          cp_act(joint_torque_capped, mj_data->ctrl);
          for (int pl=0; pl < physics_loops_per_inner_loop; pl++)
            mj_step(mj_model, mj_data);
          is_mj_stable = mj_isStable(mj_model, mj_data);
        }

        // accumulating the inner rewards
        #if agg_inner_rewards == take_average
          rew_giver.update_reward(mj_obs, mj_model, mj_data, is_mj_stable, &single_step_rew);
          *reward += add_rew_buff(single_step_rew);
        #elif agg_inner_rewards == take_last
          if (i == inner_loops_per_outer_loop - 1){
            rew_giver.update_reward(mj_obs, mj_model, mj_data, is_mj_stable, &single_step_rew);
            *reward += add_rew_buff(single_step_rew);
          }
          else
            add_rew_buff(single_step_rew);
        #else
          #error "agg_inner_rewards not implemented"
        #endif

        if (is_mj_stable)
          update_mj_obs();

        *next_obs = add_obs_buff(mj_obs);
      #else
        #error "mjstep_order not implemented"
      #endif

      // Updating the done variable with the next observation
      rew_giver.update_done(mj_obs, mj_model, mj_data, is_mj_stable, done);
      *done = add_done_buff(*done);

      inner_step_count++;
    };

    #if agg_inner_rewards == take_average
      *reward /= inner_loops_per_outer_loop;
    #elif agg_inner_rewards == take_last
    #else
      #error "agg_inner_rewards not implemented"
    #endif

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
  // free MuJoCo model and data, deactivate
  mj_deleteData(mj_data);
  mj_deleteModel(mj_model);
  mj_deactivate();
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
  write_opt("take_average", take_average, &key_p, &val_p, a, b);
  write_opt("take_last", take_last, &key_p, &val_p, a, b);
  write_opt("PI", PI, &key_p, &val_p, a, b);
  write_opt("inner_loop_rate", inner_loop_rate, &key_p, &val_p, a, b);
  write_opt("outer_loop_rate", outer_loop_rate, &key_p, &val_p, a, b);
  write_opt("physics_timestep", physics_timestep, &key_p, &val_p, a, b);
  write_opt("time_before_reset", time_before_reset, &key_p, &val_p, a, b);
  write_opt("qpos_dim", qpos_dim, &key_p, &val_p, a, b);
  write_opt("qvel_dim", qvel_dim, &key_p, &val_p, a, b);
  write_opt("ctrl_dim", ctrl_dim, &key_p, &val_p, a, b);
  write_opt("torque_delay", torque_delay, &key_p, &val_p, a, b);
  write_opt("observation_delay", observation_delay, &key_p, &val_p, a, b);
  write_opt("action_type", action_type, &key_p, &val_p, a, b);
  write_opt("kP", kP, &key_p, &val_p, a, b);
  write_opt("kD", kD, &key_p, &val_p, a, b);
  write_opt("check_mj_unstability", check_mj_unstability, &key_p, &val_p, a, b);
  write_opt("agg_inner_rewards", agg_inner_rewards, &key_p, &val_p, a, b);
  write_opt("do_obs_noise", do_obs_noise, &key_p, &val_p, a, b);
  #ifdef body_dim
    write_opt("body_dim", body_dim, &key_p, &val_p, a, b);
  #endif
  #ifdef use_contact_forces
    write_opt("use_contact_forces", use_contact_forces, &key_p, &val_p, a, b);
  #endif
  #ifdef forward_reward_weight
    write_opt("forward_reward_weight", forward_reward_weight, &key_p, &val_p, a, b);
  #endif
  #ifdef ctrl_cost_weight
    write_opt("ctrl_cost_weight", ctrl_cost_weight, &key_p, &val_p, a, b);
  #endif
  #ifdef healthy_reward_weight
    write_opt("healthy_reward_weight", healthy_reward_weight, &key_p, &val_p, a, b);
  #endif
  #ifdef terminate_when_unhealthy
    write_opt("terminate_when_unhealthy", terminate_when_unhealthy, &key_p, &val_p, a, b);
  #endif
  #ifdef healthy_state_range_low
    write_opt("healthy_state_range_low", healthy_state_range_low, &key_p, &val_p, a, b);
  #endif
  #ifdef healthy_state_range_high
    write_opt("healthy_state_range_high", healthy_state_range_high, &key_p, &val_p, a, b);
  #endif
  #ifdef healthy_z_range_low
    write_opt("healthy_z_range_low", healthy_z_range_low, &key_p, &val_p, a, b);
  #endif
  #ifdef healthy_z_range_high
    write_opt("healthy_z_range_high", healthy_z_range_high, &key_p, &val_p, a, b);
  #endif
  #ifdef healthy_angle_range_low
    write_opt("healthy_angle_range_low", healthy_angle_range_low, &key_p, &val_p, a, b);
  #endif
  #ifdef healthy_angle_range_high
    write_opt("healthy_angle_range_high", healthy_angle_range_high, &key_p, &val_p, a, b);
  #endif
  #ifdef contact_cost_weight
    write_opt("contact_cost_weight", contact_cost_weight, &key_p, &val_p, a, b);
  #endif
  #ifdef contact_force_range_low
    write_opt("contact_force_range_low", contact_force_range_low, &key_p, &val_p, a, b);
  #endif
  #ifdef contact_force_range_high
    write_opt("contact_force_range_high", contact_force_range_high, &key_p, &val_p, a, b);
  #endif
  #ifdef obs_cfrc_idx
    write_opt("obs_cfrc_idx", obs_cfrc_idx, &key_p, &val_p, a, b);
  #endif
  #ifdef obs_cfrc_len
    write_opt("obs_cfrc_len", obs_cfrc_len, &key_p, &val_p, a, b);
  #endif
  #if ENVNAME == Humanoid
    write_opt("obs_cinert_len", obs_cinert_len, &key_p, &val_p, a, b);
    write_opt("obs_cinert_idx", obs_cinert_idx, &key_p, &val_p, a, b);
    write_opt("obs_cvel_len", obs_cvel_len, &key_p, &val_p, a, b);
    write_opt("obs_cvel_idx", obs_cvel_idx, &key_p, &val_p, a, b);
    write_opt("obs_qfrcact_len", obs_qfrcact_len, &key_p, &val_p, a, b);
    write_opt("obs_qfrcact_idx", obs_qfrcact_idx, &key_p, &val_p, a, b);
  #endif
  #if defined(sparse_rew_outer_steps) && (sparse_rew_outer_steps > 1)
    write_opt("sparse_rew_outer_steps", sparse_rew_outer_steps, &key_p, &val_p, a, b);
    write_opt("sparse_rew_outer_discrate", sparse_rew_outer_discrate, &key_p, &val_p, a, b);
  #endif
  write_opt("xml_type", xml_type, &key_p, &val_p, a, b);
  write_opt("h1", h1, &key_p, &val_p, a, b);
  write_opt("h2", h2, &key_p, &val_p, a, b);
  write_opt("activation", activation, &key_p, &val_p, a, b);
  write_opt("do_mlp_output_tanh", do_mlp_output_tanh, &key_p, &val_p, a, b);
  write_opt("mlp_output_scaling", mlp_output_scaling, &key_p, &val_p, a, b);
  write_opt("obs_dim", obs_dim, &key_p, &val_p, a, b);
  write_opt("act_dim", act_dim, &key_p, &val_p, a, b);
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
  write_opt("mjstep1_after_mjstep", mjstep1_after_mjstep, &key_p, &val_p, a, b);
  write_opt("separate_mjstep1_mjstep2", separate_mjstep1_mjstep2, &key_p, &val_p, a, b);
  write_opt("delay_valid_obs", delay_valid_obs, &key_p, &val_p, a, b);
  write_opt("only_mjstep", only_mjstep, &key_p, &val_p, a, b);
  write_opt("mjstep_order", mjstep_order, &key_p, &val_p, a, b);
  write_opt("rew_dly_buflen", rew_dly_buflen, &key_p, &val_p, a, b);
  write_opt("done_dly_buflen", done_dly_buflen, &key_p, &val_p, a, b);
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
    #if ENVNAME == InvertedPendulum
      init_args_double[0] = 0.01; // pos_slider
      init_args_double[1] = 0.01; // theta_pend
      init_args_double[2] = 0.01; // vel_slider
      init_args_double[3] = 0.01; // omega_pend
    #else
      for (i=0; i < (qpos_dim+qvel_dim); i++)
        init_args_double[i] = 0.0;
    #endif



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
