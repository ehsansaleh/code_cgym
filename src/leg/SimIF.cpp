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

static_assert((inner_loop_rate % outer_loop_rate) == 0,
    "inner loop rate must be an integer multiple of outer loop rate ");

static_assert(inner_loops_per_outer_loop == (inner_loop_rate / outer_loop_rate),
  "inner_loops_per_outer_loop set incorrectly");

static_assert(inner_step_time == (1.0 / inner_loop_rate),
  "inner_step_time set incorrectly");

static_assert(physics_loops_per_inner_loop == (inner_step_time / physics_timestep),
  "physics_loops_per_inner_loop set incorrectly");

static_assert(obs_dim == (4 * obs_history_len_int + contact_obs_int + jumping_obs_int + extra_obs_dim_int),
  "obs_dim set incorrectly");

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

static_assert(neg_omega_hip_maxabs == -omega_hip_maxabs,
  "neg_omega_hip_maxabs set incorrectly");

static_assert(neg_omega_knee_maxabs == -omega_knee_maxabs,
  "neg_omega_knee_maxabs set incorrectly");

static_assert(neg_fcn_k50_co_0 == (fcn_k50_co_0 * -1),
  "neg_fcn_k50_co_0 set incorrectly");

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

void solve_sec_ode(double a, double b, double c,
                   double y_init, double ydot_init,
                   double t_f, double* y_tf, double* ydot_tf){

  // This function solves the homogeneous ODE with constant coefficients
  // y_ddot + a * y_dot + b * y - b * c = 0
  // The exact y function is computed and evaluated at the final time t_f.
  // y_tf will be y(t_f).
  // ydot_tf will be ydot(t_f).


  double x_init = y_init - c;
  double delta = (a*a) - 4*b;
  double delta_sqrt, r1, r2, c1, c2, e1, e2, sin_, cos_;

  if (delta > 0) {
    // Two real roots case
    delta_sqrt = sqrt(delta);
    r1 = (- a - delta_sqrt) / 2;
    r2 = (- a + delta_sqrt) / 2;
    c1 = (x_init * r2 - ydot_init ) / delta_sqrt;
    c2 = (ydot_init - - x_init * r1) / delta_sqrt;
    e1 = exp(r1 * t_f);
    e2 = exp(r2 * t_f);
    *y_tf = c1 * e1 + c2 * e2 + c;
    *ydot_tf = c1 * r1 * e1 + c2 * r2 * e2;
  } else if (delta == 0){
    // One real root case
    r1 = -a / 2;
    c1 = x_init;
    c2 = ydot_init - c1 * r1;
    e1 = exp(r1 * t_f);
    *y_tf = e1 * (c1 + c2 * t_f) + c;
    *ydot_tf = e1 * (c1 * r1 + c2 * (1 + r1 * t_f));
  } else {
    // Two imaginary roots case
    r1 = -a / 2;
    r2 = sqrt(-delta) / 2;
    c1 = x_init;
    c2 = (ydot_init - c1 * r1) / r2;
    e1 = exp(r1 * t_f);
    cos_ = cos(r2 * t_f);
    sin_ = sin(r2 * t_f);
    *y_tf = e1 * (c1 * cos_ + c2 * sin_) + c;
    *ydot_tf = e1 * (c1 * (r1 * cos_ - r2 * sin_) + c2 * (r1 * sin_ + r2 *cos_));
  }
}

void solve_motor_ode(double tau_cmd, double y_init, double ydot_init,
                     double t_f, double* y_tf, double* ydot_tf){

  // This function solves the homogeneous ODE with constant coefficients
  // y_ddot + 2 * zeta * omega0 * y_dot + (omega0 ** 2) * (y - tau_cmd) = 0
  // The exact y function is computed and evaluated at the final time t_f.
  // y_tf will be y(t_f).
  // ydot_tf will be ydot(t_f).

  double c1, c2, g, exp_, sin_, cos_;
  // Two imaginary roots case
  c1 = y_init - tau_cmd;
  c2 = (ydot_init - c1 * motor_root_real) / motor_root_imag;
  exp_ = exp(motor_root_real * t_f);
  g = motor_root_imag * t_f;
  cos_ = cos(g);
  sin_ = sin(g);
  *y_tf = exp_ * (c1 * cos_ + c2 * sin_) + tau_cmd;
  *ydot_tf = exp_ * (c1 * (motor_root_real * cos_ - motor_root_imag * sin_) +
                     c2 * (motor_root_real * sin_ + motor_root_imag * cos_));
}

double inline dyno_motor_model(double abs_jnt_omega, double jnt_torque) {
  double rpm;
  double cmd_units;
  double max_cmd_at_rpm;
  double joint_torque_capped;
  rpm = rpm_per_omega * abs_jnt_omega;
  if (jnt_torque > 0) {
    cmd_units = jnt_torque * fcn_k50_co_0 + fcn_k50_co_1;
    cmd_units = jnt_torque * cmd_units + fcn_k50_co_2;
    cmd_units = jnt_torque * cmd_units;
  } else {
    cmd_units = jnt_torque * neg_fcn_k50_co_0 + fcn_k50_co_1;
    cmd_units = jnt_torque * cmd_units - fcn_k50_co_2;
    cmd_units = jnt_torque * cmd_units;
  }

  max_cmd_at_rpm = rpm - rpm_for_max_cmd_dyno;
  max_cmd_at_rpm = cmd_per_rpm_slope_dyno * max_cmd_at_rpm;
  if (max_cmd_at_rpm > 0)
    max_cmd_at_rpm = 0;
  max_cmd_at_rpm += max_cmd_dyno;
  if (max_cmd_at_rpm < 0)
    max_cmd_at_rpm = 0;
  if (cmd_units > max_cmd_at_rpm)
    cmd_units = max_cmd_at_rpm;

  joint_torque_capped  = model_coeff_dyno_0 + (model_coeff_dyno_1 * cmd_units);
  joint_torque_capped += model_coeff_dyno_2 * rpm;
  joint_torque_capped += model_coeff_dyno_3 * cmd_units * cmd_units;
  joint_torque_capped += model_coeff_dyno_4 * cmd_units * rpm;
  joint_torque_capped += model_coeff_dyno_5 * rpm * rpm;
  if (jnt_torque < 0)
    joint_torque_capped *= -1;
  return joint_torque_capped;
}

double inline naive_motor_model(double abs_jnt_omega, double jnt_torque) {
  double max_tau_avail;
  double abs_jnt_torque;
  double c1, c2;

  abs_jnt_torque = (jnt_torque > 0) ? jnt_torque : (-jnt_torque);
  if (abs_jnt_omega < 30.573932974138017){
    c1 = (-1.37525388e-02);
    c2 = (9.81094800e+00);
  } else if (abs_jnt_omega < 33.39821124007745){
    c1 = (-2.48127817e-01);
    c2 = (1.69767220e+01);
  } else if (abs_jnt_omega < 33.84650937752816){
    c1 = (-1.93837450e+01);
    c2 = (6.56072108e+02);
  } else {
    c1 = 0.0;
    c2 = 0.0;
  }

  max_tau_avail = c1 * abs_jnt_omega + c2;
  if (max_tau_avail > abs_jnt_torque)
    max_tau_avail = abs_jnt_torque;
  if (jnt_torque < 0)
    max_tau_avail *= (-1);
  return max_tau_avail;
}

inline void _get_contact_state(mjModel* mj_model, mjData* mj_data,
  int ground_geom_id, int lowerleg_limb_geom_id,
  bool* foot_in_contact, double foot_force[6]){
    // Note: foot_force only needs to have 3 members. However,
    // since we want to be inline, and mj_contactForce requires 6 elements
    // We ask the user to provide 6 elements. Sorry for the inconvinience!

    // Zeroing Out the foot forces in case no contact existed.
    // We don't need the y component, so I disabled it here.
    foot_force[0] = 0;
    foot_force[2] = 0;

    for (int i=0; i < mj_data->ncon; i++)
      if ((mj_data->contact[i].geom1 == ground_geom_id) &&
          (mj_data->contact[i].geom2 == lowerleg_limb_geom_id)) {
        *foot_in_contact = true;
        mj_contactForce(mj_model, mj_data, i, foot_force);
        #define contact_frame mj_data->contact[i].frame
        // A lazy implementation of matrix vector products:
        // We don't need the y component, so I disabled it here.
        // Also, foot_force[3:6] are useless for us, so we scrap them!
        foot_force[3] = foot_force[0];
        foot_force[0] = foot_force[0] * contact_frame[0] + \
                        foot_force[1] * contact_frame[3] + \
                        foot_force[2] * contact_frame[6];
        foot_force[2] = foot_force[3] * contact_frame[2] + \
                        foot_force[1] * contact_frame[5] + \
                        foot_force[2] * contact_frame[8];
      };

  }

inline bool mj_isStable(mjModel* mj_model, mjData* mj_data){
  #if check_mj_unstability == True
    int i;
    for(i=0; i<mj_model->nq; i++ )
      if (mju_isBad(mj_data->qpos[i]))
        return false;
    for (i=0; i<mj_model->nv; i++)
      if (mju_isBad(mj_data->qvel[i]))
        return false;
    for (i=0; i<mj_model->nv; i++)
      if (mju_isBad(mj_data->qacc[i]))
        return false;
    return true;
  #else
    return true;
  #endif
}

#if add_act_sm_r == True
  inline double abs_diff(double a, double b){
      return (a > b) ? (a - b) : (b - a);
  }

  double roughness(double* arr, int st_idx, int end_idx, int depth_lvl){
    if ((end_idx - st_idx) < 5)
      return 0;

    int arg_min, arg_max;
    int i;
    int sec_idx, thrd_idx;
    double min_val, max_val, i_val, ip1_val;
    double out = 0;
    arg_min = st_idx;
    min_val = arr[arg_min];
    arg_max = st_idx;
    max_val = arr[arg_max];
    for (i = st_idx; i < end_idx; i++){
      i_val = arr[i];
      if (i_val > max_val) {
        arg_max = i;
        max_val = i_val;
      };
      if (i_val < min_val) {
        arg_min = i;
        min_val = i_val;
      };
    };

    if (arg_min < arg_max){
      sec_idx = arg_min;
      thrd_idx = arg_max;
    } else {
      sec_idx = arg_max;
      thrd_idx = arg_min;
    };


    if (depth_lvl > 0){
      // Make recursive calls
      out = 0;
      out += roughness(arr, st_idx, sec_idx, depth_lvl-1);
      out += roughness(arr, sec_idx, thrd_idx, depth_lvl-1);
      out += roughness(arr, thrd_idx, end_idx, depth_lvl-1);
      return out;
    }
    else {
      // final level of recursiveness
      double diff_abs_sum;
      double frs_val, sec_val, thr_val, frt_val;

      diff_abs_sum = 0;
      for (i = st_idx; i < (end_idx-1); i++){
        i_val = arr[i];
        ip1_val = arr[i+1];
        diff_abs_sum += (i_val > ip1_val) ? (i_val - ip1_val) : (ip1_val - i_val);
      };

      frs_val = arr[st_idx];
      sec_val = arr[sec_idx];
      thr_val = arr[thrd_idx];
      frt_val = arr[end_idx-1];

      out = 0;
      out += abs_diff(frs_val, sec_val);
      out += abs_diff(sec_val, thr_val);
      out += abs_diff(thr_val, frt_val);
      out = out - diff_abs_sum;
      return out;
    }
  }
#endif
/////////////////////////////////////////////////////
////////// RewardGiver Class Definitions ////////////
/////////////////////////////////////////////////////

SimplifiedRewardGiver::SimplifiedRewardGiver(){
  do_compute_mjids = true;
}

inline void SimplifiedRewardGiver::reset(mjModel* mj_model,
                                         mjData* mj_data,
                                         SimInterface* simintf) {
  has_touched_ground = false;
  simif = simintf;
  if (do_compute_mjids){
    do_compute_mjids = false;
    foot_center_site_mjid = mj_name2id(mj_model, mjOBJ_SITE, "foot-center");
    hip_center_site_mjid = mj_name2id(mj_model, mjOBJ_SITE, "hip-center");
    knee_center_site_mjid = mj_name2id(mj_model, mjOBJ_SITE, "knee-center");
    ground_geom_id = mj_name2id(mj_model, mjOBJ_GEOM, "ground");
    lowerleg_limb_geom_id = mj_name2id(mj_model, mjOBJ_GEOM, "lowerleg-limb");
    if (foot_center_site_mjid < 0)
      mju_error("Site 'foot-center' not found");
    if (hip_center_site_mjid < 0)
      mju_error("Site 'hip-center' not found");
    if (knee_center_site_mjid < 0)
      mju_error("Site 'knee-center' not found");
    if (ground_geom_id < 0)
      mju_error("Geom 'ground' not found");
    if (lowerleg_limb_geom_id < 0)
      mju_error("Geom 'lowerleg-limb' not found");

    #if add_act_sm_r == True
      hip_ctrl_id = mj_name2id(mj_model, mjOBJ_ACTUATOR, "torquehip");
      knee_ctrl_id = mj_name2id(mj_model, mjOBJ_ACTUATOR, "torqueknee");
      if (hip_ctrl_id < 0)
        mju_error("Control 'torquehip' not found");
      if (knee_ctrl_id < 0)
        mju_error("Control 'torqueknee' not found");
    #endif
  };

  #if add_act_sm_r == True
    mju_zero(traj_tau_hip, done_inner_steps);
    mju_zero(traj_tau_knee, done_inner_steps);
  #endif

  #ifdef Debug_reward
    slider_mjid = mj_name2id(mj_model, mjOBJ_JOINT, "slider");
    hip_mjid = mj_name2id(mj_model, mjOBJ_JOINT, "hip");
    knee_mjid = mj_name2id(mj_model, mjOBJ_JOINT, "knee");
    if (hip_mjid < 0)
      mju_error("Joint 'hip' not found");
    if (knee_mjid < 0)
      mju_error("Joint 'knee' not found");
    if (slider_mjid < 0)
      mju_error("Joint 'slider' not found");
    no_update_calls = 0;
  #endif
};

inline void SimplifiedRewardGiver::update_reward(
  double obs_current[obs_dim], // This should be current, since we'll be applying a
                               // reward delay (i.e., the same as observation delay)
  mjModel* mj_model,
  mjData* mj_data,
  double* reward) {

  #define theta_hip_oc obs_current[0]
  #define theta_knee_oc obs_current[1]
  #define omega_hip_oc obs_current[2]
  #define omega_knee_oc obs_current[3]

  // Note: Do not uncomment the following lines unless you know what you're doing!
  //       This may cause the physical variables like foot_pos_x and theta_hip
  //       to be out of sync
  // #define theta_hip_oc mj_data->qpos[mj_model->jnt_qposadr[hip_mjid]]
  // #define theta_knee_oc mj_data->qpos[mj_model->jnt_qposadr[knee_mjid]]
  // #define omega_hip_oc mj_data->qvel[mj_model->jnt_dofadr[hip_mjid]]
  // #define omega_knee_oc mj_data->qvel[mj_model->jnt_dofadr[knee_mjid]]

  #define foot_pos_x mj_data->site_xpos[3*foot_center_site_mjid]
  #define hip_pos_x mj_data->site_xpos[3*hip_center_site_mjid]
  #define knee_pos_x mj_data->site_xpos[3*knee_center_site_mjid]
  #define foot_pos_z mj_data->site_xpos[3*foot_center_site_mjid+2]
  #define hip_pos_z mj_data->site_xpos[3*hip_center_site_mjid+2]
  #define knee_pos_z mj_data->site_xpos[3*knee_center_site_mjid+2]
  #define foot_force_x foot_force[0]
  #define foot_force_z foot_force[2]
  #define PIhalf 1.57079632679489661923
  #define PI4th 0.785398163397448309615
  #define NegPIhalf -1.57079632679489661923
  #define NegPI4th -0.785398163397448309615

  // Updating foot_force and has_touched_ground
  _get_contact_state(mj_model, mj_data,
                     ground_geom_id, lowerleg_limb_geom_id,
                     &has_touched_ground, foot_force);

  // The main term
  *reward += (theta_hip_oc > NegPI4th) ? (NegPI4th - theta_hip_oc) : (PI4th + theta_hip_oc);
  *reward += (theta_knee_oc > NegPIhalf) ? (NegPIhalf - theta_knee_oc) : (PIhalf + theta_knee_oc);

  #ifdef Debug_reward
    std::cout << std::fixed << std::setprecision(8);
    double reward_total = 0;
    std::cout << "  RStep " << no_update_calls+rew_dly_buflen-1 << ":" << std::endl;
    std::cout << "    -->    theta            = " << mj_data->qpos[mj_model->jnt_qposadr[hip_mjid]] \
                                          << ", " << mj_data->qpos[mj_model->jnt_qposadr[knee_mjid]] \
                                           << std::endl;
    std::cout << "    -->    omega            = " << mj_data->qvel[mj_model->jnt_dofadr[hip_mjid]] \
                                          << ", " << mj_data->qvel[mj_model->jnt_dofadr[knee_mjid]] \
                                           << std::endl;
    std::cout << "    -->    theta_input      = " << obs_current[0] << ", " << obs_current[1] << std::endl;
    std::cout << "    -->    omega_input      = " << obs_current[2] << ", " << obs_current[3] << std::endl;
    std::cout << "    -->    foot_pos_x,z     = " << foot_pos_x << ", " << foot_pos_z << std::endl;
    std::cout << "    -->    knee_pos_x,z     = " << knee_pos_x << ", " << knee_pos_z << std::endl;
    no_update_calls++;
  #endif

  #ifdef Debug_reward
    double reward_main;
    reward_main  = (theta_hip_oc > NegPI4th) ? (NegPI4th - theta_hip_oc) : (PI4th + theta_hip_oc);
    reward_main += (theta_knee_oc > NegPIhalf) ? (NegPIhalf - theta_knee_oc) : (PIhalf + theta_knee_oc);
    std::cout << "    R1) Reward['main']      = " << reward_main << std::endl;
    reward_total += reward_main;
  #endif

  // The omega term
  *reward += (omega_hip_oc > 0) ? (-0.08 * omega_hip_oc) : (0.08 * omega_hip_oc);
  *reward += (omega_knee_oc > 0) ? (-0.08 * omega_knee_oc) : (0.08 * omega_knee_oc);
  #ifdef Debug_reward
    double reward_omega;
    reward_omega  = (omega_hip_oc > 0) ? (-0.08 * omega_hip_oc) : (0.08 * omega_hip_oc);
    reward_omega += (omega_knee_oc > 0) ? (-0.08 * omega_knee_oc) : (0.08 * omega_knee_oc);
    std::cout << "    R2) Reward['omega']     = " << reward_omega << std::endl;
    reward_total += reward_omega;
  #endif

  // The foot_x term
  if (knee_pos_z < 0.2)
    *reward += (foot_pos_x > 0) ? (-10.0 * foot_pos_x) : (10.0 * foot_pos_x);
  #ifdef Debug_reward
    double reward_foot_x;
    if (knee_pos_z < 0.2)
      reward_foot_x  = (foot_pos_x > 0) ? (-10.0 * foot_pos_x) : (10.0 * foot_pos_x);
    std::cout << "    R3) Reward['foot_x']    = " << reward_foot_x << std::endl;
    reward_total += reward_foot_x;
  #endif

  if (has_touched_ground) {
    #if mjstep_order == delay_valid_obs
      #error "'foot_force_z' needs to be delayed one step (This is not implemented yet)."
             "Read the Readme.md file for more information."
    #endif
    // The foot_f_z term
    if (foot_force_z  < 7.6) *reward += (foot_force_z - 7.6) * 0.5;

    // The foot_z term
    *reward += (foot_pos_z > 0) ? (foot_pos_z * -10) : (foot_pos_z * 10);
    // The knee_z term
    *reward += (knee_pos_z > 0.1) ? ((0.1 - knee_pos_z) * 15) : ((knee_pos_z - 0.1) * 15);
  }

  #ifdef Debug_reward
    double reward_foot_f_z = 0;
    double reward_foot_z = 0;
    double reward_knee_z = 0;
    if (has_touched_ground){
      if (foot_force_z  < 7.6) reward_foot_f_z = (foot_force_z - 7.6) * 0.5;
      reward_foot_z = (foot_pos_z > 0) ? (foot_pos_z * -10) : (foot_pos_z * 10);
      reward_knee_z = (knee_pos_z > 0.1) ? ((0.1 - knee_pos_z) * 15) : ((knee_pos_z - 0.1) * 15);
    }
    std::cout << "    R4) Reward['foot_f_z']  = " << reward_foot_f_z << std::endl;
    std::cout << "        foot_force_z        = " << foot_force_z << std::endl;
    std::cout << "    R5) Reward['foot_z']    = " << reward_foot_z << std::endl;
    std::cout << "    R6) Reward['knee_z']    = " << reward_knee_z << std::endl;
    reward_total += reward_foot_f_z + reward_foot_z + reward_knee_z;
  #endif

  #if add_act_sm_r == True
    #define r_comp_idx (done_inner_steps - 20)
    if (simif->inner_step_count < r_comp_idx) {
      traj_tau_hip[1+simif->inner_step_count] = mj_data->ctrl[hip_ctrl_id];
      traj_tau_knee[1+simif->inner_step_count] = mj_data->ctrl[knee_ctrl_id];
    } else if (simif->inner_step_count == r_comp_idx) {
      *reward += roughness(traj_tau_hip,  0, r_comp_idx, 1) * 100;
      *reward += roughness(traj_tau_knee, 0, r_comp_idx, 1) * 100;
    }
  #endif

  #ifdef Debug_reward
    std::cout << "    R*) Total Reward        = " << reward_total << std::endl;
  #endif

  #if add_omega_smoothness_reward == True
    #error "you possibly need some add_omega_smoothness_reward reward here."
  #endif

  #if add_torque_penalty_on_air == True
    #error "you possibly need some add_torque_penalty_on_air reward here."
  #endif
}

inline void SimplifiedRewardGiver::update_unstable_reward(double* reward){
  *reward -= constraint_penalty;
  #ifdef Debug_reward
    std::cout << "    R*) Reward['sim_error']           = " << -constraint_penalty << std::endl;
  #endif
}

OriginalRewardGiver::OriginalRewardGiver(){
  do_compute_mjids = true;
}

inline void OriginalRewardGiver::reset(mjModel* mj_model,
                                       mjData* mj_data,
                                       SimInterface* simintf) {
  leg_vel_peak = 0;
  has_touched_ground = false;
  simif = simintf;

  if (do_compute_mjids){
    do_compute_mjids = false;
    foot_center_site_mjid = mj_name2id(mj_model, mjOBJ_SITE, "foot-center");
    hip_center_site_mjid = mj_name2id(mj_model, mjOBJ_SITE, "hip-center");
    knee_center_site_mjid = mj_name2id(mj_model, mjOBJ_SITE, "knee-center");
    ground_geom_id = mj_name2id(mj_model, mjOBJ_GEOM, "ground");
    lowerleg_limb_geom_id = mj_name2id(mj_model, mjOBJ_GEOM, "lowerleg-limb");
    base_body_mjid = mj_name2id(mj_model, mjOBJ_BODY, "base");
    upperleg_body_mjid = mj_name2id(mj_model, mjOBJ_BODY, "upperleg");
    lowerleg_body_mjid = mj_name2id(mj_model, mjOBJ_BODY, "lowerleg");
    hip_ctrl_id = mj_name2id(mj_model, mjOBJ_ACTUATOR, "torquehip");
    knee_ctrl_id = mj_name2id(mj_model, mjOBJ_ACTUATOR, "torqueknee");
    if (foot_center_site_mjid < 0)
      mju_error("Site 'foot-center' not found");
    if (hip_center_site_mjid < 0)
      mju_error("Site 'hip-center' not found");
    if (knee_center_site_mjid < 0)
      mju_error("Site 'knee-center' not found");
    if (ground_geom_id < 0)
      mju_error("Geom 'ground' not found");
    if (lowerleg_limb_geom_id < 0)
      mju_error("Geom 'lowerleg-limb' not found");
    if (base_body_mjid < 0)
      mju_error("Body 'base' not found");
    if (upperleg_body_mjid < 0)
      mju_error("Body 'upperleg' not found");
    if (lowerleg_body_mjid < 0)
      mju_error("Body 'lowerleg' not found");
    if (hip_ctrl_id < 0)
      mju_error("Control 'torquehip' not found");
    if (knee_ctrl_id < 0)
      mju_error("Control 'torqueknee' not found");

    base_body_mass_ratio = mj_model->body_mass[base_body_mjid];
    upperleg_body_mass_ratio = mj_model->body_mass[upperleg_body_mjid];
    lowerleg_body_mass_ratio = mj_model->body_mass[lowerleg_body_mjid];
    leg_mass = base_body_mass_ratio + upperleg_body_mass_ratio + lowerleg_body_mass_ratio;
    base_body_mass_ratio = base_body_mass_ratio / leg_mass;
    upperleg_body_mass_ratio = upperleg_body_mass_ratio / leg_mass;
    lowerleg_body_mass_ratio = lowerleg_body_mass_ratio / leg_mass;
  };

  tau_hip = mj_data->ctrl[hip_ctrl_id];
  tau_knee = mj_data->ctrl[knee_ctrl_id];
  tau_hip_old = tau_hip;
  tau_knee_old = tau_knee;

  #ifdef Debug_reward
    slider_mjid = mj_name2id(mj_model, mjOBJ_JOINT, "slider");
    hip_mjid = mj_name2id(mj_model, mjOBJ_JOINT, "hip");
    knee_mjid = mj_name2id(mj_model, mjOBJ_JOINT, "knee");
    if (hip_mjid < 0)
      mju_error("Joint 'hip' not found");
    if (knee_mjid < 0)
      mju_error("Joint 'knee' not found");
    if (slider_mjid < 0)
      mju_error("Joint 'slider' not found");
    no_update_calls = 0;
  #endif
}

inline void OriginalRewardGiver::update_reward(
  double obs_current[obs_dim], // This should be current, since we'll be applying a
                               // reward delay (i.e., the same as observation delay)
  mjModel* mj_model,
  mjData* mj_data,
  double* reward) {

  #define theta_hip_oc obs_current[0]
  #define theta_knee_oc obs_current[1]
  #define omega_hip_oc obs_current[2]
  #define omega_knee_oc obs_current[3]

  // Note: Do not uncomment the following lines unless you know what you're doing!
  //       This may cause the physical variables like foot_pos_x and theta_hip
  //       to be out of sync
  // #define theta_hip_oc mj_data->qpos[mj_model->jnt_qposadr[hip_mjid]]
  // #define theta_knee_oc mj_data->qpos[mj_model->jnt_qposadr[knee_mjid]]
  // #define omega_hip_oc mj_data->qvel[mj_model->jnt_dofadr[hip_mjid]]
  // #define omega_knee_oc mj_data->qvel[mj_model->jnt_dofadr[knee_mjid]]

  #define foot_pos_x mj_data->site_xpos[3*foot_center_site_mjid]
  #define hip_pos_x mj_data->site_xpos[3*hip_center_site_mjid]
  #define knee_pos_x mj_data->site_xpos[3*knee_center_site_mjid]
  #define foot_pos_z mj_data->site_xpos[3*foot_center_site_mjid+2]
  #define hip_pos_z mj_data->site_xpos[3*hip_center_site_mjid+2]
  #define knee_pos_z mj_data->site_xpos[3*knee_center_site_mjid+2]
  #define foot_force_x foot_force[0]
  #define foot_force_z foot_force[2]

  #define base_body_vel_x mj_data->cvel[6*base_body_mjid+3]
  #define base_body_vel_y mj_data->cvel[6*base_body_mjid+4]
  #define base_body_vel_z mj_data->cvel[6*base_body_mjid+5]
  #define upperleg_body_vel_x mj_data->cvel[6*upperleg_body_mjid+3]
  #define upperleg_body_vel_y mj_data->cvel[6*upperleg_body_mjid+4]
  #define upperleg_body_vel_z mj_data->cvel[6*upperleg_body_mjid+5]
  #define lowerleg_body_vel_x mj_data->cvel[6*lowerleg_body_mjid+3]
  #define lowerleg_body_vel_y mj_data->cvel[6*lowerleg_body_mjid+4]
  #define lowerleg_body_vel_z mj_data->cvel[6*lowerleg_body_mjid+5]

  #define PIhalf 1.57079632679489661923
  #define PI4th 0.785398163397448309615
  #define NegPIhalf -1.57079632679489661923
  #define NegPI4th -0.785398163397448309615

  double tau_hip_diff;
  double tau_knee_diff;
  double err_posture;
  bool cnst_viol;
  double leg_vel_z;
  int jump_state = simif->jump_state;

  /////////////////////// Pre-Reward Computation ////////////////////////
  // *Updating foot_force and has_touched_ground*
  _get_contact_state(mj_model, mj_data,
                     ground_geom_id, lowerleg_limb_geom_id,
                     &has_touched_ground, foot_force);

  #ifdef Debug_reward
    std::cout << std::fixed << std::setprecision(8);
    double reward_total = 0;
    std::cout << "  RStep " << no_update_calls+rew_dly_buflen-1 << ":" << std::endl;
    std::cout << "    -->    theta            = " << mj_data->qpos[mj_model->jnt_qposadr[hip_mjid]] \
                                          << ", " << mj_data->qpos[mj_model->jnt_qposadr[knee_mjid]] \
                                           << std::endl;
    std::cout << "    -->    omega            = " << mj_data->qvel[mj_model->jnt_dofadr[hip_mjid]] \
                                          << ", " << mj_data->qvel[mj_model->jnt_dofadr[knee_mjid]] \
                                           << std::endl;
    std::cout << "    -->    theta_input      = " << obs_current[0] << ", " << obs_current[1] << std::endl;
    std::cout << "    -->    omega_input      = " << obs_current[2] << ", " << obs_current[3] << std::endl;
    std::cout << "    -->    foot_pos_x,z     = " << foot_pos_x << ", " << foot_pos_z << std::endl;
    std::cout << "    -->    knee_pos_x,z     = " << knee_pos_x << ", " << knee_pos_z << std::endl;
    std::cout << "    -->    tau              = " << tau_hip << ", " << tau_knee << std::endl;
    std::cout << "    -->    tau_old          = " << tau_hip_old << ", " << tau_knee_old << std::endl;
    std::cout << "    -->    jump_state       = " << jump_state << std::endl;
    no_update_calls++;
  #endif

  ////////////////////////// Reward Computation ///////////////////////////
  #if turn_off_constraints == False
    cnst_viol = (omega_hip_oc > omega_hip_maxabs);
    cnst_viol = cnst_viol || (omega_hip_oc < neg_omega_hip_maxabs);
    cnst_viol = cnst_viol || (omega_knee_oc > omega_knee_maxabs);
    cnst_viol = cnst_viol || (omega_knee_oc < neg_omega_knee_maxabs);
    cnst_viol = cnst_viol || (theta_hip_oc < theta_hip_bounds_low);
    cnst_viol = cnst_viol || (theta_hip_oc > theta_hip_bounds_high);
    cnst_viol = cnst_viol || (theta_knee_oc < theta_knee_bounds_low);
    cnst_viol = cnst_viol || (theta_knee_oc > theta_knee_bounds_high);
    cnst_viol = cnst_viol || (hip_pos_z < hip_minimum_z);
    cnst_viol = cnst_viol || (knee_pos_z < knee_minimum_z);
  #else
    cnst_viol = false;
  #endif

  if (cnst_viol) {
    *reward -= constraint_penalty;
    #ifdef Debug_reward
      double reward_constraint;
      reward_constraint = -constraint_penalty;
      std::cout << "    R0) Reward['cons_*']              = " << reward_constraint << std::endl;
      reward_total += reward_constraint;
    #endif
  } else {
    // The torque_smoothness term
    tau_hip_diff = (tau_hip - tau_hip_old);
    tau_knee_diff = (tau_knee - tau_knee_old);
    *reward -= torque_smoothness_coeff * (tau_hip_diff * tau_hip_diff + tau_knee_diff * tau_knee_diff);

    #ifdef Debug_reward
      double reward_torque_smoothness;
      reward_torque_smoothness = -torque_smoothness_coeff * ((tau_hip_diff * tau_hip_diff) +
                                                             (tau_knee_diff * tau_knee_diff));
      std::cout << "    R1) Reward['torque_smoothness']   = " << reward_torque_smoothness << std::endl;
      reward_total += reward_torque_smoothness;
    #endif

    #if stand_reward == sparse
      #error "Need some sparse reward implementation here."
    #endif
    if ((jump_state == 0) || (jump_state == 5)) {
      // The foot_pos_x term
      *reward += (-1000.0) * (foot_pos_x * foot_pos_x);
      #ifdef Debug_reward
        double reward_foot_pos_x;
        reward_foot_pos_x = (-1000.0) * (foot_pos_x * foot_pos_x);
        std::cout << "    R2) Reward['foot_pos_x']          = " << reward_foot_pos_x << std::endl;
        reward_total += reward_foot_pos_x;
      #endif

      // The posture term
      err_posture = ((hip_pos_z - foot_pos_z) - posture_height);
      *reward += (-100.0) * (err_posture * err_posture);

      #ifdef Debug_reward
        double reward_posture;
        reward_posture = (-100.0) * (err_posture * err_posture);
        std::cout << "    R3) Reward['posture']             = " << reward_posture << std::endl;
        reward_total += reward_posture;
      #endif

      // The torque term
      *reward += (-0.0001) * ((tau_hip * tau_hip) + (tau_knee * tau_knee));

      #ifdef Debug_reward
        double reward_torque;
        reward_torque = (-0.0001) * ((tau_hip * tau_hip) + (tau_knee * tau_knee));
        std::cout << "    R4) Reward['torque']              = " << reward_torque << std::endl;
        reward_total += reward_torque;
      #endif

      // The velocity term
      *reward += (-0.1) * ((omega_hip_oc * omega_hip_oc) + (omega_knee_oc * omega_knee_oc));

      #ifdef Debug_reward
        double reward_velocity;
        reward_velocity = (-0.1) * ((omega_hip_oc * omega_hip_oc) + (omega_knee_oc * omega_knee_oc));
        std::cout << "    R5) Reward['velocity']            = " << reward_velocity << std::endl;
        reward_total += reward_velocity;
      #endif
    } else if (jump_state == 1) { // jump
      #if timed_jump == True
        leg_vel_z = ((base_body_mass_ratio     * base_body_vel_z    ) +
                     (upperleg_body_mass_ratio * upperleg_body_vel_z) +
                     (lowerleg_body_mass_ratio * lowerleg_body_vel_z));
        // The jump_vel term
        if (leg_vel_z < max_leg_vel){
          if (leg_vel_z > leg_vel_peak)
            *reward += jump_vel_coeff * (leg_vel_z - leg_vel_peak);
        } else {
          *reward += (max_leg_vel - leg_vel_z);
        };

        #ifdef Debug_reward
          double reward_jump_vel;
          if (leg_vel_z < max_leg_vel){
            if (leg_vel_z > leg_vel_peak)
              reward_jump_vel = jump_vel_coeff * (leg_vel_z - leg_vel_peak);
            else
              reward_jump_vel = 0;
          } else {
            reward_jump_vel = (max_leg_vel - leg_vel_z);
          };
          std::cout << "    R6) Reward['jump_vel']            = " << reward_jump_vel << std::endl;
          reward_total += reward_jump_vel;
        #endif

        leg_vel_peak = (leg_vel_peak > leg_vel_z) ? leg_vel_peak : leg_vel_z;
      #elif timed_jump == False
        // The hip_pos_z term
        *reward += 100 * (hip_pos_z - max_reach);

        #ifdef Debug_reward
          double reward_hip_pos_z;
          reward_hip_pos_z = 100 * (hip_pos_z - max_reach);
          std::cout << "    R7) Reward['hip_pos_z']           = " << reward_hip_pos_z << std::endl;
          reward_total += reward_hip_pos_z;
        #endif

        #if jump_vel_reward == True
          #error "You need some jump_vel reward implementation here."
        #endif
      #else
        #error "timed_jump not implemented."
      #endif
    } else if (jump_state == 2) { // liftoff
      #if jump_vel_reward == True
        #error "You need some jump_vel_liftoff reward implementation here."
      #endif
    } else if (jump_state == 3) { // in air, far from ground
      // The hip_pos_z term
      *reward += 100 * (hip_pos_z - max_reach);
      #ifdef Debug_reward
        double reward_hip_pos_z;
        reward_hip_pos_z = 100 * (hip_pos_z - max_reach);
        std::cout << "    R8) Reward['hip_pos_z']           = " << reward_hip_pos_z << std::endl;
        reward_total += reward_hip_pos_z;
      #endif
      #if jump_vel_reward == True
        #error "You need some jump_vel reward implementation here."
      #endif
    };
  }

  #ifdef Debug_reward
    std::cout << "    R*) Total Reward                  = " << reward_total << std::endl;
  #endif

  /////////////////////// Post-Reward Computation ////////////////////////
  // *Setting tau_old elements
  tau_hip_old = tau_hip;
  tau_knee_old = tau_knee;
  // *Setting tau elements
  tau_hip = mj_data->ctrl[hip_ctrl_id];
  tau_knee = mj_data->ctrl[knee_ctrl_id];
}

inline void OriginalRewardGiver::update_unstable_reward(double* reward){
  *reward -= constraint_penalty;
  #ifdef Debug_reward
    std::cout << "    R*) Reward['sim_error']           = " << -constraint_penalty << std::endl;
  #endif
}

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

  // Setting the mujoco timestep option
  mj_model->opt.timestep = physics_timestep;

  // make data
  mj_data = mj_makeData(mj_model);
  set_ground_contact();

  // finding joint ids for qpos and qvel setting later
  slider_mjid = mj_name2id(mj_model, mjOBJ_JOINT, "slider");
  hip_mjid = mj_name2id(mj_model, mjOBJ_JOINT, "hip");
  knee_mjid = mj_name2id(mj_model, mjOBJ_JOINT, "knee");

  // finding ctrl ids for setting actuations later
  hip_ctrl_id = mj_name2id(mj_model, mjOBJ_ACTUATOR, "torquehip");
  knee_ctrl_id = mj_name2id(mj_model, mjOBJ_ACTUATOR, "torqueknee");

  if (slider_mjid < 0)
    mju_error("Joint 'slider' not found");
  if (hip_mjid < 0)
    mju_error("Joint 'hip' not found");
  if (knee_mjid < 0)
    mju_error("Joint 'knee' not found");
  if (hip_ctrl_id < 0)
    mju_error("Control 'torquehip' not found");
  if (knee_ctrl_id < 0)
    mju_error("Control 'torqueknee' not found");
}

int SimInterface::set_ground_contact() {
  int pair_id; // This should store the 'ground-contact' pair id
  char ref_name[20] = "ground-contact\0";
  bool is_gnd_cntct;
  int i, j; // Temporary loop variables
  int k, l; // General temporary variables

  ////////////////////////////////////////////////
  // Section 1: Finding the reference pair id. //
  ////////////////////////////////////////////////
  is_gnd_cntct = true;
  for(pair_id = 0; pair_id < mj_model->npair; pair_id++ ){
    for(i = 0 ;
        i < (mj_model->name_pairadr[pair_id+1] - mj_model->name_pairadr[pair_id]) ;
        i ++ ){
      j = i + mj_model->name_pairadr[pair_id];
      is_gnd_cntct = is_gnd_cntct && (ref_name[i] == mj_model->names[j]);
    }
    is_gnd_cntct = is_gnd_cntct && (ref_name[i] == '\0'); // Checking that ref_name is finished!
    if (is_gnd_cntct)
      break;
  }

  if ( !is_gnd_cntct ){
    std::cout << "Error: Could not find the ground pair! a 'ground-contact' pair must exist in the xml file...";
    assert(is_gnd_cntct);
  }

  /////////////////////////////////////////////////////////////////////
  // Section 2: Setting pair_solref and pair_solimp for the gnd pair //
  /////////////////////////////////////////////////////////////////////
  k = pair_id * mjNREF;
  l = pair_id * mjNIMP;
  #if ground_contact_type == compliant_v1        // legacy compliant
    mj_model->pair_solref[k    ] = 0.015;
    mj_model->pair_solref[k + 1] = 0.180;
    mj_model->pair_solimp[l    ] = 0.010;
    mj_model->pair_solimp[l + 1] = 0.950;
    mj_model->pair_solimp[l + 2] = 0.001;
  #elif ground_contact_type == compliant_v2      // new (chad) compliant
    mj_model->pair_solref[k    ] = 0.015;
    mj_model->pair_solref[k + 1] = 0.240;
    mj_model->pair_solimp[l    ] = 0.010;
    mj_model->pair_solimp[l + 1] = 0.950;
    mj_model->pair_solimp[l + 2] = 0.001;
    mj_model->pair_solimp[l + 3] = 0.500;
    mj_model->pair_solimp[l + 4] = 9.000;
  #elif ground_contact_type == noncompliant     // noncompliant
    mj_model->pair_solref[k    ] = 0.002;
    mj_model->pair_solref[k + 1] = 1.000;
    mj_model->pair_solimp[l    ] = 0.900;
    mj_model->pair_solimp[l + 1] = 0.950;
    mj_model->pair_solimp[l + 2] = 0.001;
  #else
    #error "undefined ground_contact_type"
  #endif

  return 0;
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

void SimInterface::update_mj_obs() {
  // Here is the order:
  //   1) joint_state --> (theta_hip, theta_knee, omega_hip, omega_knee)
  //   2) contact_obs --> foot_in_contact
  //   3) jumping_obs --> 1.0 if self.jump_state == 1 else 0.0
  //   4) extra_obs_dim --> zeros(extra_obs_dim)
  mj_obs[0] = mj_data->qpos[mj_model->jnt_qposadr[hip_mjid]];     //theta_hip
  mj_obs[1] = mj_data->qpos[mj_model->jnt_qposadr[knee_mjid]];    //theta_knee
  mj_obs[2] = mj_data->qvel[mj_model->jnt_dofadr[hip_mjid]];      //omega_hip
  mj_obs[3] = mj_data->qvel[mj_model->jnt_dofadr[knee_mjid]];     //omega_knee

  #if do_obs_noise == True
    non_noisy_omega_hip = mj_obs[2];
    non_noisy_omega_knee = mj_obs[3];
    int curr_nis_idx = (noise_idx + inner_step_count) % noise_rows;
    // adding hip omega noise
    if ((mj_obs[2] > 1) || (mj_obs[2] < -1))
      mj_obs[2] += omega_noise_scale * noise_arr[curr_nis_idx][0];
    else if (mj_obs[2] > 0)
      mj_obs[2] += omega_noise_scale * mj_obs[2] * noise_arr[curr_nis_idx][0];
    else
      mj_obs[2] -= omega_noise_scale * mj_obs[2] * noise_arr[curr_nis_idx][0];

    // adding knee omega noise
    if ((mj_obs[3] > 1) || (mj_obs[3] < -1))
      mj_obs[3] += omega_noise_scale * noise_arr[curr_nis_idx][1];
    else if (mj_obs[3] > 0)
      mj_obs[3] += omega_noise_scale * mj_obs[3] * noise_arr[curr_nis_idx][1];
    else
      mj_obs[3] -= omega_noise_scale * mj_obs[3] * noise_arr[curr_nis_idx][1];
  #endif

  #if contact_obs == True
    #error "contact_obs needs implementation here."
  #endif

  #if jumping_obs == True
    if (jump_state == 1)
      mj_obs[4] = 1;
    else
      mj_obs[4] = 0;
  #endif

  #if extra_obs_dim == True
    #error "extra_obs_dim needs implementation here."
  #endif
}

double* SimInterface::reset(double* init_args_double, int* init_args_int) {
  #define theta_hip init_args_double[0]
  #define theta_knee init_args_double[1]
  #define omega_hip init_args_double[2]
  #define omega_knee init_args_double[3]
  #define pos_slider init_args_double[4]
  #define vel_slider init_args_double[5]
  #define jumping_time init_args_double[6]
  #define noise_index init_args_int[0]

  inner_step_count = 0;
  outer_step_count = 0;

  // Resetting the motor state
  motor_tau_state_hip = 0;
  motor_tau_state_hip_dot = 0;
  motor_tau_state_knee = 0;
  motor_tau_state_knee_dot = 0;

  // Resetting the observation delay buffer
  obs_dlybuf_ever_pushed = false;
  obs_dlybuf_push_idx = 0;

  // Resetting the torque delay buffer
  trq_dlybuf_ever_pushed = false;
  trq_dlybuf_push_idx = 0;

  // Resetting the reward delay buffer
  rew_dlybuf_ever_pushed = false;
  rew_dlybuf_push_idx = 0;

  jump_state = 0;
  #if do_jump == True
    jump_time = jumping_time;
  #endif

  #if do_obs_noise == True
    noise_idx = noise_index;
  #endif

  // Erasing all mj_data
  mj_resetData(mj_model, mj_data);

  // Setting qpos elements
  mj_data->qpos[mj_model->jnt_qposadr[slider_mjid]] = pos_slider;
  mj_data->qpos[mj_model->jnt_qposadr[hip_mjid]] = theta_hip;
  mj_data->qpos[mj_model->jnt_qposadr[knee_mjid]] = theta_knee;
  // Setting qvel elements
  mj_data->qvel[mj_model->jnt_dofadr[slider_mjid]] = vel_slider;
  mj_data->qvel[mj_model->jnt_dofadr[hip_mjid]] = omega_hip;
  mj_data->qvel[mj_model->jnt_dofadr[knee_mjid]] = omega_knee;

  // Calling mj_forward to validate all other mj_data variables without
  // integrating through time.
  mj_forward(mj_model, mj_data);

  // Resetting the reward object
  rew_giver.reset(mj_model, mj_data, this);

  #if check_mj_unstability == True
    is_mj_stable = mj_isStable(mj_model, mj_data);
    if (!is_mj_stable)
      mju_error("SimInterface: The environment is unstable even after resetting!");
  #endif

  // Notes: 1) The step_inner call updates inner_step_count after
  //           processing the observation and possibly adding noise.
  //        2) update_mj_obs() uses inner_step_count to determine the
  //           noise index
  //        3) Therefore, in order not to have the first noise repeated,
  //           we have to decrement the reset noise index.
  //        4) This is why we artificially decrement & increment
  //           inner_step_count before calling update_mj_obs() here.
  inner_step_count--;
  update_mj_obs();
  inner_step_count++;
  return add_obs_buff(mj_obs);

  #undef theta_hip
  #undef theta_knee
  #undef omega_hip
  #undef omega_knee
  #undef pos_slider
  #undef vel_slider
  #undef jumping_time
  #undef noise_index
}

void SimInterface::step_inner(double action_raw[act_dim]) {
  double* joint_torque_current;
  double abs_jnt_omega;

  #if use_legacy_model == True
    double maxabs_deduction, maxabs_tau_jnt;
  #endif

  #ifdef Debug_step_inner
    std::cout << "  Step inner " << inner_step_count << ":" << std::endl;
  #endif

  #if filter_action == True
    #error "filter_action needs some implementation here."
  #else
    #define action action_raw
  #endif

  #if action_type == torque
    #define joint_torque_command action
  #elif action_type == jointspace_pd
    #define theta_hip_dlyd  obs_delay_buff[obs_dlybuf_push_idx][0]
    #define theta_knee_dlyd obs_delay_buff[obs_dlybuf_push_idx][1]
    #define omega_hip_dlyd  obs_delay_buff[obs_dlybuf_push_idx][2]
    #define omega_knee_dlyd obs_delay_buff[obs_dlybuf_push_idx][3]
    joint_torque_command[0] = -hip_kP *(theta_hip_dlyd  - action[0]) - hip_kD *omega_hip_dlyd;
    joint_torque_command[1] = -knee_kP*(theta_knee_dlyd - action[1]) - knee_kD*omega_knee_dlyd;
  #elif action_type == workspace_pd
    #error "workspace_pd needs some implementation here "
           "(translating workspace_pd to joint_torque_command)."
  #else
    #error "action_type not implemented."
  #endif

  #ifdef Debug_step_inner
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "    0) theta_dlyd           = " << theta_hip_dlyd << ", " << theta_knee_dlyd << std::endl;
    std::cout << "       omega_dlyd           = " << omega_hip_dlyd << ", " << omega_knee_dlyd << std::endl;
    std::cout << "    1) tau                  = " << joint_torque_command[0] << ", " \
              << joint_torque_command[1] << std::endl;
  #endif

  // Applying actuator delay
  joint_torque_current = add_trq_buff(joint_torque_command);

  #ifdef Debug_step_inner
  std::cout << "    2) joint_torque_current = " << joint_torque_current[0] << \
      ", " << joint_torque_current[1] << std::endl;
  #endif

  #if use_motor_model == True
    // Before applying the motor model
    #define joint_torque_hip motor_tau_state_hip
    #define joint_torque_knee motor_tau_state_knee
  #elif use_motor_model == False
  #else
    #error "use_motor_model not implemented."
  #endif

  #ifdef Debug_step_inner
    std::cout << "    3) joint_torque         = " << joint_torque_hip << \
        ", " << joint_torque_knee << std::endl;
  #endif

  // The motor saturation model
  #if use_legacy_model == True
    // Capping the hip torque according to its omega
    #if do_obs_noise == True
      abs_jnt_omega = non_noisy_omega_hip; // The non-noisy backup version
    #else
      abs_jnt_omega = mj_obs[2]; // omega_hip = mj_obs[2]
    #endif
    if (abs_jnt_omega < 0)
      abs_jnt_omega = abs_jnt_omega * -1;
    maxabs_deduction = tau_omega_slope * (abs_jnt_omega - maxabs_omega_for_tau);
    maxabs_deduction = (maxabs_deduction > 0) ? 0 : maxabs_deduction;
    maxabs_tau_jnt = maxabs_tau + maxabs_deduction;
    maxabs_tau_jnt = (maxabs_tau_jnt < 0) ? 0 : maxabs_tau_jnt;
    if (joint_torque_hip > maxabs_tau_jnt)
      joint_torque_hip_capped = maxabs_tau_jnt;
    else if (joint_torque_hip < -maxabs_tau_jnt)
      joint_torque_hip_capped = -maxabs_tau_jnt;
    else
      joint_torque_hip_capped = joint_torque_hip;

    // Capping the knee torque according to its omega
    #if do_obs_noise == True
      abs_jnt_omega = non_noisy_omega_knee; // The non-noisy backup version
    #else
      abs_jnt_omega = mj_obs[3]; // omega_knee = mj_obs[3]
    #endif
    if (abs_jnt_omega < 0)
      abs_jnt_omega = abs_jnt_omega * -1;
    maxabs_deduction = tau_omega_slope * (abs_jnt_omega - maxabs_omega_for_tau);
    maxabs_deduction = (maxabs_deduction > 0) ? 0 : maxabs_deduction;
    maxabs_tau_jnt = maxabs_tau + maxabs_deduction;
    maxabs_tau_jnt = (maxabs_tau_jnt < 0) ? 0 : maxabs_tau_jnt;
    if (joint_torque_knee > maxabs_tau_jnt)
      joint_torque_knee_capped = maxabs_tau_jnt;
    else if (joint_torque_knee < -maxabs_tau_jnt)
      joint_torque_knee_capped = -maxabs_tau_jnt;
    else
      joint_torque_knee_capped = joint_torque_knee;
  #elif use_dyno_model == True
    #if do_obs_noise == True
      abs_jnt_omega = non_noisy_omega_hip; // The non-noisy backup version
    #else
      abs_jnt_omega = mj_obs[2]; // omega_hip = mj_obs[2]
    #endif
    if (abs_jnt_omega < 0)
      abs_jnt_omega = abs_jnt_omega * -1;
    joint_torque_hip_capped = dyno_motor_model(abs_jnt_omega, joint_torque_hip);

    #if do_obs_noise == True
      abs_jnt_omega = non_noisy_omega_knee; // The non-noisy backup version
    #else
      abs_jnt_omega = mj_obs[3]; // omega_knee = mj_obs[3]
    #endif
    if (abs_jnt_omega < 0)
      abs_jnt_omega = abs_jnt_omega * -1;
    joint_torque_knee_capped = dyno_motor_model(abs_jnt_omega, joint_torque_knee);
  #elif use_naive_model == True
    #if do_obs_noise == True
      abs_jnt_omega = non_noisy_omega_hip; // The non-noisy backup version
    #else
      abs_jnt_omega = mj_obs[2]; // omega_hip = mj_obs[2]
    #endif
    if (abs_jnt_omega < 0)
      abs_jnt_omega = abs_jnt_omega * -1;
    joint_torque_hip_capped = naive_motor_model(abs_jnt_omega, joint_torque_hip);

    #if do_obs_noise == True
      abs_jnt_omega = non_noisy_omega_knee; // The non-noisy backup version
    #else
      abs_jnt_omega = mj_obs[3]; // omega_knee = mj_obs[3]
    #endif
    if (abs_jnt_omega < 0)
      abs_jnt_omega = abs_jnt_omega * -1;
    joint_torque_knee_capped = naive_motor_model(abs_jnt_omega, joint_torque_knee);
  #else
    #error "Need motor saturation model implementation here."
  #endif

  // Applying the output torque scale
  joint_torque_hip_capped *= output_torque_scale;
  joint_torque_knee_capped *= output_torque_scale;

  // Applying motor model (for the next round!)
  #if use_motor_model == True
    solve_motor_ode(joint_torque_current[0],
                    motor_tau_state_hip, motor_tau_state_hip_dot,
                    inner_step_time,
                    &motor_tau_state_hip, &motor_tau_state_hip_dot);

    solve_motor_ode(joint_torque_current[1],
                    motor_tau_state_knee, motor_tau_state_knee_dot,
                    inner_step_time,
                    &motor_tau_state_knee, &motor_tau_state_knee_dot);
  #elif use_motor_model == False
  #else
    #error "use_motor_model not implemented."
  #endif

  #ifdef Debug_step_inner
    std::cout << "    4) joint_torque_capped  = " << joint_torque_hip_capped << \
        ", " << joint_torque_knee_capped << std::endl;
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

  // Notes: 1) When the mujoco simulation is stable,
  //           we can safely update mj_obs.
  //       2) When mujoco simulation gets unstable
  //          -> 1) No mj_step calls whatsoever!
  //          -> 2) We shouldn't call update_mj_obs!

  #if inner_loops_per_outer_loop == 1
    // NOTE: Please read the `The Invalid State Problem of Mujoco`
    //       section of the `Readme.md` file to understand the issue behind
    //       `mjstep_order` and how to set it properly.
    #if mjstep_order == mjstep1_after_mjstep
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
        for (int pl=0; pl < extra_physics_loops_per_inner_loop; pl++){
          mj_data->ctrl[hip_ctrl_id] = joint_torque_hip_capped;
          mj_data->ctrl[knee_ctrl_id] = joint_torque_knee_capped;
          mj_step(mj_model, mj_data);
        }
        mj_data->ctrl[hip_ctrl_id] = joint_torque_hip_capped;
        mj_data->ctrl[knee_ctrl_id] = joint_torque_knee_capped;
        mj_step(mj_model, mj_data);

        // Phase 3: Calling `mj_step1`
        //    Note: This call validates all mj_data attributes
        mj_step1(mj_model, mj_data);

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
      if (is_mj_stable) {
        rew_giver.update_reward(mj_obs, mj_model, mj_data, reward);
      } else {
        rew_giver.update_unstable_reward(reward);
      }
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
        for (int pl=0; pl < extra_physics_loops_per_inner_loop; pl++){
          mj_data->ctrl[hip_ctrl_id] = joint_torque_hip_capped;
          mj_data->ctrl[knee_ctrl_id] = joint_torque_knee_capped;
          mj_step(mj_model, mj_data);
        }
        mj_data->ctrl[hip_ctrl_id] = joint_torque_hip_capped;
        mj_data->ctrl[knee_ctrl_id] = joint_torque_knee_capped;
        mj_step1(mj_model, mj_data);

        is_mj_stable = mj_isStable(mj_model, mj_data);
      }
      // Phase 3: Updating the reward
      //    Note: Now that we have all the states validated and synchoronized,
      //          we can call `update_reward`.
      // Comment: You should be able to call `update_mj_obs()`without having
      //          to worry about state validation issues. If this doesn't seem
      //          to be the case, this is possibly indicitave of a bug.
      if (is_mj_stable)
        rew_giver.update_reward(mj_obs, mj_model, mj_data, reward);
      else
        rew_giver.update_unstable_reward(reward);
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
        for (int pl=0; pl < extra_physics_loops_per_inner_loop; pl++){
          mj_data->ctrl[hip_ctrl_id] = joint_torque_hip_capped;
          mj_data->ctrl[knee_ctrl_id] = joint_torque_knee_capped;
          mj_step(mj_model, mj_data);
        }
        mj_data->ctrl[hip_ctrl_id] = joint_torque_hip_capped;
        mj_data->ctrl[knee_ctrl_id] = joint_torque_knee_capped;
        mj_step(mj_model, mj_data);
        is_mj_stable = mj_isStable(mj_model, mj_data);
      }

      // Phase 3: Updating the reward
      //    Note: Now that we have all the states validated and synchoronized,
      //          we can call `update_reward`.
      // Comment: Do not call `update_mj_obs()`. update_mj_obs will update
      //          mj_obs, which will make the physical variables out of sync.

      if (is_mj_stable)
        rew_giver.update_reward(mj_obs, mj_model, mj_data, reward);
      else
        rew_giver.update_unstable_reward(reward);

      *reward = add_rew_buff(*reward);

      if (is_mj_stable)
        update_mj_obs();

      *next_obs = add_obs_buff(mj_obs);
    #else
      #error "mjstep_order not implemented"
    #endif
    inner_step_count++;          // Incrementing the inner step counter
  #else
    double single_step_rew;
    for (int i=0; i<inner_loops_per_outer_loop; i++){
      single_step_rew = 0;
      // NOTE: I left a lot of comments in the `#if inner_loops_per_outer_loop == 1` case,
      //       and I avoided repeating them here to make the code less cluttered. These two
      //       cases should be identical except for the `for` loop needed in the second case.
      #if mjstep_order == mjstep1_after_mjstep
        if (is_mj_stable) {
          step_inner(action_raw);
          for (int pl=0; pl < extra_physics_loops_per_inner_loop; pl++){
            mj_data->ctrl[hip_ctrl_id] = joint_torque_hip_capped;
            mj_data->ctrl[knee_ctrl_id] = joint_torque_knee_capped;
            mj_step(mj_model, mj_data);
          }
          mj_data->ctrl[hip_ctrl_id] = joint_torque_hip_capped;
          mj_data->ctrl[knee_ctrl_id] = joint_torque_knee_capped;
          mj_step(mj_model, mj_data);
          mj_step1(mj_model, mj_data);
          is_mj_stable = mj_isStable(mj_model, mj_data);
        }
        if (is_mj_stable)
          update_mj_obs();
        *next_obs = add_obs_buff(mj_obs);
        if (is_mj_stable)
          rew_giver.update_reward(mj_obs, mj_model, mj_data, &single_step_rew);
        else
          rew_giver.update_unstable_reward(&single_step_rew);
        *reward += add_rew_buff(single_step_rew);
      #elif mjstep_order == separate_mjstep1_mjstep2
        if (is_mj_stable) {
          step_inner(action_raw);
          for (int pl=0; pl < extra_physics_loops_per_inner_loop; pl++){
            mj_data->ctrl[hip_ctrl_id] = joint_torque_hip_capped;
            mj_data->ctrl[knee_ctrl_id] = joint_torque_knee_capped;
            mj_step(mj_model, mj_data);
          }
          mj_data->ctrl[hip_ctrl_id] = joint_torque_hip_capped;
          mj_data->ctrl[knee_ctrl_id] = joint_torque_knee_capped;
          mj_step1(mj_model, mj_data);
          is_mj_stable = mj_isStable(mj_model, mj_data);
        }

        if (is_mj_stable)
          rew_giver.update_reward(mj_obs, mj_model, mj_data, &single_step_rew);
        else
          rew_giver.update_unstable_reward(&single_step_rew);

        *reward += add_rew_buff(single_step_rew);

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
          for (int pl=0; pl < extra_physics_loops_per_inner_loop; pl++){
            mj_data->ctrl[hip_ctrl_id] = joint_torque_hip_capped;
            mj_data->ctrl[knee_ctrl_id] = joint_torque_knee_capped;
            mj_step(mj_model, mj_data);
          }
          mj_data->ctrl[hip_ctrl_id] = joint_torque_hip_capped;
          mj_data->ctrl[knee_ctrl_id] = joint_torque_knee_capped;
          mj_step(mj_model, mj_data);
          is_mj_stable = mj_isStable(mj_model, mj_data);
        }

        if (is_mj_stable)
          rew_giver.update_reward(mj_obs, mj_model, mj_data, &single_step_rew);
        else
          rew_giver.update_unstable_reward(&single_step_rew);

        *reward += add_rew_buff(single_step_rew);

        if (is_mj_stable)
          update_mj_obs();

        *next_obs = add_obs_buff(mj_obs);
      #else
        #error "mjstep_order not implemented"
      #endif
      inner_step_count++;
    };
    *reward /= inner_loops_per_outer_loop;
  #endif

  outer_step_count++;

  // *Determining the next jump_state*
  //   In the most recent version, jump_state is updated in the outer
  //   loop after everything was calculated.
  #if do_jump == True
    #if (timed_jump == True)
      // Comment: Unlike the python script, the jump_state we compute at the end of the
      //          outer loop will not affect the jumping_obs that will be sent out.
      //          In other words, mj_obs was updated way before jump_state got updated,
      //          and we shouldn't try to change the observation crudely.
      double simif_time = (inner_step_count) * inner_step_time;
      if (jump_state == 0){
        if (simif_time >= jump_time)
          jump_state = 1;
      } else if (jump_state == 1) {
        if (simif_time >= (jump_time + jump_push_time))
          jump_state = 5;
      } else if (jump_state == 5) {
        if (simif_time >= (jump_time + jump_push_time + jump_fly_time))
          jump_state = 6;
      } else if (jump_state != 6) {
        throw std::runtime_error("invalid jump_state encountered");
      }
    #else
      #error "jump_state update rule is needs implementation in this case."
    #endif
  #endif

  *done = (inner_step_count >=  done_inner_steps) || (jump_state == 6);

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
  write_opt("compliant_v1", compliant_v1, &key_p, &val_p, a, b);
  write_opt("compliant_v2", compliant_v2, &key_p, &val_p, a, b);
  write_opt("noncompliant", noncompliant, &key_p, &val_p, a, b);
  write_opt("torque", torque, &key_p, &val_p, a, b);
  write_opt("jointspace_pd", jointspace_pd, &key_p, &val_p, a, b);
  write_opt("workspace_pd", workspace_pd, &key_p, &val_p, a, b);
  write_opt("file_path_type", file_path_type, &key_p, &val_p, a, b);
  write_opt("content_type", content_type, &key_p, &val_p, a, b);
  write_opt("tanh_activation", tanh_activation, &key_p, &val_p, a, b);
  write_opt("relu_activation", relu_activation, &key_p, &val_p, a, b);
  write_opt("dense", dense, &key_p, &val_p, a, b);
  write_opt("sparse", sparse, &key_p, &val_p, a, b);
  write_opt("simplified", simplified, &key_p, &val_p, a, b);
  write_opt("dyno", dyno, &key_p, &val_p, a, b);
  write_opt("naive", naive, &key_p, &val_p, a, b);
  write_opt("legacy", legacy, &key_p, &val_p, a, b);
  write_opt("PI", PI, &key_p, &val_p, a, b);
  // settings
  write_opt("xml_type", xml_type, &key_p, &val_p, a, b);
  write_opt("inner_loop_rate", inner_loop_rate, &key_p, &val_p, a, b);
  write_opt("outer_loop_rate", outer_loop_rate, &key_p, &val_p, a, b);
  write_opt("ground_contact_type", ground_contact_type, &key_p, &val_p, a, b);
  write_opt("do_obs_noise", do_obs_noise, &key_p, &val_p, a, b);
  write_opt("jumping_obs", jumping_obs, &key_p, &val_p, a, b);
  write_opt("physics_timestep", physics_timestep, &key_p, &val_p, a, b);
  write_opt("torque_delay", torque_delay, &key_p, &val_p, a, b);
  write_opt("observation_delay", observation_delay, &key_p, &val_p, a, b);
  write_opt("time_before_reset", time_before_reset, &key_p, &val_p, a, b);
  write_opt("add_act_sm_r", add_act_sm_r, &key_p, &val_p, a, b);
  write_opt("action_type", action_type, &key_p, &val_p, a, b);
  write_opt("use_motor_model", use_motor_model, &key_p, &val_p, a, b);
  write_opt("motor_saturation_model", motor_saturation_model, &key_p, &val_p, a, b);
  write_opt("use_dyno_model", use_dyno_model, &key_p, &val_p, a, b);
  write_opt("use_naive_model", use_naive_model, &key_p, &val_p, a, b);
  write_opt("use_legacy_model", use_legacy_model, &key_p, &val_p, a, b);
  write_opt("do_jump", do_jump, &key_p, &val_p, a, b);
  write_opt("timed_jump", timed_jump, &key_p, &val_p, a, b);
  write_opt("stand_reward", stand_reward, &key_p, &val_p, a, b);
  write_opt("motor_root_real", motor_root_real, &key_p, &val_p, a, b);
  write_opt("motor_root_imag", motor_root_imag, &key_p, &val_p, a, b);
  write_opt("hip_kP", hip_kP, &key_p, &val_p, a, b);
  write_opt("hip_kD", hip_kD, &key_p, &val_p, a, b);
  write_opt("knee_kP", knee_kP, &key_p, &val_p, a, b);
  write_opt("knee_kD", knee_kD, &key_p, &val_p, a, b);
  write_opt("maxabs_tau", maxabs_tau, &key_p, &val_p, a, b);
  write_opt("output_torque_scale", output_torque_scale, &key_p, &val_p, a, b);
  write_opt("check_mj_unstability", check_mj_unstability, &key_p, &val_p, a, b);
  write_opt("maxabs_omega_for_tau", maxabs_omega_for_tau, &key_p, &val_p, a, b);
  write_opt("tau_omega_slope", tau_omega_slope, &key_p, &val_p, a, b);
  write_opt("omega_noise_scale", omega_noise_scale, &key_p, &val_p, a, b);
  write_opt("noise_rows", noise_rows, &key_p, &val_p, a, b);
  write_opt("jump_push_time", jump_push_time, &key_p, &val_p, a, b);
  write_opt("jump_fly_time", jump_fly_time, &key_p, &val_p, a, b);
  write_opt("torque_smoothness_coeff", torque_smoothness_coeff, &key_p, &val_p, a, b);
  write_opt("posture_height", posture_height, &key_p, &val_p, a, b);
  write_opt("max_leg_vel", max_leg_vel, &key_p, &val_p, a, b);
  write_opt("jump_vel_coeff", jump_vel_coeff, &key_p, &val_p, a, b);
  write_opt("max_reach", max_reach, &key_p, &val_p, a, b);
  write_opt("jump_vel_reward", jump_vel_reward, &key_p, &val_p, a, b);
  write_opt("turn_off_constraints", turn_off_constraints, &key_p, &val_p, a, b);
  write_opt("omega_hip_maxabs", omega_hip_maxabs, &key_p, &val_p, a, b);
  write_opt("omega_knee_maxabs", omega_knee_maxabs, &key_p, &val_p, a, b);
  write_opt("constraint_penalty", constraint_penalty, &key_p, &val_p, a, b);
  write_opt("theta_hip_bounds_low", theta_hip_bounds_low, &key_p, &val_p, a, b);
  write_opt("theta_hip_bounds_high", theta_hip_bounds_high, &key_p, &val_p, a, b);
  write_opt("theta_knee_bounds_low", theta_knee_bounds_low, &key_p, &val_p, a, b);
  write_opt("theta_knee_bounds_high", theta_knee_bounds_high, &key_p, &val_p, a, b);
  write_opt("hip_minimum_z", hip_minimum_z, &key_p, &val_p, a, b);
  write_opt("knee_minimum_z", knee_minimum_z, &key_p, &val_p, a, b);
  write_opt("rpm_per_omega", rpm_per_omega, &key_p, &val_p, a, b);
  write_opt("fcn_k50_co_0", fcn_k50_co_0, &key_p, &val_p, a, b);
  write_opt("fcn_k50_co_1", fcn_k50_co_1, &key_p, &val_p, a, b);
  write_opt("fcn_k50_co_2", fcn_k50_co_2, &key_p, &val_p, a, b);
  write_opt("max_cmd_dyno", max_cmd_dyno, &key_p, &val_p, a, b);
  write_opt("rpm_for_max_cmd_dyno", rpm_for_max_cmd_dyno, &key_p, &val_p, a, b);
  write_opt("cmd_per_rpm_slope_dyno", cmd_per_rpm_slope_dyno, &key_p, &val_p, a, b);
  write_opt("model_coeff_dyno_0", model_coeff_dyno_0, &key_p, &val_p, a, b);
  write_opt("model_coeff_dyno_1", model_coeff_dyno_1, &key_p, &val_p, a, b);
  write_opt("model_coeff_dyno_2", model_coeff_dyno_2, &key_p, &val_p, a, b);
  write_opt("model_coeff_dyno_3", model_coeff_dyno_3, &key_p, &val_p, a, b);
  write_opt("model_coeff_dyno_4", model_coeff_dyno_4, &key_p, &val_p, a, b);
  write_opt("model_coeff_dyno_5", model_coeff_dyno_5, &key_p, &val_p, a, b);
  write_opt("contact_obs", contact_obs, &key_p, &val_p, a, b);
  write_opt("extra_obs_dim", extra_obs_dim, &key_p, &val_p, a, b);
  write_opt("filter_state", filter_state, &key_p, &val_p, a, b);
  write_opt("filter_action", filter_action, &key_p, &val_p, a, b);
  write_opt("action_is_delta", action_is_delta, &key_p, &val_p, a, b);
  write_opt("add_torque_penalty_on_air", add_torque_penalty_on_air, &key_p, &val_p, a, b);
  write_opt("add_omega_smoothness_reward", add_omega_smoothness_reward, &key_p, &val_p, a, b);
  write_opt("h1", h1, &key_p, &val_p, a, b);
  write_opt("h2", h2, &key_p, &val_p, a, b);
  write_opt("activation", activation, &key_p, &val_p, a, b);
  write_opt("obs_history_len_int", obs_history_len_int, &key_p, &val_p, a, b);
  write_opt("contact_obs_int", contact_obs_int, &key_p, &val_p, a, b);
  write_opt("jumping_obs_int", jumping_obs_int, &key_p, &val_p, a, b);
  write_opt("extra_obs_dim_int", extra_obs_dim_int, &key_p, &val_p, a, b);
  write_opt("inner_step_time", inner_step_time, &key_p, &val_p, a, b);
  write_opt("inner_loops_per_outer_loop", inner_loops_per_outer_loop, &key_p, &val_p, a, b);
  write_opt("physics_loops_per_inner_loop", physics_loops_per_inner_loop, &key_p, &val_p, a, b);
  write_opt("obs_dim", obs_dim, &key_p, &val_p, a, b);
  write_opt("act_dim", act_dim, &key_p, &val_p, a, b);
  write_opt("torque_delay_steps", torque_delay_steps, &key_p, &val_p, a, b);
  write_opt("observation_delay_steps", observation_delay_steps, &key_p, &val_p, a, b);
  write_opt("trq_dly_buflen", trq_dly_buflen, &key_p, &val_p, a, b);
  write_opt("obs_dly_buflen", obs_dly_buflen, &key_p, &val_p, a, b);
  write_opt("done_inner_steps", done_inner_steps, &key_p, &val_p, a, b);
  write_opt("neg_omega_hip_maxabs", neg_omega_hip_maxabs, &key_p, &val_p, a, b);
  write_opt("neg_omega_knee_maxabs", neg_omega_knee_maxabs, &key_p, &val_p, a, b);
  write_opt("neg_fcn_k50_co_0", neg_fcn_k50_co_0, &key_p, &val_p, a, b);
  write_opt("fc1_size", fc1_size, &key_p, &val_p, a, b);
  write_opt("fc2_size", fc2_size, &key_p, &val_p, a, b);
  write_opt("fc3_size", fc3_size, &key_p, &val_p, a, b);
  write_opt("h1_div_4", h1_div_4, &key_p, &val_p, a, b);
  write_opt("h2_div_4", h2_div_4, &key_p, &val_p, a, b);
  write_opt("mjstep1_after_mjstep", mjstep1_after_mjstep, &key_p, &val_p, a, b);
  write_opt("separate_mjstep1_mjstep2", separate_mjstep1_mjstep2, &key_p, &val_p, a, b);
  write_opt("delay_valid_obs", delay_valid_obs, &key_p, &val_p, a, b);
  write_opt("mjstep_order", mjstep_order, &key_p, &val_p, a, b);
  write_opt("rew_dly_buflen", rew_dly_buflen, &key_p, &val_p, a, b);
  write_opt("Shared_Obj", Shared_Obj, &key_p, &val_p, a, b);
  write_opt("SimIF_CPP", SimIF_CPP, &key_p, &val_p, a, b);
  write_opt("MlpIF_CPP", MlpIF_CPP, &key_p, &val_p, a, b);
  write_opt("Rollout_CPP", Rollout_CPP, &key_p, &val_p, a, b);
  write_opt("__MAINPROG__", __MAINPROG__, &key_p, &val_p, a, b);
  write_opt("do_mlp_output_tanh", do_mlp_output_tanh, &key_p, &val_p, a, b);
  write_opt("mlp_output_scaling", mlp_output_scaling, &key_p, &val_p, a, b);
  write_opt("actdim_div_2", actdim_div_2, &key_p, &val_p, a, b);
  write_opt("n_init_args_double", n_init_args_double, &key_p, &val_p, a, b);
  write_opt("n_init_args_int", n_init_args_int, &key_p, &val_p, a, b);
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
    init_args_double[0] = -PI * 50  / 180; // theta_hip
    init_args_double[1] = -PI * 100 / 180; // theta_knee
    init_args_double[2] = 0.0; // omega_hip
    init_args_double[3] = 0.0; // omega_knee
    init_args_double[4] = 0.4; // pos_slider
    init_args_double[5] = 0.0; // vel_slider
    init_args_double[6] = 3.0; // jumping_time
    init_args_int[0] = 0; // noise_index

    double* init_state;

    init_state = simiface.reset(init_args_double, init_args_int);

    #ifdef Debug_main
      std::cout << "  -> Done Resetting the SimInterface!" << std::endl;
      std::cout << "init_state[0] = " << init_state[0] << std::endl;
      std::cout << "init_state[1] = " << init_state[1] << std::endl;
      std::cout << "init_state[2] = " << init_state[2] << std::endl;
      std::cout << "init_state[3] = " << init_state[3] << std::endl;
      std::cout << "--------------------" << std::endl;
    #endif

    #ifdef Debug_main
      std::cout << "Step 3) Stepping the SimInterface" << std::endl;
      std::cout << "  --> Started Stepping the SimInterface!" << std::endl;
    #endif

    double action_raw[act_dim] = {-PI / 4, -PI * 100 / 180};
    double* next_state;
    double reward;
    bool done;
    double sim_time = gettm();
    for (int i=0; i < (outer_loop_rate*time_before_reset); i++)
      simiface.step(action_raw, &next_state, &reward, &done);
    sim_time = gettm() - sim_time;
    #ifdef Debug_main
      std::cout << "  --> Done Stepping the SimInterface!" << std::endl;
      std::cout << "  --> Simulation Time: " << sim_time << std::endl;
      std::cout << "next_state[0] = " << next_state[0] << std::endl;
      std::cout << "next_state[1] = " << next_state[1] << std::endl;
      std::cout << "next_state[2] = " << next_state[2] << std::endl;
      std::cout << "next_state[3] = " << next_state[3] << std::endl;
      std::cout << "--------------------" << std::endl;
    #endif

    return 0;
  }

#endif
