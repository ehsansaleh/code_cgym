#pragma once

#include "mujoco.h"
#include "MlpIF.hpp"
#include "defs.hpp"
#include "SimIF.hpp"

class Rollout{
  public:
    Rollout();                         // This is the constructor
    //~Rollout();                      // This is the destructor

    void set_simif_inits(double* args_double, int* args_int){
      init_args_double = args_double;
      init_args_int = args_int;
    };

    void set_mlp_weights(double* fc1, double* fc2, double* fc3,
      double* fc1_bias, double* fc2_bias, double* fc3_bias){
      net.set_weights(fc1, fc2, fc3, fc1_bias, fc2_bias, fc3_bias);
    };

    void greedy_lite(int traj_num, int n_steps, double gamma,
                     double* eta_greedy, double* return_greedy,
                     int* done_steps);

    void vine(int traj_num, int n_steps, double gamma,
              int expl_steps, int* reset_times, double* expl_noise,
              double* obs_greedy, double* action_greedy, double* action_vine,
              double* Q_greedy, double* eta_greedy, double* return_greedy,
              double* Q_vine, double* eta_vine, double* return_vine,
              int* done_steps, int* done_steps_vine, double* obs_greedy_all,
              double* action_greedy_all, double* obs_vine_all,
              double* action_vine_all, int* reset_times_out);

    void stochastic(int traj_num, int n_steps, double* expl_noise,
                    double* obs, double* action, double* rewards,
                    int* done_steps);

    void reset(int traj_idx, double* obs);
    void partial_stochastic(int n_steps, double* expl_noise, double* obs,
                            double* action, double* rewards, bool* dones);

    void infer_mlp(int input_num, double* mlp_input, double* mlp_output) {
      for (int i = 0; i < input_num; i++){
        mlp_action = net.forward(mlp_input + obs_dim * i);
        mju_copy(mlp_output + act_dim * i, mlp_action, act_dim);
      }
    }

  private:
    SimInterface simiface;
    MLP3 net;

    // Sim Interface Initialization Arguments (Arrays)
    double* init_args_double;
    int* init_args_int;

    // A bunch of scratch variables
    int traj_idx, step;
    double* mlp_action;
    double* observation;
    double reward;
    bool done;
    double gamma_pow;

    // vine_lite scratch Variables
    double* eta;
    double* return_;
    double* Q;
    int gen_idx;
    int obs_g_idx;
    int act_idx;
    int st_expl;
    int end_expl;
    double gamma_pow_q;
};
