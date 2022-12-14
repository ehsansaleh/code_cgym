#include "mujoco.h"
#include "iostream"
#include <iomanip>    // std::setprecision, std::setw
#include <chrono>
#include <cstring>    // strcpy
#include "Rollout.hpp"
#include "defs.hpp"

#if xml_type == content_type
  #include "mj_xml.hpp"    // xml_bytes, xml_content
#endif


inline void addTo_act(double* src, double* dst){
  #if act_dim == 2
    dst[0] += src[0];
    dst[1] += src[1];
  #else
    mju_addTo(dst, src, act_dim);
  #endif
}

inline void copy_act(double* src, double* dst){
  #if act_dim == 2
    dst[0] = src[0];
    dst[1] = src[1];
  #else
    mju_copy(dst, src, act_dim);
  #endif
}

inline void copy_obs(double* src, double* dst){
  #if obs_dim == 4
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
  #else
    mju_copy(dst, src, obs_dim);
  #endif
}

Rollout::Rollout(void){
  // No need to do anything!
}

#define tiad (double*) &(init_args_double[traj_idx*n_init_args_double])
#define tiai (int*) &(init_args_int[traj_idx*n_init_args_int])

void Rollout::greedy_lite(int traj_num, int n_steps, double gamma,
                          double* eta_greedy, double* return_greedy,
                          int* done_steps){
  for (traj_idx=0; traj_idx < traj_num; traj_idx++){
    // Resetting the Sim Interface
    observation = simiface.reset(tiad, tiai);

    gamma_pow = 1;
    eta_greedy[traj_idx] = 0;
    return_greedy[traj_idx] = 0;
    for (step = 0; step < n_steps; step++){
      mlp_action = net.forward(observation);
      simiface.step(mlp_action, &observation, &reward, &done);
      eta_greedy[traj_idx] += gamma_pow * reward;
      return_greedy[traj_idx] += reward;
      gamma_pow *= gamma;
      if (done) break;
    }
    done_steps[traj_idx] = step + (int) done;
    // (int) done is added for the edge case of breaking out!
  }
}

void Rollout::vine(int traj_num, int n_steps, double gamma,
                   int expl_steps, int* reset_times, double* expl_noise,
                   double* obs_greedy, double* action_greedy, double* action_vine,
                   double* Q_greedy, double* eta_greedy, double* return_greedy,
                   double* Q_vine, double* eta_vine, double* return_vine,
                   int* done_steps, int* done_steps_vine, double* obs_greedy_all,
                   double* action_greedy_all, double* obs_vine_all,
                   double* action_vine_all, int* reset_times_out){
  int active_traj_len;
  int gen_idx_all, obs_idx_all, act_idx_all, step_capped;
  double *obs_all, *action_all;
  for (traj_idx=0; traj_idx < traj_num; traj_idx++){
    st_expl = reset_times[traj_idx];
    end_expl = st_expl + expl_steps;
    for (bool do_exploration : { false, true }){
      // Resetting the Sim Interface
      done = false;
      observation = simiface.reset(tiad, tiai);

      gamma_pow = 1;
      gamma_pow_q = 1;
      eta = (do_exploration) ? eta_vine : eta_greedy;
      return_ = (do_exploration) ? return_vine : return_greedy;
      Q = (do_exploration) ? Q_vine : Q_greedy;
      eta[traj_idx] = 0;
      return_[traj_idx] = 0;
      Q[traj_idx] = 0;

      gen_idx = traj_idx * expl_steps;
      obs_g_idx = gen_idx * obs_dim;
      act_idx  = gen_idx * act_dim;
      active_traj_len = 0;

      obs_all = (do_exploration) ? obs_vine_all : obs_greedy_all;
      action_all = (do_exploration) ? action_vine_all : action_greedy_all;
      gen_idx_all = traj_idx * n_steps;
      obs_idx_all = gen_idx_all * obs_dim;
      act_idx_all  = gen_idx_all * act_dim;

      for (step = 0; step < n_steps; step++){
        mlp_action = net.forward(observation);
        if ((step >= st_expl) && (step < end_expl)){
          if (do_exploration){
            addTo_act(expl_noise + act_idx, mlp_action);
            copy_act(mlp_action, action_vine + act_idx);
          } else {
            copy_obs(observation, obs_greedy + obs_g_idx);
            copy_act(mlp_action, action_greedy + act_idx);
            obs_g_idx += obs_dim;
          }
          act_idx += act_dim;
        }

        copy_obs(observation, obs_all + obs_idx_all);
        copy_act(mlp_action, action_all + act_idx_all);
        obs_idx_all += obs_dim;
        act_idx_all += act_dim;

        if (!done) {
          simiface.step(mlp_action, &observation, &reward, &done);
          active_traj_len = step + 1;
        } else {
          // Since the env is already done, let's pretend that the
          // observation is frozen, and we're taking the same action.
          reward = 0;
        }
        eta[traj_idx] += gamma_pow * reward;
        return_[traj_idx] += reward;
        gamma_pow *= gamma;
        if (step >= st_expl) {
          Q[traj_idx] += gamma_pow_q * reward;
          gamma_pow_q *= gamma;
        }
      }

      // just in case exploration isn't yet over when the for loop is over...
      while ((step >= st_expl) && (step < end_expl)){
        mlp_action = net.forward(observation);
        if ((step >= st_expl) && (step < end_expl)){
          if (do_exploration){
            addTo_act(expl_noise + act_idx, mlp_action);
            copy_act(mlp_action, action_vine + act_idx);
          } else {
            copy_obs(observation, obs_greedy + obs_g_idx);
            copy_act(mlp_action, action_greedy + act_idx);
            obs_g_idx += obs_dim;
          }
          act_idx += act_dim;
        }
        step++;
      }

      if (!do_exploration) {
        done_steps[traj_idx] = active_traj_len;
        if (active_traj_len <= st_expl) {
          st_expl = reset_times[traj_idx] % active_traj_len;
          end_expl = st_expl + expl_steps;
          for (step = 0; step < expl_steps; step++){
            step_capped = (st_expl + step >= n_steps - 1) ? (n_steps - 1) : (st_expl + step);
            copy_obs(obs_greedy_all + (traj_idx * n_steps + step_capped) * obs_dim,
              obs_greedy + (traj_idx * expl_steps + step) * obs_dim);
            copy_act(action_greedy_all + (traj_idx * n_steps + step_capped) * act_dim,
              action_greedy + (traj_idx * expl_steps + step) * act_dim);
          }
        }
        reset_times_out[traj_idx] = st_expl;
      } else {
        done_steps_vine[traj_idx] = active_traj_len;
      }
    }
  }
}

void Rollout::stochastic(int traj_num, int n_steps, double* expl_noise,
                         double* obs, double* action, double* rewards,
                         int* done_steps){
  int act_idx=0;
  int obs_idx=0;
  int gen_idx=0;
  double* rew_ptr;

  for (traj_idx=0; traj_idx < traj_num; traj_idx++){
    // Resetting the Sim Interface
    observation = simiface.reset(tiad, tiai);

    gen_idx = (traj_idx * n_steps);
    rew_ptr = rewards + gen_idx;
    for (step = 0; step < n_steps; step++){
      act_idx = gen_idx * act_dim;
      obs_idx = gen_idx * obs_dim;
      gen_idx++;

      mlp_action = action + act_idx;
      net.forward(observation, mlp_action);
      addTo_act(expl_noise + act_idx, mlp_action);
      copy_obs(observation, obs + obs_idx);

      simiface.step(mlp_action, &observation, rew_ptr, &done);
      rew_ptr++;
      if (done) break;
    }
    done_steps[traj_idx] = step + (int) done;
    // (int) done is added for the edge case of breaking out!
  }
}

void Rollout::reset(int traj_idx, double* obs){
  observation = simiface.reset(tiad, tiai);
  copy_obs(observation, obs);
}

#include <chrono>

void Rollout::partial_stochastic(int n_steps, double* expl_noise, double* obs,
                                 double* action, double* rewards, bool* dones){
  for (step = 0; step < n_steps; step++){
    mlp_action = action + step * act_dim;
    net.forward(observation, mlp_action);
    addTo_act(expl_noise + step * act_dim, mlp_action);
    copy_obs(observation, obs + step * obs_dim);
    simiface.step(mlp_action, &observation, rewards + step, &done); // 0.8998 sec for 14 trajs
    dones[step] = done;
    if (done) break;
  }
}

// These will be used by python's ctypes library for binding
extern "C"
{
  Rollout* rollout_new() {return new Rollout();}

  void rollout_set_simif_inits(Rollout* rollout, double* args_double, int* args_int){
    rollout->set_simif_inits(args_double, args_int);
  }

  void rollout_set_mlp_weights(Rollout* rollout, double* fc1, double* fc2, double* fc3,
    double* fc1_bias, double* fc2_bias, double* fc3_bias){
    rollout->set_mlp_weights(fc1, fc2, fc3, fc1_bias, fc2_bias, fc3_bias);
  }

  void rollout_greedy_lite(Rollout* rollout, int traj_num, int n_steps, double gamma,
    double* eta_greedy, double* return_greedy,
    int* done_steps) {
    rollout->greedy_lite(traj_num, n_steps, gamma,
      eta_greedy, return_greedy,
      done_steps);
  }

  void rollout_vine(Rollout* rollout, int traj_num, int n_steps, double gamma, int expl_steps,
    int* reset_times, double* expl_noise, double* obs_greedy, double* action_greedy, double* action_vine,
    double* Q_greedy, double* eta_greedy, double* return_greedy, double* Q_vine, double* eta_vine,
    double* return_vine, int* done_steps, int* done_steps_vine, double* obs_greedy_all,
    double* action_greedy_all, double* obs_vine_all, double* action_vine_all, int* reset_times_out) {
    rollout->vine(traj_num, n_steps, gamma, expl_steps, reset_times, expl_noise,
        obs_greedy, action_greedy, action_vine, Q_greedy, eta_greedy, return_greedy,
        Q_vine, eta_vine, return_vine, done_steps, done_steps_vine, obs_greedy_all,
        action_greedy_all, obs_vine_all, action_vine_all, reset_times_out);
  }

  void rollout_stochastic(Rollout* rollout, int traj_num, int n_steps,
    double* expl_noise, double* obs, double* action, double* rewards,
    int* done_steps){
    rollout->stochastic(traj_num, n_steps, expl_noise, obs,
      action, rewards, done_steps);
  }

  void rollout_infer_mlp(Rollout* rollout, int input_num,
    double* mlp_input, double* mlp_output) {
    rollout->infer_mlp(input_num, mlp_input, mlp_output);
  }

  void rollout_partial_stochastic(Rollout* rollout, int n_steps, double* expl_noise,
    double* obs, double* action, double* rewards, bool* dones) {
    rollout->partial_stochastic(n_steps, expl_noise, obs,
                                action, rewards, dones);
  }

  void rollout_reset(Rollout* rollout, int traj_idx, double* obs) {
    rollout->reset(traj_idx, obs);
  }
}


#if __MAINPROG__ == Rollout_CPP
  std::chrono::system_clock::time_point tm_start;
  mjtNum gettm(void)
  {
      std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - tm_start;
      return elapsed.count();
  }

  int main(int argc, char *argv[]){
    // Rollout Arguments
    const int traj_num = 1;
    const int n_steps = 8000;
    const double gamma = 0.99;

    // SimIF variables
    double init_args_double[traj_num*n_init_args_double];
    double init_args_int[traj_num*n_init_args_int];
    double* iad; int* iai;

    // MLP variables
    double fc1[obs_dim * h1] = {0};
    double fc2[h1      * h2] = {0};
    double fc3[h2 * act_dim] = {0};
    double fc1_bias[h1] = {0};
    double fc2_bias[h2] = {0};
    double fc3_bias[act_dim] = {-PI / 4, -PI * 100 / 180};

    // Output variables
    double eta_greedy[traj_num];
    double return_greedy[traj_num];
    int done_steps[traj_num];

    // Scratch Variables
    int i;
    double sim_time;

    // Initializing Our Arrays
    iad = init_args_double;
    iai = init_args_int;
    for (i=0; i<traj_num; i++) {
      *iad = -PI * 50  / 180; iad++; // theta_hip_inits[i]
      *iad = -PI * 100 / 180; iad++; // theta_knee_inits[i]
      *iad = 0; iad++; // omega_hip_inits[i]
      *iad = 0; iad++; // omega_knee_inits[i]
      *iad = 0.4; iad++;// pos_slider_inits[i]
      *iad = 0; iad++; // vel_slider_inits[i]
      *iad = 3; iad++; // jumping_time_inits[i]
      *iai = 0; iai++; // noise_index_inits[i]
    };

    #ifdef Debug_main
      std::cout << "Step 1) Creating the Rollout" << std::endl;
      std::cout << "  --> Started Creating a Rollout instance!" << std::endl;
    #endif

    Rollout rollout = Rollout();

    #ifdef Debug_main
      std::cout << "  --> Done Creating a Rollout instance!" << std::endl;
      std::cout << "--------------------" << std::endl;
    #endif


    #ifdef Debug_main
      std::cout << "Step 2) Initializing the Rollout" << std::endl;
      std::cout << "  --> Started Initializing the Rollout's Sim Interface!" << std::endl;
    #endif

    rollout.set_simif_inits(init_args_double, init_args_int);

    #ifdef Debug_main
      std::cout << "  --> Started Initializing the Rollout's MLP!" << std::endl;
    #endif

    rollout.set_mlp_weights(fc1, fc2, fc3, fc1_bias, fc2_bias, fc3_bias);

    #ifdef Debug_main
      std::cout << "  --> Done Initializing the Rollout!" << std::endl;
      std::cout << "--------------------" << std::endl;
    #endif

    #ifdef Debug_main
      std::cout << "Step 5) Starting the Greedy Simulations" << std::endl;
      std::cout << "  --> Started Rolling Trajectories!" << std::endl;
    #endif

    sim_time = gettm();
    rollout.greedy_lite(traj_num, n_steps, gamma,
                        eta_greedy, return_greedy,
                        done_steps);
    sim_time = gettm() - sim_time;

    #ifdef Debug_main
      std::cout << "  --> Done  Rolling Trajectories!" << std::endl;
      std::cout << "  --> Full Simulation Time: " << sim_time << std::endl;
      std::cout << "--------------------" << std::endl;

      std::cout << std::fixed << std::setprecision(8);
      std::cout << " The Discounted Payoff Values Are:     " << std::endl;
      for (i = 0; i < traj_num; i++)
        std::cout << eta_greedy[i] << ", ";
      std::cout << std::endl;

      std::cout << " The Non-discounted Payoff Values Are: " << std::endl;
      for (i = 0; i < traj_num; i++)
        std::cout << return_greedy[i] << ", ";
      std::cout << std::endl;

      std::cout << " The Trajectory Lengths Are:           " << std::endl;
      for (i = 0; i < traj_num; i++)
        std::cout << done_steps[i] << ", ";
      std::cout << std::endl;
    #endif

    return 0;
  }

#endif
