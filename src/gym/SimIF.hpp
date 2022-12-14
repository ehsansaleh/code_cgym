#include "defs.hpp"

/////////////////////////////////////////////////////
//////////////// Utility Functions //////////////////
/////////////////////////////////////////////////////


/////////////////////////////////////////////////////
////////// RewardGiver Class Definitions ////////////
/////////////////////////////////////////////////////

class SimInterface;

class RewardGiver {
  public:
    RewardGiver();                           // This is the constructor
    //~RewardGiver();                        // This is the destructor
    void reset(double obs_current[obs_dim],  // This should be current, since we'll be applying a
                                             // reward delay (i.e., the same as observation delay)
               mjModel* mj_model,
               mjData* mj_data,
               bool is_mj_stable,
               SimInterface* simintf);               // re-initialize the reward giver state

    void update_reward(double obs_current[obs_dim],  // This should be current, since we'll be applying a
                                                     // reward delay (i.e., the same as observation delay)
                       mjModel* mj_model,
                       mjData* mj_data,
                       bool is_mj_stable,
                       double* reward);

    void update_done(double obs_current[obs_dim],  // This should be current, since we'll be applying a
                                                   // reward delay (i.e., the same as observation delay)
                     mjModel* mj_model,
                     mjData* mj_data,
                     bool is_mj_stable,
                     bool* done);

  private:
    SimInterface* simif;

    #if ((ENVNAME == HalfCheetah) || (ENVNAME == Swimmer)               || \
      (ENVNAME == Hopper) || (ENVNAME == Walker2d) || (ENVNAME == Ant)  || \
      (ENVNAME == Humanoid))
      double x_position_before;
    #endif
    #if (ENVNAME == Ant)
      bool do_compute_mjids;
    #endif
    #if (ENVNAME == Ant)
      int torso_body_mjid;
    #endif
    #if (ENVNAME == Reacher)
      double vec_norm_before;
    #endif
    #if (ENVNAME == Humanoid) || (ENVNAME == HumanoidStandup)
      double contact_cost;
    #endif
    #if (ENVNAME == Humanoid)
      double x_position_after;
    #endif
};

/////////////////////////////////////////////////////
////////// SimInterface Class Definition ////////////
/////////////////////////////////////////////////////

class SimInterface {
  public:
    SimInterface();                 // This is the constructor
    ~SimInterface();                // This is the destructor

    double* reset(double* init_args_double, int* init_args_int);     // resets the robot to a particular state.

    double* add_trq_buff(double new_trqs[]);
    double* add_obs_buff(double new_obs[]);
    double  add_rew_buff(double new_rew);
    bool add_done_buff(bool new_done);
    void update_mj_obs();
    void step_inner(double action_raw[]);
    void step(double action_raw[act_dim],
              double** next_obs,
              double* reward,
              bool* done);

    // step counts
    int inner_step_count;
    int outer_step_count;

  private:
    // mujoco elements
    mjModel* mj_model;
    mjData* mj_data;

    // The raw observations from mujoco (i.e., non-delayed)
    double mj_obs[obs_dim];

    // observation delay buffer items
    double obs_delay_buff[obs_dly_buflen][obs_dim];
    bool obs_dlybuf_ever_pushed;
    int obs_dlybuf_push_idx;

    // torque delay buffer items
    double trq_delay_buff[trq_dly_buflen][act_dim];
    bool trq_dlybuf_ever_pushed;
    int trq_dlybuf_push_idx;

    // reward delay buffer items
    double rew_delay_buff[rew_dly_buflen];
    bool rew_dlybuf_ever_pushed;
    int rew_dlybuf_push_idx;

    bool done_delay_buff[done_dly_buflen];
    bool done_dlybuf_ever_pushed;
    int done_dlybuf_push_idx;

    // The final torque values that will be sent to mujoco's controls
    double joint_torque_capped[act_dim];

    bool is_mj_stable;

    // The reward giver object
    RewardGiver rew_giver;

    #if action_type == torque
      // No joint_torque_command variable creation is necessary
    #elif action_type == jointspace_pd
      double joint_torque_command[act_dim];
    #elif action_type == workspace_pd
      #error "you possibly need some joint_torque_command definition here."
    #else
      #error "action_type not implemented."
    #endif

    #if do_obs_noise == True
      double non_noisy_observation[obs_dim];
      #error "you should define the noise related variables here."
    #endif

    #if (ENVNAME == Reacher)
      int fingertip_body_mjid;
      int target_body_mjid;
    #endif

    #if defined(sparse_rew_outer_steps) && (sparse_rew_outer_steps > 1)
      double sparse_rew_accumlator;
      double sparse_rew_discount;
    #endif
};

void write_opt(const std::string key_str, double val, char** key_write_ptr,
               double** val_write_ptr, char* key_write_ptr_max,
               double* val_write_ptr_max);

void get_build_options(char* keys, double* vals, int keys_len, int vals_len);
