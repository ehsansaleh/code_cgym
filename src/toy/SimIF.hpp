#include "defs.hpp"

/////////////////////////////////////////////////////
////////// SimInterface Class Definition ////////////
/////////////////////////////////////////////////////

class SimInterface {
  public:
    SimInterface();                 // This is the constructor
    ~SimInterface();                // This is the destructor

    double* reset(double* init_args_double, int* init_args_int);     // resets the robot to a particular state.

    void step(double action_raw[act_dim],
              double** next_obs,
              double* reward,
              bool* done);

    int step_count;

  private:
    double state;
};

void write_opt(const std::string key_str, double val, char** key_write_ptr,
               double** val_write_ptr, char* key_write_ptr_max,
               double* val_write_ptr_max);

void get_build_options(char* keys, double* vals, int keys_len, int vals_len);
