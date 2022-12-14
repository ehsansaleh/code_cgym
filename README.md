# CGym

CGym is a fast C++ implementation of [OpenAI's Gym](https://github.com/openai/gym) interface. The current way of rollout collection in RL libraries requires a back and forth travel between an external simulator (e.g., Mujoco) and the python RL code for generating the next actions for every time-step. This interface overhead leaves a lot of performance on the table. This is particularly inefficient when the simulator is fast and a lot of samples can be collected quickly but the interface is bottlenecking this scaling.

This project collects the policy parameters at the beginning, and rolls out the entire trajectory in C++ (i.e., even the action generation happens in C++). The python RL code only gets an array of collected samples after all the time-steps are simulated. This reduces the interfacing overhead and makes the sample collection much faster.

The following table shows the speed-ups obtained by CGym.

<div align="center">
<table><thead><tr><th rowspan="2">Environment</th><th colspan="2">Rollout Time (ms)</th><th rowspan="2">Speed-Up</th></tr><tr><th>CGym (C++)</th><th>Gym (Python)</th></tr></thead><tbody><tr><td>Pendulum</td><td>0.4</td><td>33.72</td><td>84.3x</td></tr><tr><td>NLR-Pendulum</td><td>0.41</td><td>29.83</td><td>72.8x</td></tr><tr><td>InvertedPendulum</td><td>0.44</td><td>2.77</td><td>6.3x</td></tr><tr><td>InvertedDoublePendulum</td><td>0.58</td><td>2.32</td><td>4x</td></tr><tr><td>HalfCheetah</td><td>40.49</td><td>161.45</td><td>4x</td></tr><tr><td>Swimmer</td><td>35.35</td><td>148.33</td><td>4.2x</td></tr><tr><td>Hopper</td><td>10.73</td><td>32.78</td><td>3.1x</td></tr><tr><td>Walker2d</td><td>4.47</td><td>10.9</td><td>2.4x</td></tr><tr><td>Ant</td><td>280.17</td><td>506.92</td><td>1.8x</td></tr><tr><td>Humanoid</td><td>25.65</td><td>39.97</td><td>1.6x</td></tr><tr><td>HumanoidStandup</td><td>674.19</td><td>878.94</td><td>1.3x</td></tr></tbody></table>
</div>

# Usage

First, add the following line to your `~/.bashrc` file:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cgym/lib
```
Make sure you replace `/path/to/cgym/lib` with the proper value.

Now, run
```
make
python usr/binding.py
```

Here is an example output for the pendulum environment:
```
Starting 10 C++ Pendulum Rollouts
Finished 10 C++ Pendulum Rollouts
Total Time: 0.00404 sec
--------------------
Starting 10 Python Pendulum Rollouts
Finished 10 Python Pendulum Rollouts
Total Time: 0.33718 sec
--------------------
Maximum       Observation Difference: 0.0000000000
Maximum       Action      Difference: 0.0000000000
Maximum       Reward      Difference: 0.0000000000
Mean Absolute Observation Difference: 0.0000000000
Mean Absolute Action      Difference: 0.0000000000
Mean Absolute Reward      Difference: 0.0000000000
```

### Options and Notes
* You can modify the simulation interface options to your likings in the `src/defs.hpp` file (look for the `SimIF Options Defs` section).
* The `src/leg.xml` file is the file that will be embedded into the shared object for Mujoco's compiled (i.e., this file will be read and baked in at the compilation time, and will not be read at the run-time).
* The `make` call will recreate the `bin/libRollout.so` shared object, which is the only binary the python code uses.
* The python binding happens in the `usr/binding.py` file. The `LegCRoller` Class is supposed to be a full environment for use in RL methods.

*Why Is the `bashrc export` Necessary?*
  * This is necessary for python to find and link the C++ libraries. Note that by the time python has started, the `LD_LIBRARY_PATH` is already loaded into python's spawned linker. Therefore, changing the environmental variable inside python using `os.environ['LD_LIBRARY_PATH'] += ...` would not have any effect on the already loaded linker.

## Compiler Issues
**Compiler Error**: If you happen to get the following error
```
g++: error: unrecognized command line option '-faligned-new'
```
it is because you're running an old `gcc` compiler. It's best if you use `gcc` version `7.2.0` or newer, which can compile a more efficient object file with better memory alignments.

**Linking Error**: If you encounter the following error:
```
/usr/bin/ld: warning: libxxx.so.6, needed by /a/b/c/libyyy.so, not found (try using -rpath or -rpath-link)
```
You may actually need to follow this [Stack Overflow](https://stackoverflow.com/questions/13507600/get-rid-of-gcc-usr-bin-ld-warning-lib-not-found) advice:

*You need to add the dynamic library equivalent of -L:*

```
-Wl,-rpath-link,/path/to/lib
```

This will cause the linker to look for shared libraries in non-standard places, but only for the purpose of verifying the link is correct.

If you want the program to find the library at that location at run-time, then there's a similar option to do that:
```
-Wl,-rpath,/path/to/lib
```
But, if your program runs fine without this then you don't need it.

# Resolved Mujoco Issues
The following are a few problems which were encountered, diagnosed, and then resolved in the development of CGym. I am only documenting these problems to keep a reference.
#### The Invalid State Problem of Mujoco
**Problem Description**: After calling Mujoco's `mj_step` some of the model attributes do not get updated. For instance, while the `qpos` and `qvel` values are updated after an `mj_step` call, the foot position (i.e., `mj_data->site_xpos[3*foot_center_site_mjid]`) is not updated. This can make the results of the python wrapper different from the C++ implementation.

**Problem Discovery**: I noticed this by observing that the `foot_x` and other reward terms are delayed one step from the `theta` and `omega` reward terms even when I extracted all the raw values (i.e., `theta`, `omega`, `foot_pos`, `knee_pos`, etc.) from `mj_data` at the same time:
```
#define theta_hip_oc   mj_data->qpos[mj_model->jnt_qposadr[hip_mjid]]
#define theta_knee_oc  mj_data->qpos[mj_model->jnt_qposadr[knee_mjid]]
#define omega_hip_oc   mj_data->qvel[mj_model->jnt_dofadr[hip_mjid]]
#define omega_knee_oc  mj_data->qvel[mj_model->jnt_dofadr[knee_mjid]]
#define foot_pos_x     mj_data->site_xpos[3*foot_center_site_mjid]
#define hip_pos_x      mj_data->site_xpos[3*hip_center_site_mjid]
#define knee_pos_x     mj_data->site_xpos[3*knee_center_site_mjid]
#define foot_pos_z     mj_data->site_xpos[3*foot_center_site_mjid+2]
#define hip_pos_z      mj_data->site_xpos[3*hip_center_site_mjid+2]
#define knee_pos_z     mj_data->site_xpos[3*knee_center_site_mjid+2]
```
While `theta_hip_oc` and `omega_hip_oc` were in sync with the `dm_control` implementation, the `foot_pos_x` and `foot_pos_z` values were behind one step.

**Diagnosis**: Consider setting the initial state using the following assignments:
```
// Setting qpos elements
mj_data->qpos[mj_model->jnt_qposadr[slider_mjid]] = pos_slider;
mj_data->qpos[mj_model->jnt_qposadr[hip_mjid]]    = theta_hip;
mj_data->qpos[mj_model->jnt_qposadr[knee_mjid]]   = theta_knee;
// Setting qvel elements
mj_data->qvel[mj_model->jnt_dofadr[slider_mjid]]  = vel_slider;
mj_data->qvel[mj_model->jnt_dofadr[hip_mjid]]     = omega_hip;
mj_data->qvel[mj_model->jnt_dofadr[knee_mjid]]    = omega_knee;
```
Once you do these, `mujoco` does not even know that these values were updated so that it updates the other model attributes that are a function of these attributes (e.g., the physical contact states, reaction forces, etc.). Therefore, even if you don't want your model to progress through time, you're supposed to call `mj_forward` to make sure that every attribute gets recomputed and the model would be in a *valid* state.

The aforementioned issue is somewhat related to the necessity of the `mj_forward` call after initialization; some of the `mujoco`'s `mj_data` attributes could be left outdated even after you call `mj_step`! To make sure that all of the `mj_data` variables are in a *valid* state you need to call `mj_step1` after the `mj_data` call. The `dm_control` folks also admitted this issue in their comments at

https://github.com/deepmind/dm_control/blob/a669634a9bdd5be5d78654b2370f9ef8fd987817/dm_control/mujoco/engine.py#L154

Here are the `dm_control` comments I am referring to in their step function definition:
```
def step(self):
  """Advances physics with up-to-date position and velocity dependent fields.
  The actuation can be updated by calling the `set_control` function first.
  """
  # In the case of Euler integration we assume mj_step1 has already been
  # called for this state, finish the step with mj_step2 and then update all
  # position and velocity related fields with mj_step1. This ensures that
  # (most of) mjData is in sync with qpos and qvel. In the case of non-Euler
  # integrators (e.g. RK4) an additional mj_step1 must be called after the
  # last mj_step to ensure mjData syncing.
  with self.check_invalid_state():
    if self.model.opt.integrator == enums.mjtIntegrator.mjINT_EULER:
      mjlib.mj_step2(self.model.ptr, self.data.ptr)
    else:
      mjlib.mj_step(self.model.ptr, self.data.ptr)

    mjlib.mj_step1(self.model.ptr, self.data.ptr)
```
This issue is also validated by `mujoco`'s documentation at

http://www.mujoco.org/book/reference.html

Here are the specific comments I am referring to:
```
Main simulation

These are the main entry points to the simulator. Most users will
only need to call mj_step, which computes everything and advanced the
simulation state by one time step. Controls and applied forces must
either be set in advance (in mjData.ctrl, qfrc_applied and
xfrc_applied), or a control callback mjcb_control must be installed
which will be called just before the controls and applied forces are
needed. Alternatively, one can use mj_step1 and mj_step2 which break
down the simulation pipeline into computations that are executed
before and after the controls are needed; in this way one can set
controls that depend on the results from mj_step1. Keep in mind
though that the RK4 solver does not work with mj_step1/2.


mj_forward performs the same computations as mj_step but without the
integration. It is useful after loading or resetting a model (to put the
entire mjData in a valid state), and also for out-of-order computations
that involve sampling or finite-difference approximations.
```

**Issue**: Although calling `mj_step1` after `mj_step` solves the invalid state issue, it is somewhat excessive; `mj_step1` is already included within `mj_step`, and calling it after `mj_step` is excessive since it would be called again in the next time-step. To avoid this inefficiency, we could call `mj_step1` and `mj_step2` separately, and evaluate the reward between the two calls (which is when the model is in a valid state). However, `mujoco`'s documentation advises against this when using the `RK4` solver. I'm not sure why this restriction exists. I tried calling `mj_step1` and `mj_step2` separately even when using the `RK4` solver, and it seemed to work just fine. Therefore, I included this as an option in the solution options.

Here is a post from the mujoco forums:

http://www.mujoco.org/forum/index.php?threads/continuous-time-nonlinear-control.3436/

Emo Todorov seems to suggest that calling `mj_step1` and `mj_step2` separately should be fine even when using the `RK4` solver as long as you do not change the controls between `mj_step1` and `mj_step2`. This apprach is implemented with the following definition in our C++ code:
```
#define mjstep_order separate_mjstep1_mjstep2
```
We do not need a completely valid `mj_data` for feeding the agent at this moment, since our observation variables (i.e., `theta` and `omega` values) are already in valid states before calling `mj_step1`. However, if we hypothetically wanted to (1) include the foot force as an observation to the agent, and (2) use the `RK4` integrator, then we could have problems when placing the control assignment between `mj_step1` and `mj_step2`. Otherwise, we should be just fine in all other situations (e.g., when using the Euler integrator).

**Solution**: There is a C++ definition named `mjstep_order`, and it can be set to either of three values: (1) `mjstep1_after_mjstep`, (2) `separate_mjstep1_mjstep2`, and (3) `delay_valid_obs`. The default value is set to the `separate_mjstep1_mjstep2` since it is efficient and safe.

Generally speaking:
  1. `mjstep1_after_mjstep` option will call `mj_step1` after a full `mj_step` to validate all variables. While this is the approach taken by `dm_control` and solves all invalidation problems, it can be inefficient; `mj_step` is calling the two `mj_step1` and `mj_step2` functions inside itself, and the third `mj_step1` call is extra. This inefficiency can make the simulation twice as expensive.

  2. `separate_mjstep1_mjstep2` will call `mj_step1` and `mj_step2` separately, and places the reward calculation in between them where all the physical variables are validated. The only caveat is that the `mj_data` control variables should not change between the `mj_step1` and `mj_step2` calls if the `RK4` integrator is being used. Currently, we don't even need to do this, since our observations only consist of valid variables like `theta` and `omega`. Besides, the default mujoco integrator is Euler, which doesn't care if your change the control variables in between the `mj_step1` and `mj_step2` calls.

  3. `delay_valid_obs` is a bit more nuanced option, and I strongly advise against using it if you don't exactly need it and know what you are doing. Basically, the idea is to extract the *valid* observations and store them in some "last step" variables in order to delay them and having the delayed valid observations be in sync with the non-delayed invalid observations. While this could theoretically work, I didn't completely implement it; in the `SimplifiedRewardGiver` definition, you need the `foot_force_z` variable which is in a valid state since we compute it freshly. If you want to use the `delay_valid_obs` mode, you need to create a `last_foot_force_z` array, and store a 1-step delayed version of `foot_force_z` in it.

  Here is a partial list of variables that are in a *valid* state before calling `mj_step`:
    1. `theta_*`
    2. `omega_*`
    3. `foot_force_*`
    4. `has_touched_ground`

  Here is a partial list of variables that are in an *invalid* state before calling `mj_step`:
    1. `foot_pos_*`
    2. `hip_pos_*`
    3. `knee_pos_*`

  The `theta_*` and `omega_*` variables are already buffered by the `SimInterface` in the `mj_obs` variable, and are passed to the `RewardGiver`'s step function. `foot_force_*` and `has_touched_ground` get computed freshly by calling the `_get_contact_state` function inside the `RewardGiver`, and that's possibly why they're in a *valid* state and need to be delayed.

I will leave the inner workings and pros and cons of these options to be described as comments in the code.

#### The `mj_data` Reset Issue

**Description**: I found a case where if the `SimIF` ran 4 consecutive trajectories from the exact same initial state (i.e., theta, omega, etc.), the first trajectory's payoff would be 3.8 units smaller than the other 3 trajectories! After further investigation, I figured that while all theta and omega values were identical among all trajectories, there is a single step in the trajectory where the `force_foot_z` reward term is different.

Here is the outer step 872 from the first trajectory:
```
Step outer 872:
  Step inner 872:
    0) theta_dlyd           = -0.84037967, -1.74509046
       omega_dlyd           = 0.12058722, 0.00074280
    1) tau                  = 0.12539505, -0.00522145
    2) joint_torque_current = 0.12566718, -0.00520711
    3) joint_torque         = 0.12574479, -0.00520302
    4) joint_torque_capped  = 0.10059583, -0.00416242
  RStep 875:
    -->    theta            = -0.84025927, -1.74508972
    -->    omega            = 0.12029173, 0.00074194
    -->    theta_input      = -0.84025927, -1.74508972
    -->    omega_input      = 0.12029173, 0.00074194
    -->    foot_pos_x,z     = -0.02547646, 0.00458399
    -->    knee_pos_x,z     = 0.09341776, 0.07850398
    R1) Reward['main']      = -0.22915450
    R2) Reward['omega']     = -0.00968269
    R3) Reward['foot_x']    = -0.25476459
    R4) Reward['foot_f_z']  = -3.80000000
        foot_force_z        = 0.00000000
    R5) Reward['foot_z']    = -0.04583993
    R6) Reward['knee_z']    = -0.32244035
    R*) Total Reward        = -4.66188206
Reward: -0.49387166
Done:   0
```

However, the corresponding step in all other trajectories is the following:

```
Step outer 872:
  Step inner 872:
    0) theta_dlyd           = -0.84037967, -1.74509046
       omega_dlyd           = 0.12058722, 0.00074280
    1) tau                  = 0.12539505, -0.00522145
    2) joint_torque_current = 0.12566718, -0.00520711
    3) joint_torque         = 0.12574479, -0.00520302
    4) joint_torque_capped  = 0.10059583, -0.00416242
  RStep 875:
    -->    theta            = -0.84025927, -1.74508972
    -->    omega            = 0.12029173, 0.00074194
    -->    theta_input      = -0.84025927, -1.74508972
    -->    omega_input      = 0.12029173, 0.00074194
    -->    foot_pos_x,z     = -0.02547646, 0.00458399
    -->    knee_pos_x,z     = 0.09341776, 0.07850398
    R1) Reward['main']      = -0.22915450
    R2) Reward['omega']     = -0.00968269
    R3) Reward['foot_x']    = -0.25476459
    R4) Reward['foot_f_z']  = 0.00000000
        foot_force_z        = 31.96727837
    R5) Reward['foot_z']    = -0.04583993
    R6) Reward['knee_z']    = -0.32244035
    R*) Total Reward        = -0.86188206
Reward: -0.49387166
Done:   0
```

Notice, how `foot_force_z` is zero in the first trajectory while it is `31.96727837` in all other trajectories. Since `Reward['foot_z']` is not zero, it means that the robot has touched the ground (i.e., `has_touched_ground` is correctly set to be true). However, somehow in the first trajectory `foot_force_z` is zero although the foot has touched the ground, while the rest of trajectories are showing more reasonable numbers.

`foot_force` is being computed by calling `mj_contactForce` when there is a foot contact in `SimIF.cpp`'s `_get_contact_state` function call. While I couldn't exactly understand how `mj_contactForce` works, I made a decompilation attempt. Here is the original decompiled functionality of `mj_contactForce`:

```
void mj_contactForce(long param_1,long param_2,int param_3,undefined8 *param_4){
  bool bVar1;
  undefined7 extraout_var;
  long lVar2;

  mju_zero(param_4,6);
  if (((-1 < param_3) && (param_3 < *(int *)(param_2 + 0x9d8c))) &&
     (lVar2 = (long)param_3 * 0x210 + *(long *)(param_2 + 0x9f30), -1 < *(int *)(lVar2 + 0x208))) {
    bVar1 = mj_isPyramidal(param_1);
    if ((int)CONCAT71(extraout_var,bVar1) == 0) {
      mju_copy(param_4,(void *)(*(long *)(param_2 + 0xa068) + (long)*(int *)(lVar2 + 0x208) * 8),
               *(int *)(lVar2 + 0x1f8));
      return;
    }
                    /* WARNING: Treating indirect jump as call */
    mju_decodePyramid(param_4,(undefined8 *)
                              (*(long *)(param_2 + 0xa068) + (long)*(int *)(lVar2 + 0x208) * 8),
                      lVar2 + 0x70,*(int *)(lVar2 + 0x1f8));
    return;
  }
  return;
}
```

This is my attempt at understanding the decompiled code:
```
void mj_contactForce(const mjModel* mj_model, const mjData* mj_data, int id, mjtNum* result){
  bool is_pyr;

  mju_zero(result,6);
  if (( (-1 < id) && (id < mj_data->ncon ) &&
      (-1 < mj_data->contact[id].efc_address)
     ) {
    is_pyr = mj_isPyramidal(mj_model);
    if (is_pyr) {
      mju_copy(result, &(mj_data->efc_force[mj_data->contact[id].efc_address]),
               mj_data->contact[id].dim);
      return;
    }
    mju_decodePyramid(result, &(mj_data->efc_force[mj_data->contact[id].efc_address]),
                      &(mj_data->contact[id].mu), mj_data->contact[id].dim);
    return;
  }
  else
    return;
}
```
I found out that
  1. the function is entering the if condition `(( (-1 < id) && (id < mj_data->ncon ) && ...` in the problematic outer step 872,
  2. `is_pyr` is being set to true, and therefore the function makes the `mju_decodePyramid` call, and
  3. `mju_decodePyramid` is zeroing `foot_force` (i.e., `result`) in the first trajectory.

**Mitigation**: Clearly, there is a variable that's being changed in the first trajectory upon which `mju_decodePyramid`'s output depends. To make all trajectories are consistent, I added an `mj_data` reset line to the Sim Interface's reset function as follows:
```
mj_resetData(mj_model, mj_data);
```
This made all trajectories have `foot_force_z = 0.0` at this problematic step! This seems like a bug in mujoco, and I had to stop looking here.

**Example Output**: Open the directory `opt/stdouts/3_mjdata_reset`. You will find a snapshot of the code in the `code` sub-directory.
  * The file `roll_out_without_mjdata_reset.txt` contains debugging output when the `mj_resetData(mj_model, mj_data);` line is commented out from `SimIF.cpp`. Search for `RStep 875:` in the file, and you'll find 4 instances. The first instance will report zero `foot_force_z` value, while the rest will report non-zero `foot_force_z`.
  * The file `roll_out_with_mjdata_reset.txt` is a similar file, however, the  `mj_resetData(mj_model, mj_data);` line is enabled. As you can check, `foot_force_z` will be zero in all 4 instances of `RStep 875:`.

You should be able to get the same output with running the snapshot code and running `make roll` in the terminal. You can comment/uncomment the `mj_resetData(mj_model, mj_data);` line in the `SimIF.cpp`'s line 604.
