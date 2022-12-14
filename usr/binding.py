import numpy as np
import ctypes
from ctypes import c_int, c_double, c_void_p, c_char_p
from ctypes import create_string_buffer
import time
from collections import namedtuple
import json
import os.path as osp
from os.path import abspath, dirname, exists, realpath, basename
import filecmp
import os

import importlib
has_gym = importlib.util.find_spec('gym') is not None
if has_gym:
    import gym
    from gym.spaces.space import Space
    from gym import spaces
    from gym.utils import seeding

#############################################################
####################### Binding Types #######################
#############################################################

cnp_double_p = np.ctypeslib.ndpointer(dtype=np.float64)
cnp_int_p = np.ctypeslib.ndpointer(dtype=np.int32)
cnp_bool_p = np.ctypeslib.ndpointer(dtype=np.bool)

# See
#   1. https://docs.python.org/3/library/ctypes.html
#   2. https://nesi.github.io/perf-training/python-scatter/ctypes
# if you needed to add more types to the following translation table
c2py_type = {'double*': cnp_double_p, 'int*': cnp_int_p, 'void*':c_void_p,
             'int': c_int, 'double':c_double, 'void':c_void_p,
             'Rollout*':c_void_p, 'char*':c_char_p, 'bool*': cnp_bool_p}

#############################################################
################# Binding Utility Functions #################
#############################################################

CGYMDIR = dirname(dirname(abspath(__file__)))
DFLTXMLDIR = osp.join(CGYMDIR, 'src', 'xml')

def describe_cfunc(d_str):
    a = d_str.index('(')
    b = d_str.rindex(')')
    c = d_str[a+1:b]
    e = [x.strip() for x in c.split(',')]
    e = [x for x in e if x!='']
    arg_types_c = [''.join(x.split(' ')[:-1]) for x in e]
    arg_names_c = [x.split(' ')[1] for x in e]

    f = d_str[:a]
    res_type_c = ''.join(f.split(' ')[:-1])
    func_name = ''.join(f.split(' ')[-1])
    out_dict = dict(func_name=func_name, res_type_c=res_type_c,
                    arg_types_c=arg_types_c, arg_names_c=arg_names_c)
    return out_dict

def pyarg_type(c_description):
    arg_types_c = c_description['arg_types_c']
    res_type_c = c_description['res_type_c']
    arg_py_ctypes = [c2py_type[x] for x in arg_types_c]
    return arg_py_ctypes

def set_lib_types(lib, c_description, arg_ctypes, res_ctype):
    func_name = c_description['func_name']
    assert hasattr(lib, func_name)
    pyfunc = getattr(lib, func_name)
    pyfunc.argtypes = arg_ctypes
    pyfunc.restype = res_ctype

def check_args(c_description, args):
    arg_types_c = c_description['arg_types_c']
    for i, (arg, ctype) in enumerate(zip(args, arg_types_c)):
        if ctype == 'double*':
            assert isinstance(arg, np.ndarray), f'argument {i} has a C++ {ctype} type, and must be np.ndarray.'
            assert arg.dtype == np.float64, f'argument {i} has a C++ {ctype} type, and must have dtype == np.float64'
            assert arg.flags['BEHAVED'], f'argument {i} has a C++ {ctype} type, and must be a BEHAVED numpy array'
            assert arg.flags['C_CONTIGUOUS'], f'argument {i} has a C++ {ctype} type, and must be a C_CONTIGUOUS numpy array'
            assert arg.flags['CARRAY'], f'argument {i} has a C++ {ctype} type, and must be a CARRAY numpy array'
        elif ctype == 'int*':
            assert isinstance(arg, np.ndarray), f'argument {i} has a C++ {ctype} type, and must be np.ndarray.'
            assert arg.dtype == np.int32, f'argument {i} has a C++ {ctype} type, and must have dtype == np.int32'
            assert arg.flags['BEHAVED'], f'argument {i} has a C++ {ctype} type, and must be a BEHAVED numpy array'
            assert arg.flags['C_CONTIGUOUS'], f'argument {i} has a C++ {ctype} type, and must be a C_CONTIGUOUS numpy array'
            assert arg.flags['CARRAY'], f'argument {i} has a C++ {ctype} type, and must be a CARRAY numpy array'
        elif ctype == 'bool*':
            assert isinstance(arg, np.ndarray), f'argument {i} has a C++ {ctype} type, and must be np.ndarray.'
            assert arg.dtype == np.bool, f'argument {i} has a C++ {ctype} type, and must have dtype == np.bool'
            assert arg.flags['BEHAVED'], f'argument {i} has a C++ {ctype} type, and must be a BEHAVED numpy array'
            assert arg.flags['C_CONTIGUOUS'], f'argument {i} has a C++ {ctype} type, and must be a C_CONTIGUOUS numpy array'
            assert arg.flags['CARRAY'], f'argument {i} has a C++ {ctype} type, and must be a CARRAY numpy array'
        elif ctype == 'double':
            assert isinstance(arg, np.float64), f'argument {i} has a C++ {ctype} type, and must have be of np.float64 type'
        elif ctype == 'int':
            assert isinstance(arg, np.int32), f'argument {i} has a C++ {ctype} type, and must have be of np.int32 type'
        elif c2py_type[ctype] == c_void_p:
            assert isinstance(arg, int) or (arg is None), f'argument {i} has a C++ {ctype} type, which we treat in python as c_void_p, and must be of int or None type.'
        else:
            raise Exception(f'check undefined for argument {i}, which has a C++ type {ctype}')

def is_within_f32_range(np_arr):
    c1 = np.isfinite(np_arr).all()
    c2 = np.logical_and(np_arr>=np.finfo(np.float32).min,
                        np_arr<=np.finfo(np.float32).max).all()
    return np.logical_and(c1, c2)

def lookup_xml_path(xml_path):
    if xml_path is None:
        out_path = None
    elif os.sep in xml_path:
        out_path = abspath(xml_path)
    else:
        schdirs = [os.getcwd(), DFLTXMLDIR]
        out_path = None
        for schdir in schdirs:
            potential_path = abspath(osp.join(schdir, xml_path))
            if exists(potential_path):
                out_path = potential_path
                break

    assert xml_path is not None, f'XML file {xml_path} is invalid.'
    assert out_path is not None, f'XML file {xml_path} cannot be located.'
    assert exists(out_path), f'XML file {xml_path} does not exist.'

    return out_path

#############################################################
#################### C++ Wrapper Classes ####################
#############################################################

si_args = ('args_double', 'args_int')
SimInitArgs = namedtuple("SimInitArgs", si_args)

vro_args = ("obs_greedy", "action_greedy", "action_vine", "Q_greedy",
            "eta_greedy", "return_greedy", "Q_vine", "eta_vine",
            "return_vine", "done_steps", "done_steps_vine", "obs_greedy_all",
            "action_greedy_all", "obs_vine_all", "action_vine_all",
            "reset_times_out")
VineRollOuts = namedtuple("VineRollOuts", vro_args)

glro_args = ("eta_greedy", "return_greedy", "done_steps")
GreedyLiteRollOuts = namedtuple("GreedyLiteRollOuts", glro_args)

fro_outs = ("observations", "actions", "rewards", "done_steps")
FullRollOuts = namedtuple("FullRollOuts", fro_outs)

#############################################################
############# The Leg Roller Utility Functions ##############
#############################################################

def sample_leg_siminit_args(traj_num, options, np_random, rng_states):
    # Extracting the arguments
    th_hip_low, th_hip_high = tuple(options['theta_hip_init'])
    th_knee_low, th_knee_high = tuple(options['theta_knee_init'])
    om_hip_low, om_hip_high = tuple(options['omega_hip_init'])
    om_knee_low, om_knee_high = tuple(options['omega_knee_init'])
    pos_slider_low, pos_slider_high = tuple(options['pos_slider_init'])
    vel_slider_low, vel_slider_high = tuple(options['vel_slider_init'])
    jump_time_low, jump_time_high = tuple(options['jump_time_bounds'])

    # SimIF Variables
    theta_hip = np.zeros((traj_num,), np.float64)
    theta_knee = np.zeros((traj_num,), np.float64)
    omega_hip = np.zeros((traj_num,), np.float64)
    omega_knee = np.zeros((traj_num,), np.float64)
    pos_slider = np.zeros((traj_num,), np.float64)
    vel_slider = np.zeros((traj_num,), np.float64)
    jumping_time = np.zeros((traj_num,), np.float64)
    noise_index = np.zeros((traj_num,), np.int32)

    if rng_states is None:
        rng_states = [None] * traj_num
    else:
        assert len(rng_states) == traj_num

    for i, rng_st in enumerate(rng_states):
        if rng_st is not None:
            np_random.set_state(rng_st)
        theta_hip[i] = np_random.uniform(th_hip_low, th_hip_high)
        theta_knee[i] = np_random.uniform(th_knee_low, th_knee_high)
        omega_hip[i] = np_random.uniform(om_hip_low, om_hip_high)
        omega_knee[i] = np_random.uniform(om_knee_low, om_knee_high)
        pos_slider[i] = np_random.uniform(pos_slider_low, pos_slider_high)
        vel_slider[i] = np_random.uniform(vel_slider_low, vel_slider_high)
        jumping_time[i] = np_random.uniform(jump_time_low, jump_time_high)
        noise_index[i] = np_random.randint(0, 400, dtype=np.int32)

    lsia = dict(theta_hip=theta_hip, theta_knee=theta_knee, omega_hip=omega_hip,
                omega_knee=omega_knee, pos_slider=pos_slider, vel_slider=vel_slider,
                jumping_time=jumping_time, noise_index=noise_index)

    # These must have the same order as specified in SimIF.cpp under
    # double* SimInterface::reset(double* init_args_double, int* init_args_int)
    init_args_double = np.stack((theta_hip, theta_knee, omega_hip, omega_knee,
                                 pos_slider, vel_slider, jumping_time), axis=1)
    init_args_int = np.stack((noise_index,), axis=1)
    sia = SimInitArgs(init_args_double, init_args_int)

    return sia, lsia

#############################################################
################### The Leg Roller Class ####################
#############################################################

class CRoller:
    do_check_xml = False
    def __init__(self, lib_path):
        defmsg_ = 'You must define self.{name} in the __init__ constructor'
        typmsg_ = 'The defined self.{attr} must have the {objtype} type'
        must_defs = [('dflt_opts', dict), ('options', dict), ('obs_dim', int),
                     ('act_dim', int), ('max_steps_per_traj', int),
                     ('opt2choics', dict)]
        if has_gym:
            must_defs += [('action_space', Space), ('observation_space', Space),
                          ('reward_range', tuple), ('metadata', dict)]
        for attr_name, attr_type in must_defs:
            assert hasattr(self, attr_name), defmsg_.format(name=attr_name)
            assert isinstance(getattr(self, attr_name), attr_type), \
                typmsg_.format(objtype=attr_type, attr=attr_name)

        lib = ctypes.cdll.LoadLibrary(lib_path)
        func_cdescriptions = dict()
        self.lib_get_options_available = True
        for declaration_str in self.declarations_list:
            c_description = describe_cfunc(declaration_str)
            func_name = c_description['func_name']
            if not(hasattr(lib, func_name)):
                if func_name == 'rollout_get_build_options':
                    self.lib_get_options_available = False
                else:
                    m = f'function {func_name} is not in the lib.'
                    raise Exception(m)
                continue
            func_cdescriptions[func_name] = c_description
            res_type_c = c_description['res_type_c']
            arg_ctypes = pyarg_type(c_description)
            res_ctype = c2py_type[res_type_c]
            set_lib_types(lib, c_description, arg_ctypes, res_ctype)

        self.lib = lib
        self.obj = lib.rollout_new()
        self.func_cdescriptions = func_cdescriptions

        self.h1 = self.options['h1']
        self.h2 = self.options['h2']
        assert self.h1 == 64, 'h1 != 64 has not been implemented yet'
        assert self.h2 == 64, 'h2 != 64 has not been implemented yet'

        self.np_random = None
        self.np_random_tmp = np.random.RandomState(0)

        self.observation_dim = self.obs_dim
        self.action_dim = self.act_dim

        self.set_policy_calls = 0
        self.rollout_calls = 0
        self.r_init_named = None
        self.r_init_args_double = None
        self.r_init_args_int = None

        self.cpp_build_opts = None
        self.xml_var = None
        self.get_build_options()
        self.check_build_options()

    @classmethod
    def get_dflt_opts(cls):
        raise RuntimeError('not implemented')

    @property
    def declarations_list(self):
        dec_list = []
        dec_list.append("""Rollout* rollout_new()""")

        dec_list.append("""void rollout_set_simif_inits(Rollout* rollout, double* args_double, int* args_int)""")

        dec_list.append("""void rollout_set_mlp_weights(Rollout* rollout, double* fc1, double* fc2, double* fc3,
                           double* fc1_bias, double* fc2_bias, double* fc3_bias)""")

        dec_list.append("""void rollout_greedy_lite(Rollout* rollout, int traj_num, int n_steps, double gamma,
                           double* eta_greedy, double* return_greedy,
                           int* done_steps)""")

        dec_list.append("""void rollout_infer_mlp(Rollout* rollout, int input_num,
                           double* mlp_input, double* mlp_output)""")

        dec_list.append("""void rollout_vine(Rollout* rollout, int traj_num, int n_steps, double gamma, int expl_steps,
                        int* reset_times, double* expl_noise, double* obs_greedy, double* action_greedy, double* action_vine,
                        double* Q_greedy, double* eta_greedy, double* return_greedy, double* Q_vine, double* eta_vine,
                        double* return_vine, int* done_steps, int* done_steps_vine, double* obs_greedy_all,
                        double* action_greedy_all, double* obs_vine_all, double* action_vine_all, int* reset_times_out)""")

        dec_list.append("""void rollout_stochastic(Rollout* rollout, int traj_num, int n_steps,
                           double* expl_noise, double* obs, double* action, double* rewards,
                           int* done_steps)""")

        dec_list.append("""void rollout_get_build_options(char* keys, double* vals, char* xml_var,
                           int keys_len, int vals_len, int xml_var_len)""")

        dec_list.append("""void rollout_partial_stochastic(Rollout* rollout, int n_steps, double* expl_noise,
                           double* obs, double* action, double* rewards, bool* dones)""")

        dec_list.append("""void rollout_reset(Rollout* rollout, int traj_idx, double* obs)""")

        return dec_list

    def seed(self, seed=None):
        if has_gym:
            self.np_random, seed_ = seeding.np_random(seed)
        else:
            seed_ = None
            self.np_random = np.random.RandomState(seed_)
        return [seed_]

    def get_spaces(self, useless_arg = None):
        if has_gym:
            return self.observation_space, self.action_space
        else:
            return None, None

    def _check_call_nums(self):
        msg_ = f'You called rollout {self.rollout_calls} times, and set_policy {self.set_policy_calls}'
        assert self.rollout_calls == self.set_policy_calls, msg_

    def _set_simif_inits(self, args_double, args_int):
        c_description = self.func_cdescriptions['rollout_set_simif_inits']
        # NOTE: It's imperative that you keep some reference of r_init elements
        #       so that python's garbage collector would not reallocate data in it.
        self.r_init_args_double = args_double
        self.r_init_args_int = args_int
        args_ = (self.obj, args_double, args_int)
        check_args(c_description, args_)
        self.lib.rollout_set_simif_inits(*args_)

    def set_torch_policy(self, policy):
        named_parameters = dict(fc1 = policy.fc1.weight.data.numpy(),
                                fc2 = policy.fc2.weight.data.numpy(),
                                fc3 = policy.fc3_mu.weight.data.numpy(),
                                fc1_bias = policy.fc1.bias.data.numpy(),
                                fc2_bias = policy.fc2.bias.data.numpy(),
                                fc3_bias = policy.fc3_mu.bias.data.numpy())
        self.set_np_policy(named_parameters)
        self.policy_logstd = None

    def set_np_policy(self, named_parameters):
        obs_dim, act_dim = self.obs_dim, self.act_dim
        h1, h2 = self.h1, self.h2

        def_name_params = dict(fc1=None, fc2=None, fc3=None,
                               fc1_bias=None, fc2_bias=None, fc3_bias=None,)
        extra_names = named_parameters.keys() - def_name_params.keys()
        if len(extra_names) > 0:
            raise RuntimeError(f'Unknown parameter names: {extra_names}')

        fc1 = named_parameters['fc1']
        fc2 = named_parameters['fc2']
        fc3 = named_parameters['fc3']
        fc1_bias = named_parameters['fc1_bias']
        fc2_bias = named_parameters['fc2_bias']
        fc3_bias = named_parameters['fc3_bias']

        assert isinstance(fc1, np.ndarray)
        assert isinstance(fc2, np.ndarray)
        assert isinstance(fc3, np.ndarray)
        assert isinstance(fc1_bias, np.ndarray)
        assert isinstance(fc2_bias, np.ndarray)
        assert isinstance(fc3_bias, np.ndarray)

        assert fc1.shape == (h1, obs_dim), f'{fc1.shape} != {(h1, obs_dim)}'
        assert fc2.shape == (h2, h1)
        assert fc3.shape == (act_dim, h2)
        assert fc1_bias.shape == (h1,)
        assert fc2_bias.shape == (h2,)
        assert fc3_bias.shape == (act_dim,)

        assert np.isfinite(fc1).all()
        assert np.isfinite(fc2).all()
        assert np.isfinite(fc3).all()
        assert np.isfinite(fc1_bias).all()
        assert np.isfinite(fc2_bias).all()
        assert np.isfinite(fc3_bias).all()

        assert is_within_f32_range(fc1)
        assert is_within_f32_range(fc2)
        assert is_within_f32_range(fc3)
        assert is_within_f32_range(fc1_bias)
        assert is_within_f32_range(fc2_bias)
        assert is_within_f32_range(fc3_bias)

        fc1 = np.ascontiguousarray(fc1, dtype=np.float64)
        fc2 = np.ascontiguousarray(fc2, dtype=np.float64)
        fc3 = np.ascontiguousarray(fc3, dtype=np.float64)
        fc1_bias = np.ascontiguousarray(fc1_bias, dtype=np.float64)
        fc2_bias = np.ascontiguousarray(fc2_bias, dtype=np.float64)
        fc3_bias = np.ascontiguousarray(fc3_bias, dtype=np.float64)

        self._set_mlp_weights(fc1, fc2, fc3, fc1_bias, fc2_bias, fc3_bias)
        self.set_policy_calls += 1

    def set_tf_policy(self, tf_session, naming='trpo'):
        tf_graph = tf_session.graph
        if naming in ('trpo', 'ppo', 'ppo1', 'ppo2'):
            fc1_w_tensor = tf_graph.get_tensor_by_name("model/pi_fc0/w:0")
            fc2_w_tensor = tf_graph.get_tensor_by_name("model/pi_fc1/w:0")
            fc3_w_tensor = tf_graph.get_tensor_by_name("model/pi/w:0")
            fc1_b_tensor = tf_graph.get_tensor_by_name("model/pi_fc0/b:0")
            fc2_b_tensor = tf_graph.get_tensor_by_name("model/pi_fc1/b:0")
            fc3_b_tensor = tf_graph.get_tensor_by_name("model/pi/b:0")
            logstd_tensor = tf_graph.get_tensor_by_name("model/pi/logstd:0")

            pkg = tf_session.run([fc1_w_tensor, fc2_w_tensor, fc3_w_tensor,
                                  fc1_b_tensor, fc2_b_tensor, fc3_b_tensor,
                                  logstd_tensor])
            fc1_w, fc2_w, fc3_w, fc1_b, fc2_b, fc3_b, logstd = pkg
            self.policy_logstd = logstd
        elif naming in ('td3',):
            fc1_w_tensor = tf_graph.get_tensor_by_name("model/pi/fc0/kernel:0")
            fc2_w_tensor = tf_graph.get_tensor_by_name("model/pi/fc1/kernel:0")
            fc3_w_tensor = tf_graph.get_tensor_by_name("model/pi/dense/kernel:0")
            fc1_b_tensor = tf_graph.get_tensor_by_name("model/pi/fc0/bias:0")
            fc2_b_tensor = tf_graph.get_tensor_by_name("model/pi/fc1/bias:0")
            fc3_b_tensor = tf_graph.get_tensor_by_name("model/pi/dense/bias:0")

            pkg = tf_session.run([fc1_w_tensor, fc2_w_tensor, fc3_w_tensor,
                                  fc1_b_tensor, fc2_b_tensor, fc3_b_tensor])
            fc1_w, fc2_w, fc3_w, fc1_b, fc2_b, fc3_b = pkg
            self.policy_logstd = None
        else:
            raise ValueError(f'Unknown naming {naming}')

        named_parameters = dict(fc1 = fc1_w.T,
                                fc2 = fc2_w.T,
                                fc3 = fc3_w.T,
                                fc1_bias = fc1_b,
                                fc2_bias = fc2_b,
                                fc3_bias = fc3_b)
        self.set_np_policy(named_parameters)

    def greedy_lite(self, traj_num, n_steps, gamma, rng_states=None):
        self._set_rand_simif_inits(traj_num, rng_states)
        obs_dim = self.obs_dim
        act_dim = self.act_dim
        reward_scaling = self.options['reward_scaling']
        self.rollout_calls += 1
        self._check_call_nums()

        eta_greedy = np.zeros((traj_num,), np.float64, order='C')
        return_greedy = np.zeros((traj_num,), np.float64, order='C')
        done_steps = np.zeros((traj_num,), np.int32, order='C')

        traj_num = np.int32(traj_num)
        n_steps = np.int32(n_steps)
        gamma = np.float64(gamma)

        c_description = self.func_cdescriptions['rollout_greedy_lite']
        args_ = (self.obj, traj_num, n_steps, gamma, eta_greedy, return_greedy, done_steps)
        check_args(c_description, args_)
        self.lib.rollout_greedy_lite(*args_)

        if reward_scaling is not None:
            eta_greedy *= reward_scaling
            return_greedy *= reward_scaling

        output = GreedyLiteRollOuts(eta_greedy=eta_greedy, return_greedy=return_greedy,
                                    done_steps=done_steps)

        for key, val in output._asdict().items():
            assert not np.isnan(val).any(), f'Output {key} has a NaN values: {val}'

        return output._asdict()

    def vine(self, traj_num, n_steps, gamma, expl_steps,
             reset_times, expl_noise, rng_states):
        assert expl_noise is not None

        self._set_rand_simif_inits(traj_num, rng_states)
        obs_dim = self.obs_dim
        act_dim = self.act_dim
        reward_scaling = self.options['reward_scaling']
        self.rollout_calls += 1
        self._check_call_nums()

        # Sanitizing the Input Arrays
        assert isinstance(reset_times, np.ndarray)
        assert isinstance(expl_noise, np.ndarray)
        assert reset_times.shape == (traj_num,)
        assert expl_noise.shape == (traj_num, expl_steps, act_dim)
        reset_times = np.ascontiguousarray(reset_times, dtype=np.int32)
        expl_noise = np.ascontiguousarray(expl_noise, dtype=np.float64)
        traj_num = np.int32(traj_num)
        n_steps = np.int32(n_steps)
        gamma = np.float64(gamma)
        expl_steps = np.int32(expl_steps)

        obs_greedy = np.zeros((traj_num, expl_steps, obs_dim), np.float64, order='C')
        action_greedy = np.zeros((traj_num, expl_steps, act_dim), np.float64, order='C')
        action_vine = np.zeros((traj_num, expl_steps, act_dim), np.float64, order='C')
        Q_greedy = np.zeros((traj_num,), np.float64, order='C')
        eta_greedy = np.zeros((traj_num,), np.float64, order='C')
        return_greedy = np.zeros((traj_num,), np.float64, order='C')
        Q_vine = np.zeros((traj_num,), np.float64, order='C')
        eta_vine = np.zeros((traj_num,), np.float64, order='C')
        return_vine = np.zeros((traj_num,), np.float64, order='C')
        done_steps = np.zeros((traj_num,), np.int32, order='C')
        done_steps_vine = np.zeros((traj_num,), np.int32, order='C')

        obs_greedy_all = np.zeros((traj_num, n_steps, obs_dim), np.float64, order='C')
        action_greedy_all = np.zeros((traj_num, n_steps, act_dim), np.float64, order='C')
        obs_vine_all = np.zeros((traj_num, n_steps, obs_dim), np.float64, order='C')
        action_vine_all = np.zeros((traj_num, n_steps, act_dim), np.float64, order='C')
        reset_times_out = np.zeros((traj_num,), np.int32, order='C')

        c_description = self.func_cdescriptions['rollout_vine']
        args_ = (self.obj, traj_num, n_steps, gamma, expl_steps, reset_times, expl_noise,
                 obs_greedy, action_greedy, action_vine, Q_greedy, eta_greedy, return_greedy,
                 Q_vine, eta_vine, return_vine, done_steps, done_steps_vine,
                 obs_greedy_all, action_greedy_all, obs_vine_all, action_vine_all,
                 reset_times_out)

        check_args(c_description, args_)
        self.lib.rollout_vine(*args_)

        if reward_scaling is not None:
            Q_greedy *= reward_scaling
            Q_vine *= reward_scaling
            eta_greedy *= reward_scaling
            return_greedy *= reward_scaling
            eta_vine *= reward_scaling
            return_vine *= reward_scaling

        output = VineRollOuts(obs_greedy=obs_greedy, action_greedy=action_greedy,
                              action_vine=action_vine, Q_greedy=Q_greedy,
                              eta_greedy=eta_greedy, return_greedy=return_greedy,
                              Q_vine=Q_vine, eta_vine=eta_vine, return_vine=return_vine,
                              done_steps=done_steps, done_steps_vine=done_steps_vine,
                              obs_greedy_all=obs_greedy_all, action_greedy_all=action_greedy_all,
                              obs_vine_all=obs_vine_all, action_vine_all=action_vine_all,
                              reset_times_out=reset_times_out)

        for key, val in output._asdict().items():
            assert not np.isnan(val).any(), f'Output {key} has a NaN values: {val}'
            # note to self: if this assertion fails, it could be because I didn't
            # write the C++ rollout code in a way that it would "safely" populate
            # the outputs in case the sim interface is done before the full n_steps
            # were taken.

        return output._asdict()

    def stochastic(self, traj_num, n_steps, expl_noise=None, rng_states=None):
        self._set_rand_simif_inits(traj_num, rng_states)
        obs_dim = self.obs_dim
        act_dim = self.act_dim
        self.rollout_calls += 1
        reward_scaling = self.options['reward_scaling']
        self._check_call_nums()

        if expl_noise is None:
            expl_noise = np.zeros((traj_num, n_steps, act_dim), dtype=np.float64, order='C')
        else:
            assert isinstance(expl_noise, np.ndarray)
            assert expl_noise.shape == (traj_num, n_steps, act_dim)
            expl_noise = np.ascontiguousarray(expl_noise, dtype=np.float64)
            assert is_within_f32_range(expl_noise)

        traj_num = np.int32(traj_num)
        n_steps = np.int32(n_steps)

        # Output Variables
        obs = np.zeros((traj_num, n_steps, obs_dim), np.float64, order='C')
        action = np.zeros((traj_num, n_steps, act_dim), np.float64, order='C')
        rewards = np.zeros((traj_num, n_steps), np.float64, order='C')
        done_steps = np.zeros((traj_num,), np.int32, order='C')

        c_description = self.func_cdescriptions['rollout_stochastic']
        args_ = (self.obj, traj_num, n_steps, expl_noise, obs, action, rewards, done_steps)
        check_args(c_description, args_)
        self.lib.rollout_stochastic(*args_)

        if reward_scaling is not None:
            rewards *= reward_scaling

        output = FullRollOuts(observations=obs, actions=action,
                              rewards=rewards, done_steps=done_steps)

        for key, val in output._asdict().items():
            assert not np.isnan(val).any(), f'Output {key} has a NaN values.'
            # note to self: if this assertion fails, it could be because I didn't
            # write the C++ rollout code in a way that it would "safely" populate
            # the outputs in case the sim interface is done before the full n_steps
            # were taken.

        return output._asdict()

    def greedy(self, traj_num, n_steps, rng_states=None):
        return self.stochastic(traj_num, n_steps, None, rng_states)

    def reset(self):
        traj_num, traj_idx = 1, 0
        self._set_rand_simif_inits(traj_num, None)
        obs = np.zeros((self.obs_dim,), np.float64, order='C')
        c_description = self.func_cdescriptions['rollout_reset']
        args_ = (self.obj, np.int32(traj_idx), obs)
        check_args(c_description, args_)
        self.lib.rollout_reset(*args_)
        traj_rands = {key: val[0].copy() for key, val in
                      self.r_init_named.items()}
        output = dict(observation=obs, traj_rands=traj_rands)
        return output

    def multiple_steps(self, n_steps, policy, expl_noise=None, policy_lib='tf', naming='trpo'):
        obs_dim = self.obs_dim
        act_dim = self.act_dim
        reward_scaling = self.options['reward_scaling']
        if self.max_steps_per_traj is not None:
            n_steps = min(self.max_steps_per_traj, n_steps)

        if policy_lib == 'tf':
            tf_session = policy.sess
            self.set_tf_policy(tf_session, naming=naming)
        elif policy_lib == 'np':
            # In this case, policy is a named_parameters dictionary
            # with an extra key 'policy_logstd'
            named_parameters = policy.copy() # This is a shallow copy, so don't worry about performance!
            self.policy_logstd = named_parameters.pop('policy_logstd')
            self.set_np_policy(named_parameters)
        else:
            raise RuntimeError(f'Unknown policy_lib {policy_lib}')

        self.rollout_calls += 1
        self._check_call_nums()

        if expl_noise is None:
            if self.policy_logstd is None:
                expl_noise = np.zeros((n_steps, act_dim), dtype=np.float64, order='C')
            else:
                policy_std = np.exp(self.policy_logstd).reshape(1, act_dim)
                assert is_within_f32_range(policy_std)
                expl_noise = self.np_random.randn(n_steps, act_dim)
                expl_noise *= policy_std
        elif isinstance(expl_noise, np.ndarray):
            assert expl_noise.shape == (n_steps, act_dim)
            assert is_within_f32_range(expl_noise)
        elif callable(expl_noise):
            expl_noise = expl_noise(n_steps)
            assert isinstance(expl_noise, np.ndarray)
            assert expl_noise.shape == (n_steps, act_dim)
            assert is_within_f32_range(expl_noise)
        else:
            raise RunTimeError(f'rule for expl_noise not implemented.')

        expl_noise = np.ascontiguousarray(expl_noise, dtype=np.float64)

        n_steps = np.int32(n_steps)

        # Output Variables
        obs = np.zeros((n_steps, obs_dim), np.float64, order='C')
        action = np.zeros((n_steps, act_dim), np.float64, order='C')
        rewards = np.zeros((n_steps,), np.float64, order='C')
        dones = np.zeros((n_steps,), np.bool, order='C')

        c_description = self.func_cdescriptions['rollout_partial_stochastic']
        args_ = (self.obj, n_steps, expl_noise, obs, action, rewards, dones)
        check_args(c_description, args_)
        self.lib.rollout_partial_stochastic(*args_)

        if dones.any():
            T = np.argmax(dones) + 1
            obs = obs[:T]
            action = action[:T]
            rewards = rewards[:T]
            dones = dones[:T]

        if reward_scaling is not None:
            rewards *= reward_scaling

        output = dict(observations=obs, actions=action,
                      rewards=rewards, dones=dones)

        for key, val in output.items():
            # assert not np.isnan(val).any(), f'Output {key} has a NaN values.'
            pass
            # note to self: if this assertion fails, it could be because I didn't
            # write the C++ rollout code in a way that it would "safely" populate
            # the outputs in case the sim interface is done before the full n_steps
            # were taken.

        return output

    def _set_mlp_weights(self, fc1, fc2, fc3, fc1_bias, fc2_bias, fc3_bias):
        c_description = self.func_cdescriptions['rollout_set_mlp_weights']
        self.set_mlp_args_ = (self.obj, fc1, fc2, fc3, fc1_bias, fc2_bias, fc3_bias)
        check_args(c_description, self.set_mlp_args_)
        return self.lib.rollout_set_mlp_weights(*self.set_mlp_args_)

    def infer_mlp(self, mlp_input):
        obs_dim = self.obs_dim
        act_dim = self.act_dim

        assert mlp_input.shape[1] == obs_dim
        input_num = np.int32(mlp_input.shape[0])
        mlp_output = np.zeros((input_num, act_dim), np.float64, order='C')

        c_description = self.func_cdescriptions['rollout_infer_mlp']
        args_ = (self.obj, input_num, mlp_input, mlp_output)
        check_args(c_description, args_)
        self.lib.rollout_infer_mlp(*args_)

        return mlp_output

    def get_options(self):
        # This is a meta-data function for tensorboard logging
        output = dict()
        if self.lib_get_options_available:
            cpp_build_opts_str = ''.join([f'\n    * {key} = {value}'
                                          for key, value in
                                          self.cpp_build_opts.items()
                                          if not key.endswith('__')])
            output['**C++ Build Options**\n'] = cpp_build_opts_str

        options_str = ''.join([f'\n      * {key} = {value}'
                               for key, value in
                               self.options.items()])
        output['**Python Options**\n'] = options_str
        return output

    def get_nondefault_options(self):
        # This is a meta-data function for tensorboard logging
        return dict()

    def get_metadata(self):
        # This is a meta-data function for tensorboard logging
        src = dict()
        with open(__file__, 'r') as my_fh:
            src['usr/binding.py'] = my_fh.read()

        cleg_dir_ = dirname(dirname(__file__))
        for filename in ['src/SimIF.cpp', 'src/SimIF.hpp',
                         'src/Rollout.cpp', 'src/Rollout.hpp',
                         'src/MlpIF.cpp', 'src/MlpIF.hpp',
                         'Makefile', 'src/defs.hpp']:
            simif_cpp = cleg_dir_ + '/' + filename
            if exists(simif_cpp):
                with open(__file__, 'r') as my_fh:
                    src[filename] = my_fh.read()
        src_str = ''.join([f'{key}:\n```\n{val}\n```\n' + '*'*100 + '\n'
                           for key, val in src.items()])
        output = dict()
        output['source_code'] = src_str
        if self.lib_get_options_available:
            output['xml_file'] = '```\n' + self.xml_var + '\n```'
        return output

    def save(self, filename):
        # This is a meta-data function for tensorboard logging
        pass

    def simulate_and_plot(self, *args, **kwargs):
        msg_ = 'Currently, another plotting environemnt is necessary.'
        assert hasattr(self, 'plot_env'), msg_
        assert self.plot_env is not None, msg_
        return self.plot_env.simulate_and_plot(*args, **kwargs)

    def get_build_options(self):
        if not self.lib_get_options_available:
            return
        keys_len = 10000
        vals_len = 1000
        xml_var_len = 20000
        keys = create_string_buffer(b"\0" * keys_len)
        vals = np.empty((vals_len,), np.float64, order='C')
        xml_var = create_string_buffer(b"\0" * xml_var_len)
        self.lib.rollout_get_build_options(keys, vals, xml_var, keys_len, vals_len, xml_var_len)
        keys_list = "".join([x.decode("utf8") for x in keys]).split(b"\0".decode("utf8"))
        keys_list = keys_list[:keys_list.index('')]

        xml_var_str = "".join([x.decode("utf8") for x in xml_var])
        xml_var_str = xml_var_str[:xml_var_str.index(b"\0".decode("utf8"))]

        vals_list = vals[:len(keys_list)].tolist()
        bld_dict_raw = {key:val for key,val in zip(keys_list, vals_list)}

        bld_dict = bld_dict_raw.copy()
        exception_msgs = []
        for key, mapping in self.opt2choics.items():
            if key not in bld_dict:
                continue
            found_trans = False
            for cpp_name, py_name in mapping.items():
                if np.isclose(bld_dict[key], bld_dict[cpp_name]):
                    bld_dict[key] = py_name
                    found_trans = True
                    break
            if not found_trans:
                m_  = f'I could not find a category translation for the '
                m_ += f'build option "{key}" = {bld_dict[key]}.\n'
                m_ += f'You probably need to add a translation entry to "{key}" of'
                m_ += f'the opt2choics dictionary in my get_build_options function.'
                exception_msgs.append(m_)
        if len(exception_msgs) > 0:
            full_exc_msg = f'I found {len(exception_msgs)} issues:\n'
            full_exc_msg += ('\n' + '*'*50 + '\n').join(exception_msgs)
            raise Exception(full_exc_msg)

        for key, mapping in self.opt2choics.items():
            for cpp_name, py_name in mapping.items():
                bld_dict.pop(cpp_name, None)

        self.cpp_build_opts = bld_dict
        self.xml_var = xml_var_str

    def check_build_options(self):
        if not self.lib_get_options_available:
            return

        assert self.cpp_build_opts is not None
        known_opt_types = ['init_opts', 'model_opts', 'bnd_opts',
                           'unimplmntd_opts', 'irrlvnt_opts',
                           'wrapper_opts', 'other_opts', 'mlp_opts']

        for opt_type in self.dflt_opts.keys():
            m_ = f'check rule undefined for "{opt_type}" opt type'
            assert opt_type in known_opt_types, m_

        err_msg_list = []
        for key in self.dflt_opts['model_opts'].keys():
            if (key not in self.cpp_build_opts):
                continue
            m_  = 'The C++ library used a different build option than the '
            m_ += 'one provided to the python binder:\n'
            m_ += f'  C++    Build --> {key}: {self.cpp_build_opts[key]}\n'
            m_ += f'  My   Options --> {key}: {self.options[key]}\n'
            if not(self.options[key] == self.cpp_build_opts[key]):
                err_msg_list.append(m_)

        for key in self.dflt_opts['bnd_opts'].keys():
            if (key not in self.cpp_build_opts):
                continue
            cpp_opt = [self.cpp_build_opts[key+'_low'], self.cpp_build_opts[key+'_high']]
            m_  = 'The C++ library used a different build option than the '
            m_ += 'one provided to the python binder:\n'
            m_ += f'  C++    Build --> {key}: {cpp_opt}\n'
            m_ += f'  My   Options --> {key}: {self.options[key]}\n'
            if not(self.options[key] == cpp_opt):
                err_msg_list.append(m_)

        for key in self.dflt_opts['mlp_opts'].keys():
            cpp_opt = self.cpp_build_opts.get(key, self.dflt_opts['mlp_opts'][key])
            m_  = 'The C++ library used a different build option than the '
            m_ += 'one provided to the python binder:\n'
            m_ += f'  C++    Build --> {key}: {cpp_opt}\n'
            m_ += f'  My   Options --> {key}: {self.options[key]}\n'
            if not(self.options[key] == cpp_opt):
                err_msg_list.append(m_)

        assert set(self.dflt_opts['other_opts'].keys()).issubset({'xml_file'})
        if self.do_check_xml:
            assert 'xml_type' in self.cpp_build_opts
            py_xml_path = self.options['xml_file']
            with open(py_xml_path, "r") as fp:
                py_xml_content = "".join(fp.readlines()).strip()

            if self.cpp_build_opts['xml_type'] == 'file_path':
                cpp_xml_path = self.xml_var
                m_ = f'The C++ Build expects an xml file at {cpp_xml_path} which does not exist!'
                assert exists(xml_path), m_
                with open(cpp_xml_path, "r") as fp:
                    cpp_xml_content = "".join(fp.readlines()).strip()
            elif self.cpp_build_opts['xml_type'] == 'content':
                cpp_xml_path = None
                cpp_xml_content = self.xml_var.strip()
            else:
                raise Exception(f'Undefined C++ xml_type: %s' % self.cpp_build_opts["xml_type"])
            if py_xml_content != cpp_xml_content:
                m_  = f'The python interface xml file has different content than the one used by C++:\n'
                m_ += f'  -> The Python interface xml file path: {py_xml_path}\n'
                if cpp_xml_path is not None:
                    m_ += f'  -> The C++              xml file path: {cpp_xml_path}\n'
                m_ += f'Python XML Content: Length={len(py_xml_content)}\n\n'
                m_ += py_xml_content
                m_ += '\n' + '-'*40 + '\n'
                m_ += f'C++ XML Content: Length={len(cpp_xml_content)}\n\n'
                m_ += cpp_xml_content
                m_ += '\n' + '-'*40 + '\n'
                err_msg_list.append(m_)

        if len(err_msg_list) > 0:
            full_err_msg = f'There were {len(err_msg_list)} errors when '
            full_err_msg += 'matching the compiled file to the input options:\n'
            err_msg_list = [f'Error {i} --> {x}' for i,x in enumerate(err_msg_list)]
            full_err_msg = ('*'*100 + '\n').join(err_msg_list)
            print(full_err_msg)
            raise RuntimeError('C++ Build Options Do Not Matching the Python Options.')

    @classmethod
    def preproc_opts(cls, options):
        pass # nothing global needs to happen

    @classmethod
    def get_clashing_opts(cls, opts1, opts2):
        dflt_opts = cls.get_dflt_opts()

        ###############################################
        ##### Sanitizing/Completing opts1, opts2 ######
        ###############################################
        opts1, opts2 = dict(opts1), dict(opts2)
        opts0 = dict()
        for opt_dict in dflt_opts.values():
            opts0.update(opt_dict)

        extra_keys1 = opts1.keys() - opts0.keys()
        extra_keys2 = opts2.keys() - opts0.keys()
        if len(extra_keys1) > 0:
            raise Exception(f'Unknown options from opts1: {extra_keys1}')
        if len(extra_keys2) > 0:
            raise Exception(f'Unknown options from opts2: {extra_keys2}')

        opts1_, opts2_ = opts1, opts2
        opts1, opts2 = dict(opts0), dict(opts0)
        opts1.update(opts1_)
        opts2.update(opts2_)

        ###############################################
        ###### Translating items in opts1, opts2 ######
        ###############################################
        cls.preproc_opts(opts1)
        cls.preproc_opts(opts2)

        ###############################################
        ######### Performing Equality Checks ##########
        ###############################################
        ineq_opts = []
        for opt_type in ['model_opts', 'bnd_opts', 'mlp_opts']:
            ineq_opts += [opt for opt, dfltval in dflt_opts[opt_type].items()
                          if opts1.pop(opt, dfltval) != opts2.pop(opt, dfltval)]

        if cls.do_check_xml:
            dflt_xmlpth = dflt_opts['other_opts']['xml_file']
            opts1_xml_path = lookup_xml_path(opts1.pop('xml_file', dflt_xmlpth))
            opts2_xml_path = lookup_xml_path(opts2.pop('xml_file', dflt_xmlpth))
            if opts1_xml_path != opts2_xml_path:
                if osp.getsize(opts1_xml_path) != osp.getsize(opts2_xml_path):
                    ineq_opts.append('xml_file')
                else:
                    if not filecmp.cmp(opts1_xml_path, opts2_xml_path):
                        ineq_opts.append('xml_file')
        else:
            opts1.pop('xml_file', None)
            opts2.pop('xml_file', None)

        ### Making sure no other unknown keys exist
        extra_keys_opts1 = opts1.keys()
        extra_keys_opts2 = opts2.keys()
        for opt_type in ['init_opts', 'unimplmntd_opts', 'irrlvnt_opts',
                         'wrapper_opts', 'other_opts']:
            extra_keys_opts1 = extra_keys_opts1 - dflt_opts[opt_type].keys()
            extra_keys_opts2 = extra_keys_opts2 - dflt_opts[opt_type].keys()
        if len(extra_keys_opts1) > 0:
            raise ValueError(f'Unknown options in opts1: {extra_keys_opts1}')
        if len(extra_keys_opts2) > 0:
            raise ValueError(f'Unknown options in opts2: {extra_keys_opts2}')

        return ineq_opts

class LegCRoller(CRoller):
    do_check_xml = True
    def __init__(self, lib_path, options, plot_env=None):
        self.dflt_opts = self.get_dflt_opts()

        self.options = dict()
        for opt_dict in self.dflt_opts.values():
            self.options.update(opt_dict)

        extra_keys = options.keys() - self.options.keys()
        if len(extra_keys) > 0:
            raise Exception(f'Unknown options: {extra_keys}')
        self.options.update(options)

        self.obs_dim = 4 + int(self.options['jumping_obs'] == True)
        self.act_dim = 2

        if ('time_before_reset' in self.options) and ('outer_loop_rate' in self.options):
            self.max_steps_per_traj  = self.options['outer_loop_rate']
            self.max_steps_per_traj *= self.options['time_before_reset']
        else:
            self.max_steps_per_traj = None

        if has_gym:
            max_acts = np.array([10.0, 10.0])
            self.action_space = spaces.Box(low=-max_acts, high=max_acts, dtype=np.float64)

            low = [
                self.options['theta_hip_bounds'][0],
                self.options['theta_knee_bounds'][0],
                -self.options['omega_hip_maxabs'],
                -self.options['omega_knee_maxabs'],
            ]
            high = [
                self.options['theta_hip_bounds'][1],
                self.options['theta_knee_bounds'][1],
                self.options['omega_hip_maxabs'],
                self.options['omega_knee_maxabs'],
            ]
            if self.options['obs_history_taps'] is None:
                obs_history_taps = [0]
            else:
                obs_history_taps = self.options['obs_history_taps']
            obs_history_len = len(obs_history_taps)

            low = low * obs_history_len
            high = high * obs_history_len
            if self.options['contact_obs']:
                low.append(0)
                high.append(1)
            if self.options['jumping_obs']:
                low.append(0)
                high.append(1)
            if self.options['extra_obs_dim'] > 0:
                low.append(0)
                high.append(0)
            low = np.hstack(low)
            high = np.hstack(high)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
            #self.reward_range = (-float("inf"), float("inf"))
            self.reward_range = (-1000.0, 0.0)
            self.metadata = {'render.modes': []}

        self.plot_env = plot_env

        bool_cases = {'True': True, 'False': False}
        opt2choics = {'do_obs_noise': bool_cases,
                      'ground_contact_type' : {'compliant_v1':'compliant',
                                               'compliant_v2':'compliant',
                                               'noncompliant':'noncompliant'},
                      'jumping_obs': bool_cases,
                      'action_type': {k: k for k in ('torque', 'jointspace_pd',
                                                     'workspace_pd')},
                      'use_motor_model': bool_cases,
                      'motor_saturation_model': {k: k for k in ('dyno', 'naive', 'legacy')},
                      'use_dyno_model': bool_cases,
                      'use_naive_model': bool_cases,
                      'use_legacy_model': bool_cases,
                      'add_act_sm_r': bool_cases,
                      'do_jump': bool_cases,
                      'timed_jump': bool_cases,
                      'stand_reward': {k: k for k in ('dense', 'sparse', 'simplified')},
                      'check_mj_unstability': bool_cases,
                      'jump_vel_reward': bool_cases,
                      'turn_off_constraints': bool_cases,
                      'contact_obs': bool_cases,
                      'extra_obs_dim': bool_cases,
                      'filter_state': bool_cases,
                      'filter_action': bool_cases,
                      'action_is_delta': bool_cases,
                      'add_torque_penalty_on_air': bool_cases,
                      'add_omega_smoothness_reward': bool_cases,
                      'activation' : {'tanh_activation': 'tanh', 'relu_activation':'relu'},
                      'mjstep_order': {k: k for k in ('mjstep1_after_mjstep',
                                                      'separate_mjstep1_mjstep2',
                                                      'delay_valid_obs')},
                      'xml_type': {'file_path_type': 'file_path', 'content_type': 'content'},
                      '__MAINPROG__': {k: k for k in ('Shared_Obj', 'SimIF_CPP',
                                                      'MlpIF_CPP', 'Rollout_CPP')},
                      'do_mlp_output_tanh': bool_cases,
                     }
        self.opt2choics = opt2choics

        super().__init__(lib_path)

    @staticmethod
    def get_dflt_opts():
        init_opts = {'theta_hip_init': [-3.141592653589793, 0.5235987755982988],
                     'theta_knee_init': [-2.705260340591211, -0.6108652381980153],
                     'omega_hip_init': [0, 0],
                     'omega_knee_init': [0, 0],
                     'pos_slider_init': [0.4, 0.8],
                     'vel_slider_init': [0, 0],
                     'jump_time_bounds': [1.35, 1.35]}

        model_opts = {'ground_contact_type': 'compliant',
                      'omega_hip_maxabs': 100,
                      'omega_knee_maxabs': 100,
                      'knee_minimum_z': 0.02,
                      'hip_minimum_z': 0.05,
                      'maxabs_tau': 10,
                      'maxabs_omega_for_tau': 33.6,
                      'jump_vel_reward': False,
                      'torque_smoothness_coeff': 2000,
                      'constraint_penalty': 100,
                      'turn_off_constraints': False,
                      'contact_obs': False,
                      'extra_obs_dim': 0,
                      'action_is_delta': False,
                      'stand_reward': 'dense',
                      'posture_height': 0.1,
                      'max_reach': 0.28,
                      'time_before_reset': 2,
                      'add_act_sm_r' : False,
                      'do_jump': False,
                      'jumping_obs': False,
                      'timed_jump': True,
                      'jump_push_time': 0.2,
                      'jump_fly_time': 0.2,
                      'max_leg_vel': 100,
                      'jump_vel_coeff': 10000,
                      'do_obs_noise': True,
                      'omega_noise_scale': 1,
                      'use_motor_model': True,
                      'output_torque_scale': 1,
                      'torque_delay': 0.001,
                      'observation_delay': 0.001,
                      'filter_state': False,
                      'filter_action': False,
                      'action_type': 'torque',
                      'hip_kP': 1.25,
                      'hip_kD': 0.05,
                      'knee_kP': 1.25,
                      'knee_kD': 0.05,
                      'outer_loop_rate': 4000,
                      'inner_loop_rate': 4000,
                      'obs_dim': 4,
                      'act_dim': 2}

        mlp_opts = {'h1': 64, 'h2': 64,
                    'activation': 'tanh',
                    'do_mlp_output_tanh': False,
                    'mlp_output_scaling': 1}

        # Options that need [low, high] conversions
        bnd_opts = {'theta_hip_bounds':[-3.9269908169872414, 1.5707963267948966],
                    'theta_knee_bounds': [-2.705260340591211, -0.6108652381980153]}

        # Non-implemented options in C++
        unimplmntd_opts = {'tau_decay_constant': None,
                           'maxabs_desired_joint_angle': None,
                           'filter_state_window': None,
                           'filter_state_numtaps': None,
                           'filter_state_cutoff': None,
                           'filter_action_window': None,
                           'filter_action_numtaps': None,
                           'filter_action_cutoff': None,
                           'a_osc_coeff': None,
                           'a_smooth_coeff': None,
                           'obs_history_taps': None,
                           'work_kP': None,
                           'work_kD': None,
                           'fd_velocities': None}

        # Irrelevant/other options
        irrlvnt_opts = {'slurm_job_file': None}
        wrapper_opts = {'reward_scaling': None}
        other_opts = {'xml_file': osp.join(DFLTXMLDIR, 'leg.xml')}

        dflt_opts = dict(init_opts=init_opts, model_opts=model_opts,
                         bnd_opts=bnd_opts, unimplmntd_opts=unimplmntd_opts,
                         irrlvnt_opts=irrlvnt_opts, other_opts=other_opts,
                         wrapper_opts=wrapper_opts, mlp_opts=mlp_opts)

        return dflt_opts

    def _set_rand_simif_inits(self, traj_num, rng_states):
        if rng_states is not None:
            assert len(rng_states) == traj_num
            np_random = self.np_random_tmp
        else:
            np_random = self.np_random

        r_init, r_init_named = sample_leg_siminit_args(traj_num, self.options,
            np_random, rng_states)
        self.r_init_named = r_init_named
        self._set_simif_inits(r_init.args_double, r_init.args_int)

    def get_timestep(self):
        return 1.0 / self.options['outer_loop_rate']

    def check_build_options(self):
        if not self.lib_get_options_available:
            return

        assert self.cpp_build_opts is not None
        known_opt_types = ['init_opts', 'model_opts', 'bnd_opts',
                           'unimplmntd_opts', 'irrlvnt_opts',
                           'wrapper_opts', 'other_opts', 'mlp_opts']

        for opt_type in self.dflt_opts.keys():
            m_ = f'check rule undefined for "{opt_type}" opt type'
            assert opt_type in known_opt_types, m_

        err_msg_list = []
        for key in self.dflt_opts['model_opts'].keys():
            if key == 'maxabs_omega_for_tau' or (key not in self.cpp_build_opts):
                # the python leg environment ignores the input
                # maxabs_omega_for_tau entirely!
                continue
            m_  = 'The C++ library used a different build option than the '
            m_ += 'one provided to the python binder:\n'
            m_ += f'  C++    Build --> {key}: {self.cpp_build_opts[key]}\n'
            m_ += f'  My   Options --> {key}: {self.options[key]}\n'
            if not(self.options[key] == self.cpp_build_opts[key]):
                err_msg_list.append(m_)

        for key in self.dflt_opts['bnd_opts'].keys():
            if (key not in self.cpp_build_opts):
                continue
            cpp_opt = [self.cpp_build_opts[key+'_low'], self.cpp_build_opts[key+'_high']]
            m_  = 'The C++ library used a different build option than the '
            m_ += 'one provided to the python binder:\n'
            m_ += f'  C++    Build --> {key}: {cpp_opt}\n'
            m_ += f'  My   Options --> {key}: {self.options[key]}\n'
            if not(self.options[key] == cpp_opt):
                err_msg_list.append(m_)

        for key in self.dflt_opts['mlp_opts'].keys():
            cpp_opt = self.cpp_build_opts.get(key, self.dflt_opts['mlp_opts'][key])
            m_  = 'The C++ library used a different build option than the '
            m_ += 'one provided to the python binder:\n'
            m_ += f'  C++    Build --> {key}: {cpp_opt}\n'
            m_ += f'  My   Options --> {key}: {self.options[key]}\n'
            if not(self.options[key] == cpp_opt):
                err_msg_list.append(m_)

        assert tuple(self.dflt_opts['other_opts'].keys()) == ('xml_file',)
        py_xml_path = self.options['xml_file']
        with open(py_xml_path, "r") as fp:
            py_xml_content = "".join(fp.readlines()).strip()

        if self.cpp_build_opts['xml_type'] == 'file_path':
            cpp_xml_path = self.xml_var
            m_ = f'The C++ Build expects an xml file at {cpp_xml_path} which does not exist!'
            assert exists(xml_path), m_
            with open(cpp_xml_path, "r") as fp:
                cpp_xml_content = "".join(fp.readlines()).strip()
        elif self.cpp_build_opts['xml_type'] == 'content':
            cpp_xml_path = None
            cpp_xml_content = self.xml_var.strip()
        else:
            raise Exception(f'Undefined C++ xml_type: %s' % self.cpp_build_opts["xml_type"])
        if py_xml_content != cpp_xml_content:
            m_  = f'The python interface xml file has different content than the one used by C++:\n'
            m_ += f'  -> The Python interface xml file path: {py_xml_path}\n'
            if cpp_xml_path is not None:
                m_ += f'  -> The C++              xml file path: {cpp_xml_path}\n'
            m_ += f'Python XML Content: Length={len(py_xml_content)}\n\n'
            m_ += py_xml_content
            m_ += '\n' + '-'*40 + '\n'
            m_ += f'C++ XML Content: Length={len(cpp_xml_content)}\n\n'
            m_ += cpp_xml_content
            m_ += '\n' + '-'*40 + '\n'
            err_msg_list.append(m_)

        if len(err_msg_list) > 0:
            full_err_msg = f'There were {len(err_msg_list)} errors when '
            full_err_msg += 'matching the compiled file to the input options:\n'
            err_msg_list = [f'Error {i} --> {x}' for i,x in enumerate(err_msg_list)]
            full_err_msg = ('*'*100 + '\n').join(err_msg_list)
            print(full_err_msg)
            raise RuntimeError('C++ Build Options Do Not Matching the Python Options.')

class ToyCRoller(CRoller):
    def __init__(self, lib_path, options):
        self.dflt_opts = self.get_dflt_opts()

        self.options = dict()
        for opt_dict in self.dflt_opts.values():
            self.options.update(opt_dict)

        extra_keys = options.keys() - self.options.keys()
        if len(extra_keys) > 0:
            raise Exception(f'Unknown options: {extra_keys}')
        self.options.update(options)

        self.obs_dim = self.options['obs_dim']
        self.act_dim = self.options['act_dim']
        self.max_steps_per_traj = 200

        bool_cases = {'True': True, 'False': False}
        opt2choics = {'activation' : {'tanh_activation': 'tanh', 'relu_activation':'relu'},
                      '__MAINPROG__': {k: k for k in ('Shared_Obj', 'SimIF_CPP',
                                                      'MlpIF_CPP', 'Rollout_CPP')},
                      'do_mlp_output_tanh': bool_cases}
        self.opt2choics = opt2choics

        if has_gym:
            max_acts = np.array([1.] * self.act_dim)
            self.action_space = spaces.Box(low=-max_acts, high=max_acts, dtype=np.float64)
            max_states = np.array([1.] * self.obs_dim)
            self.observation_space = spaces.Box(low=-max_states, high=max_states, dtype=np.float64)
            self.reward_range = (-1.0, 0.0)
            self.metadata = {'render.modes': []}

        super().__init__(lib_path)

    @staticmethod
    def get_dflt_opts():
        init_opts = {'state_init': [-1., 1.]}
        model_opts = {'obs_dim': 1, 'act_dim': 1}
        mlp_opts = {'h1': 64, 'h2': 64,
                    'activation': 'tanh',
                    'do_mlp_output_tanh': False,
                    'mlp_output_scaling': 1}

        # Options that need [low, high] conversions
        bnd_opts = {}

        # Non-implemented options in C++
        unimplmntd_opts = {}

        # Irrelevant/other options
        irrlvnt_opts = {}
        wrapper_opts = {'reward_scaling': None}
        other_opts = {'xml_file': None}

        dflt_opts = dict(init_opts=init_opts, model_opts=model_opts,
                         bnd_opts=bnd_opts, unimplmntd_opts=unimplmntd_opts,
                         irrlvnt_opts=irrlvnt_opts, other_opts=other_opts,
                         wrapper_opts=wrapper_opts, mlp_opts=mlp_opts)

        return dflt_opts

    def _set_rand_simif_inits(self, traj_num, rng_states):
        if rng_states is not None:
            assert len(rng_states) == traj_num
            np_random = self.np_random_tmp
        else:
            rng_states = [None] * traj_num
            np_random = self.np_random

        # SimIF Variables
        ini_states = np.zeros((traj_num,), dtype=np.float64)
        for i, rng_state in enumerate(rng_states):
            if rng_state is not None:
                np_random.set_state(rng_state)
            ini_states[i] = np_random.uniform(-1.0, 1.0)
        # These must have the same order as specified in SimIF.cpp under
        # double* SimInterface::reset(double* init_args_double, int* init_args_int)
        init_args_double = np.stack((ini_states,), axis=1)
        init_args_int = np.zeros((traj_num, 0), np.int32)
        r_init = SimInitArgs(init_args_double, init_args_int)
        self.r_init_named = dict(ini_states=ini_states)
        self._set_simif_inits(r_init.args_double, r_init.args_int)

    def get_timestep(self):
        return 0.01

class GymMujCRoller(CRoller):
    do_check_xml = True
    def __init__(self, lib_path, options, plot_env=None):
        self.dflt_opts = self.get_dflt_opts()

        self.options = dict()
        for opt_dict in self.dflt_opts.values():
            self.options.update(opt_dict)

        extra_keys = options.keys() - self.options.keys()
        if len(extra_keys) > 0:
            raise Exception(f'Unknown options: {extra_keys}')
        self.options.update(options)
        self.preproc_opts(self.options)

        self.act_dim = self.options['act_dim']
        self.obs_dim = self.options['obs_dim']

        if ('time_before_reset' in self.options) and ('outer_loop_rate' in self.options):
            self.max_steps_per_traj  = self.options['outer_loop_rate']
            self.max_steps_per_traj *= self.options['time_before_reset']
            self.max_steps_per_traj = int(self.max_steps_per_traj)
        else:
            self.max_steps_per_traj = None

        if has_gym:
            low_acts = np.array([self.options['action_low']] * self.act_dim)
            high_acts = np.array([self.options['action_high']] * self.act_dim)
            self.action_space = spaces.Box(low=low_acts, high=high_acts, dtype=np.float64)
            obs_low = np.full((self.obs_dim,), -float('inf'))
            obs_high = np.full((self.obs_dim,), float('inf'))
            self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)
            self.reward_range = (-float('inf'), float('inf'))
            self.metadata = {'render.modes': []}

        self.plot_env = plot_env

        bool_cases = {'True': True, 'False': False}
        opt2choics = {'action_type': {k: k for k in ('torque', 'jointspace_pd',
                                                     'workspace_pd')},
                      'check_mj_unstability': bool_cases,
                      'agg_inner_rewards': {k: k for k in ('take_average', 'take_last')},
                      'do_obs_noise': bool_cases,
                      'xml_type': {'file_path_type': 'file_path', 'content_type': 'content'},
                      'activation' : {'tanh_activation': 'tanh', 'relu_activation':'relu'},
                      'do_mlp_output_tanh': bool_cases,
                      'mjstep_order': {k: k for k in ('mjstep1_after_mjstep',
                                                      'separate_mjstep1_mjstep2',
                                                      'delay_valid_obs',
                                                      'only_mjstep')},
                      '__MAINPROG__': {k: k for k in ('Shared_Obj', 'SimIF_CPP',
                                                      'MlpIF_CPP', 'Rollout_CPP')},
                     }
        self.opt2choics = opt2choics

        super().__init__(lib_path)

    @classmethod
    def get_dflt_opts(cls):
        init_opts = {'init_dqvel_dist' : 'normal',
                     'reset_noise_scale' : 0.1}

        model_opts = {'do_obs_noise': False,
                      'agg_inner_rewards': 'take_last',
                      'torque_delay': 0.0,
                      'observation_delay': 0.0,
                      'action_type': 'torque',
                      'kP': 1.25,
                      'kD': 0.05,
                      'sparse_rew_outer_steps': 1,
                      'sparse_rew_outer_discrate': 1.0}

        mlp_opts = {'h1': 64, 'h2': 64,
                    'activation': 'tanh',
                    'do_mlp_output_tanh': False,
                    'mlp_output_scaling': 1}

        # Options that need [low, high] conversions
        bnd_opts = {}

        # Non-implemented options in C++
        unimplmntd_opts = {}

        # Irrelevant/other options
        irrlvnt_opts = {}
        wrapper_opts = {'reward_scaling': None, 'action_low': None,
                        'action_high': None}
        other_opts = {'xml_file': None}

        dflt_opts = dict(init_opts=init_opts, model_opts=model_opts,
                         bnd_opts=bnd_opts, unimplmntd_opts=unimplmntd_opts,
                         irrlvnt_opts=irrlvnt_opts, other_opts=other_opts,
                         wrapper_opts=wrapper_opts, mlp_opts=mlp_opts)

        return dflt_opts

    @classmethod
    def preproc_opts(cls, options):
        outer_loop_rate = options['outer_loop_rate']
        for opt in ('torque_delay', 'observation_delay'):
            optval = options.get(opt, None)
            if isinstance(optval, dict):
                optval = dict(optval) # shallow copy
                unit = optval.pop('unit', 'second')
                optlen = optval.pop('len', 0)
                assert len(optval) == 0, f'Unknown keys: {optval.keys()}'
                assert unit in ('step', 'second')
                if unit == 'step':
                    val_sec = optlen / outer_loop_rate
                elif unit == 'second':
                    val_sec = optlen
                else:
                    raise ValueError(f'Unknwon unit {unit}')
                options[opt] = val_sec

    def get_timestep(self):
        return 1.0 / self.options['outer_loop_rate']

    def _set_rand_simif_inits(self, traj_num, rng_states):
        reset_noise_scale = self.options['reset_noise_scale']
        qpos_dim = self.options['qpos_dim']
        init_dqvel_dist = self.options['init_dqvel_dist']
        qvel_dim = self.options['qvel_dim']

        if rng_states is not None:
            assert len(rng_states) == traj_num
            np_random = self.np_random_tmp
        else:
            rng_states = [None] * traj_num
            np_random = self.np_random

        # SimIF Variables
        # These must have the same order as specified in SimIF.cpp under
        # double* SimInterface::reset(double* init_args_double, int* init_args_int)

        ##### The following will make sure the init states are
        ##### initialized the same way as python's gym.
        iad_list = []
        for i, rng_state in enumerate(rng_states):
            if rng_state is not None:
                np_random.set_state(rng_state)
            init_dqpos = np_random.uniform(low=-reset_noise_scale,
                high=reset_noise_scale, size=(1, qpos_dim)).astype(np.float64)
            if init_dqvel_dist == "normal":
                init_dqvel = (np_random.randn(1, qvel_dim) *
                              reset_noise_scale).astype(np.float64)
            elif init_dqvel_dist == "uniform":
                init_dqvel = np_random.uniform(low=-reset_noise_scale,
                    high=reset_noise_scale, size=(1, qvel_dim)).astype(np.float64)
            else:
                raise ValueError(f'{init_dqvel_dist} not implemented')
            iad_list.append(np.concatenate((init_dqpos, init_dqvel), axis=1))
        init_args_double = np.concatenate(iad_list, axis=0)

        init_args_int = np.zeros((traj_num, 0), np.int32)
        r_init = SimInitArgs(init_args_double, init_args_int)
        self.r_init_named = dict(delta_qposvel=init_args_double)
        self._set_simif_inits(r_init.args_double, r_init.args_int)

    def _set_rand_simif_inits_batch(self, traj_num):
        reset_noise_scale = self.options['reset_noise_scale']
        qpos_dim = self.options['qpos_dim']
        init_dqvel_dist = self.options['init_dqvel_dist']
        qvel_dim = self.options['qvel_dim']
        np_random = self.np_random

        # This is the same as _set_rand_simif_inits in terms of functionality,
        # except it samples init_dqpos and init_dqvel in batch. If traj_num==1,
        # then this implementation should yield the same values as
        # _set_rand_simif_inits. For traj_num > 1, the two implementations will
        # not be synced anymore, but they will follow the same distribution,

        # SimIF Variables
        # These must have the same order as specified in SimIF.cpp under
        # double* SimInterface::reset(double* init_args_double, int* init_args_int)
        init_dqpos = np_random.uniform(low=-reset_noise_scale,
            high=reset_noise_scale, size=(traj_num, qpos_dim)).astype(np.float64)

        if init_dqvel_dist == "normal":
            init_dqvel = (np_random.randn(traj_num, qvel_dim) * reset_noise_scale).astype(np.float64)
        elif init_dqvel_dist == "uniform":
            init_dqvel = np_random.uniform(low=-reset_noise_scale,
                high=reset_noise_scale, size=(traj_num, qvel_dim)).astype(np.float64)
        else:
            raise ValueError(f'{init_dqvel_dist} not implemented')
        init_args_double = np.concatenate((init_dqpos, init_dqvel), axis=1)

        init_args_int = np.zeros((traj_num, 0), np.int32)
        r_init = SimInitArgs(init_args_double, init_args_int)
        self.r_init_named = dict(delta_qposvel=init_args_double)
        self._set_simif_inits(r_init.args_double, r_init.args_int)

class InvertedPendulumCRoller(GymMujCRoller):
    @classmethod
    def get_dflt_opts(cls):
        dflt_opts = GymMujCRoller.get_dflt_opts()

        modelopts = {'outer_loop_rate': 25,
                     'inner_loop_rate': 50,
                     'physics_timestep': 0.02,
                     'time_before_reset': 40,
                     'qpos_dim': 2,
                     'qvel_dim': 2,
                     'ctrl_dim': 1,
                     'obs_dim': 4,
                     'act_dim': 1}

        initopts = {'init_dqvel_dist': 'uniform',
                    'reset_noise_scale': 0.01}

        wrapperopts = {'action_low': -3.0, 'action_high': 3.0}

        otheropts = {'xml_file': osp.join(DFLTXMLDIR, 'InvertedPendulum.xml')}

        dflt_opts['model_opts'].update(modelopts)
        dflt_opts['init_opts'].update(initopts)
        dflt_opts['wrapper_opts'].update(wrapperopts)
        dflt_opts['other_opts'].update(otheropts)

        return dflt_opts

class InvertedDoublePendulumCRoller(GymMujCRoller):
     @classmethod
     def get_dflt_opts(cls):
         dflt_opts = GymMujCRoller.get_dflt_opts()

         modelopts = {'outer_loop_rate': 20,
                      'inner_loop_rate': 100,
                      'physics_timestep': 0.01,
                      'time_before_reset': 50,
                      'qpos_dim': 3,
                      'qvel_dim': 3,
                      'ctrl_dim': 1,
                      'obs_dim': 11,
                      'act_dim': 1}

         initopts = {}

         wrapperopts = {'action_low': -1.0, 'action_high': 1.0}

         otheropts = {'xml_file': osp.join(DFLTXMLDIR, 'InvertedDoublePendulum.xml')}

         dflt_opts['model_opts'].update(modelopts)
         dflt_opts['init_opts'].update(initopts)
         dflt_opts['wrapper_opts'].update(wrapperopts)
         dflt_opts['other_opts'].update(otheropts)

         return dflt_opts

class HalfCheetahCRoller(GymMujCRoller):
     @classmethod
     def get_dflt_opts(cls):
         dflt_opts = GymMujCRoller.get_dflt_opts()

         modelopts = {'outer_loop_rate': 20,
                      'inner_loop_rate': 100,
                      'physics_timestep': 0.01,
                      'time_before_reset': 50,
                      'qpos_dim': 9,
                      'qvel_dim': 9,
                      'ctrl_dim': 6,
                      'forward_reward_weight': 1.,
                      'ctrl_cost_weight': 0.1,
                      'obs_dim': 17,
                      'act_dim': 6}

         initopts = {}

         wrapperopts = {'action_low': -1.0, 'action_high': 1.0}

         otheropts = {'xml_file': osp.join(DFLTXMLDIR, 'HalfCheetah.xml')}

         dflt_opts['model_opts'].update(modelopts)
         dflt_opts['init_opts'].update(initopts)
         dflt_opts['wrapper_opts'].update(wrapperopts)
         dflt_opts['other_opts'].update(otheropts)

         return dflt_opts

class SwimmerCRoller(GymMujCRoller):
    @classmethod
    def get_dflt_opts(cls):
        dflt_opts = GymMujCRoller.get_dflt_opts()

        modelopts = {'outer_loop_rate': 25,
                     'inner_loop_rate': 100,
                     'physics_timestep': 0.01,
                     'time_before_reset': 40,
                     'qpos_dim': 5,
                     'qvel_dim': 5,
                     'ctrl_dim': 2,
                     'forward_reward_weight': 1.,
                     'ctrl_cost_weight': 0.0001,
                     'obs_dim': 8,
                     'act_dim': 2}

        initopts = {'init_dqvel_dist': 'uniform'}

        wrapperopts = {'action_low': -1.0, 'action_high': 1.0}

        otheropts = {'xml_file': osp.join(DFLTXMLDIR, 'Swimmer.xml')}

        dflt_opts['model_opts'].update(modelopts)
        dflt_opts['init_opts'].update(initopts)
        dflt_opts['wrapper_opts'].update(wrapperopts)
        dflt_opts['other_opts'].update(otheropts)

        return dflt_opts

class HopperCRoller(GymMujCRoller):
    @classmethod
    def get_dflt_opts(cls):
        dflt_opts = GymMujCRoller.get_dflt_opts()

        modelopts = {'outer_loop_rate': 125,
                      'inner_loop_rate': 500,
                      'physics_timestep': 0.002,
                      'time_before_reset': 8,
                      'qpos_dim': 6,
                      'qvel_dim': 6,
                      'ctrl_dim': 3,
                      'forward_reward_weight': 1.,
                      'ctrl_cost_weight': 0.001,
                      'healthy_reward_weight': 1.0,
                      'terminate_when_unhealthy': True,
                      'healthy_state_range_low': -100.0,
                      'healthy_state_range_high': 100.0,
                      'healthy_z_range_low': 0.7,
                      'healthy_angle_range_low': -0.2,
                      'healthy_angle_range_high': 0.2,
                      'obs_dim': 11,
                      'act_dim': 3}

        initopts = {'init_dqvel_dist': 'uniform',
                    'reset_noise_scale': 0.005}

        wrapperopts = {'action_low': -1.0, 'action_high': 1.0}

        otheropts = {'xml_file': osp.join(DFLTXMLDIR, 'Hopper.xml')}

        dflt_opts['model_opts'].update(modelopts)
        dflt_opts['init_opts'].update(initopts)
        dflt_opts['wrapper_opts'].update(wrapperopts)
        dflt_opts['other_opts'].update(otheropts)

        return dflt_opts

class Walker2dCRoller(GymMujCRoller):
    @classmethod
    def get_dflt_opts(cls):
        dflt_opts = GymMujCRoller.get_dflt_opts()

        modelopts = {'outer_loop_rate': 125,
                     'inner_loop_rate': 500,
                     'physics_timestep': 0.002,
                     'time_before_reset': 8,
                     'qpos_dim': 9,
                     'qvel_dim': 9,
                     'ctrl_dim': 6,
                     'forward_reward_weight': 1.,
                     'ctrl_cost_weight': 0.001,
                     'healthy_reward_weight': 1.0,
                     'terminate_when_unhealthy': True,
                     'healthy_z_range_low': 0.8,
                     'healthy_z_range_high': 2.0,
                     'healthy_angle_range_low': -1.0,
                     'healthy_angle_range_high': 1.0,
                     'obs_dim': 17,
                     'act_dim': 6}

        initopts = {'init_dqvel_dist': 'uniform',
                    'reset_noise_scale': 0.005}

        wrapperopts = {'action_low': -1.0, 'action_high': 1.0}

        otheropts = {'xml_file': osp.join(DFLTXMLDIR, 'Walker2d.xml')}

        dflt_opts['model_opts'].update(modelopts)
        dflt_opts['init_opts'].update(initopts)
        dflt_opts['wrapper_opts'].update(wrapperopts)
        dflt_opts['other_opts'].update(otheropts)

        return dflt_opts

class AntCRoller(GymMujCRoller):
    @classmethod
    def get_dflt_opts(cls):
        dflt_opts = GymMujCRoller.get_dflt_opts()

        modelopts = {'outer_loop_rate': 20,
                     'inner_loop_rate': 100,
                     'physics_timestep': 0.01,
                     'time_before_reset': 50,
                     'use_contact_forces': True,
                     'qpos_dim': 15,
                     'qvel_dim': 14,
                     'ctrl_dim': 8,
                     'body_dim': 14,
                     'forward_reward_weight': 1,
                     'ctrl_cost_weight': 0.5,
                     'contact_cost_weight': 0.0005,
                     'healthy_reward_weight': 1.0,
                     'terminate_when_unhealthy': True,
                     'healthy_z_range_low': 0.2,
                     'healthy_z_range_high': 1.0,
                     'contact_force_range_low': -1.0,
                     'contact_force_range_high': 1.0,
                     'obs_dim': 111,
                     'act_dim': 8}

        initopts = {'init_dqvel_dist': 'normal',
                    'reset_noise_scale': 0.1}

        wrapperopts = {'action_low': -1.0, 'action_high': 1.0}

        otheropts = {'xml_file': osp.join(DFLTXMLDIR, 'Ant.xml')}

        dflt_opts['model_opts'].update(modelopts)
        dflt_opts['init_opts'].update(initopts)
        dflt_opts['wrapper_opts'].update(wrapperopts)
        dflt_opts['other_opts'].update(otheropts)

        return dflt_opts

class HumanoidCRoller(GymMujCRoller):
    @classmethod
    def get_dflt_opts(cls):
        dflt_opts = GymMujCRoller.get_dflt_opts()

        modelopts = {'outer_loop_rate': 66.66666666666666,
                     'inner_loop_rate': 333.3333333333333,
                     'physics_timestep': 0.003,
                     'time_before_reset': 15,
                     'use_contact_forces': True,
                     'qpos_dim': 24,
                     'qvel_dim': 23,
                     'ctrl_dim': 17,
                     'body_dim': 14,
                     'forward_reward_weight': 1.25,
                     'ctrl_cost_weight': 0.1,
                     'contact_cost_weight': 0.0000005,
                     'contact_cost_range_high': 10.0,
                     'healthy_reward_weight': 5.0,
                     'terminate_when_unhealthy': True,
                     'healthy_z_range_low': 1.0,
                     'healthy_z_range_high': 2.0,
                     'obs_dim': 376,
                     'act_dim': 17}

        initopts = {'init_dqvel_dist': 'uniform',
                    'reset_noise_scale': 0.01}

        wrapperopts = {'action_low': -1.0, 'action_high': 1.0}

        otheropts = {'xml_file': osp.join(DFLTXMLDIR, 'Humanoid.xml')}

        dflt_opts['model_opts'].update(modelopts)
        dflt_opts['init_opts'].update(initopts)
        dflt_opts['wrapper_opts'].update(wrapperopts)
        dflt_opts['other_opts'].update(otheropts)

        return dflt_opts

class HumanoidStandupCRoller(GymMujCRoller):
    @classmethod
    def get_dflt_opts(cls):
        dflt_opts = GymMujCRoller.get_dflt_opts()

        modelopts = {'outer_loop_rate': 66.66666666666666,
                     'inner_loop_rate': 333.3333333333333,
                     'physics_timestep': 0.003,
                     'time_before_reset': 15,
                     'use_contact_forces': True,
                     'qpos_dim': 24,
                     'qvel_dim': 23,
                     'ctrl_dim': 17,
                     'body_dim': 14,
                     'forward_reward_weight': 1.0,
                     'ctrl_cost_weight': 0.1,
                     'contact_cost_weight': 0.0000005,
                     'contact_cost_range_high': 10.0,
                     'obs_dim': 376,
                     'act_dim': 17}

        initopts = {'init_dqvel_dist': 'uniform',
                    'reset_noise_scale': 0.01}

        wrapperopts = {'action_low': -0.4, 'action_high': 0.4}

        otheropts = {'xml_file': osp.join(DFLTXMLDIR, 'HumanoidStandup.xml')}

        dflt_opts['model_opts'].update(modelopts)
        dflt_opts['init_opts'].update(initopts)
        dflt_opts['wrapper_opts'].update(wrapperopts)
        dflt_opts['other_opts'].update(otheropts)

        return dflt_opts

class ReacherCRoller(GymMujCRoller):
    @classmethod
    def get_dflt_opts(cls):
        dflt_opts = GymMujCRoller.get_dflt_opts()

        modelopts = {'outer_loop_rate': 50,
                     'inner_loop_rate': 100,
                     'physics_timestep': 0.01,
                     'time_before_reset': 1,
                     'qpos_dim': 4,
                     'qvel_dim': 4,
                     'ctrl_dim': 2,
                     'obs_dim': 11,
                     'act_dim': 2}

        initopts = {'init_dqvel_dist': 'uniform',
                    'reset_noise_scale': 0.1}

        wrapperopts = {'action_low': -1.0, 'action_high': 1.0}

        otheropts = {'xml_file': osp.join(DFLTXMLDIR, 'Reacher.xml')}

        dflt_opts['model_opts'].update(modelopts)
        dflt_opts['init_opts'].update(initopts)
        dflt_opts['wrapper_opts'].update(wrapperopts)
        dflt_opts['other_opts'].update(otheropts)

        return dflt_opts

    def _set_rand_simif_inits(self, traj_num, rng_states):
        reset_noise_scale = self.options['reset_noise_scale']
        qpos_dim = self.options['qpos_dim']
        init_dqvel_dist = self.options['init_dqvel_dist']
        qvel_dim = self.options['qvel_dim']

        if rng_states is not None:
            assert len(rng_states) == traj_num
            np_random = self.np_random_tmp
        else:
            rng_states = [None] * traj_num
            np_random = self.np_random

        # SimIF Variables
        # These must have the same order as specified in SimIF.cpp under
        # double* SimInterface::reset(double* init_args_double, int* init_args_int)

        ##### The following will make sure the init states are
        ##### initialized the same way as python's gym.
        iad_list = []
        for i, rng_state in enumerate(rng_states):
            if rng_state is not None:
                np_random.set_state(rng_state)

            init_dqpos = np_random.uniform(low=-reset_noise_scale,
                high=reset_noise_scale, size=(1, qpos_dim)).astype(np.float64)

            while True:
                goal = np_random.uniform(low=-2*reset_noise_scale,
                    high=2*reset_noise_scale, size=2)
                if np.linalg.norm(goal) < 2*reset_noise_scale:
                    # subtracting init_qpos to compute init_dqpos
                    dgoal = goal - np.array([0.1, -0.1])
                    break
            init_dqpos[0, -2:] = dgoal

            if init_dqvel_dist == "normal":
                init_dqvel = (np_random.randn(1, qvel_dim) * reset_noise_scale
                              * 0.05).astype(np.float64)
            elif init_dqvel_dist == "uniform":
                init_dqvel = np_random.uniform(low=-reset_noise_scale * 0.05,
                    high=reset_noise_scale * 0.05, size=(1, qvel_dim)).astype(np.float64)
            else:
                raise ValueError(f'{init_dqvel_dist} not implemented')
            init_dqvel[0, -2:] = 0 # init_qvel[-2:] is zero, so no correction is needed.

            iad_list.append(np.concatenate((init_dqpos, init_dqvel), axis=1))
        init_args_double = np.concatenate(iad_list, axis=0)

        init_args_int = np.zeros((traj_num, 0), np.int32)
        r_init = SimInitArgs(init_args_double, init_args_int)
        self.r_init_named = dict(delta_qposvel=init_args_double)
        self._set_simif_inits(r_init.args_double, r_init.args_int)

class PendulumCRoller(GymMujCRoller):
    do_check_xml = False

    @classmethod
    def get_dflt_opts(cls):
        dflt_opts = GymMujCRoller.get_dflt_opts()

        modelopts = {'outer_loop_rate': 20,
                     'inner_loop_rate': 20,
                     'physics_timestep': 0.05,
                     'time_before_reset': 10,
                     'ctrl_dim': 1,
                     'min_torque': -2.0,
                     'max_torque': 2.0,
                     'min_speed': -8.0,
                     'max_speed': 8.0,
                     'gravity_acc': 10.0,
                     'link_mass': 1.0,
                     'link_length': 1.0,
                     'obs_dim': 3,
                     'act_dim': 1,
                     'sparse_rew_outer_steps': 1,
                     'sparse_rew_outer_discrate': 1.0,
                     'stft_inner_steps': 200,
                     'stft_gamma': 0.99}

        initopts = {'reset_noise_scale': 1.0}

        wrapperopts = {'action_low': -2.0, 'action_high': 2.0}

        dflt_opts['model_opts'].update(modelopts)
        dflt_opts['init_opts'].update(initopts)
        dflt_opts['wrapper_opts'].update(wrapperopts)

        return dflt_opts

    def _set_rand_simif_inits(self, traj_num, rng_states):
        reset_noise_scale = self.options['reset_noise_scale']

        if rng_states is not None:
            assert len(rng_states) == traj_num
            np_random = self.np_random_tmp
        else:
            rng_states = [None] * traj_num
            np_random = self.np_random

        # SimIF Variables
        # These must have the same order as specified in SimIF.cpp under
        # double* SimInterface::reset(double* init_args_double, int* init_args_int)
        a = np.array([np.pi, 1])
        low = -reset_noise_scale * a
        high = reset_noise_scale * a

        init_args_double = np.zeros((traj_num, 2), np.float64)
        for i, rng_state in enumerate(rng_states):
            if rng_state is not None:
                np_random.set_state(rng_state)
            init_args_double[i, :] = np_random.uniform(low=low, high=high, size=2)

        init_args_int = np.zeros((traj_num, 0), np.int32)
        r_init = SimInitArgs(init_args_double, init_args_int)
        self.r_init_named = dict(delta_qposvel=init_args_double)
        self._set_simif_inits(r_init.args_double, r_init.args_int)

class NLRPendulumCRoller(GymMujCRoller):
    do_check_xml = False

    @classmethod
    def get_dflt_opts(cls):
        dflt_opts = GymMujCRoller.get_dflt_opts()

        modelopts = {'outer_loop_rate': 20,
                     'inner_loop_rate': 20,
                     'physics_timestep': 0.05,
                     'time_before_reset': 10.0,
                     'ctrl_dim': 1,
                     'min_torque': -80.0,
                     'max_torque': 80.0,
                     'min_speed': -8.0,
                     'max_speed': 8.0,
                     'gravity_acc': 10.0,
                     'link_mass': 1.0,
                     'link_length': 1.0,
                     'des_period': 0.5,
                     'des_offset': float(np.pi * 2. / 12.),
                     'des_amp': 0.2,
                     'obs_dim': 3,
                     'act_dim': 1,
                     'sparse_rew_outer_steps': 1,
                     'sparse_rew_outer_discrate': 1.0,
                     'stft_inner_steps': 200,
                     'stft_gamma': 0.99}

        initopts = {'reset_noise_scale': 1.0}

        wrapperopts = {'action_low': -80.0, 'action_high': 80.0}

        dflt_opts['model_opts'].update(modelopts)
        dflt_opts['init_opts'].update(initopts)
        dflt_opts['wrapper_opts'].update(wrapperopts)

        return dflt_opts

    def _set_rand_simif_inits(self, traj_num, rng_states):
        reset_noise_scale = self.options['reset_noise_scale']

        if rng_states is not None:
            assert len(rng_states) == traj_num
            np_random = self.np_random_tmp
        else:
            rng_states = [None] * traj_num
            np_random = self.np_random

        # SimIF Variables
        # These must have the same order as specified in SimIF.cpp under
        # double* SimInterface::reset(double* init_args_double, int* init_args_int)
        high = np.array([np.pi + 0.1 * reset_noise_scale,  0.02 * reset_noise_scale])
        low  = np.array([np.pi - 0.1 * reset_noise_scale, -0.02 * reset_noise_scale])

        init_args_double = np.zeros((traj_num, 2), np.float64)
        for i, rng_state in enumerate(rng_states):
            if rng_state is not None:
                np_random.set_state(rng_state)
            init_args_double[i, :] = np_random.uniform(low=low, high=high, size=2)

        init_args_int = np.zeros((traj_num, 0), np.int32)
        r_init = SimInitArgs(init_args_double, init_args_int)
        self.r_init_named = dict(delta_qposvel=init_args_double)
        self._set_simif_inits(r_init.args_double, r_init.args_int)

registry = {'Pendulum': PendulumCRoller,
            'NLRPendulum': NLRPendulumCRoller,
            'InvertedPendulum': InvertedPendulumCRoller,
            'InvertedDoublePendulum': InvertedDoublePendulumCRoller,
            'HalfCheetah': HalfCheetahCRoller,
            'Swimmer': SwimmerCRoller,
            'Hopper': HopperCRoller,
            'Walker2d': Walker2dCRoller,
            'Ant': AntCRoller,
            'Humanoid': HumanoidCRoller,
            'HumanoidStandup': HumanoidStandupCRoller,
            'Reacher': ReacherCRoller,
            'leg': LegCRoller,
            'toy': ToyCRoller}

def make(name, options):
    if name not in registry:
        raise ValueError(f'Unknown environment {name}')
    CRollerCls = registry[name]
    ref_name = name
    options = options or dict()
    optionspxml = dict(options)

    variants_lst = os.listdir(osp.join(CGYMDIR, 'share', ref_name))
    compatible_variant = None
    env_collection = osp.join(CGYMDIR, 'share', ref_name)

    for variant in variants_lst:
        var_json = osp.join(env_collection, variant, 'options.json')
        with open(var_json, 'r') as fp:
            var_opts = json.load(fp)
        if var_opts.get('xml_file', None) is not None:
            xml_fullp = osp.join(env_collection, variant, var_opts['xml_file'])
            var_opts['xml_file'] = xml_fullp
            if 'xml_file' not in options:
                optionspxml['xml_file'] = xml_fullp
        clash_lst = CRollerCls.get_clashing_opts(var_opts, optionspxml)
        if len(clash_lst) == 0:
            compatible_variant = variant
            break

    msg_  = f'Could not find matching builds for your options.\n'
    msg_ += f'Please compile the environemnt with these options, and '
    msg_ += f'add it to the share directory in cgym.'
    assert compatible_variant is not None, msg_

    lib_path = osp.join(env_collection, variant, 'libRollout.so')
    environment = CRollerCls(lib_path, optionspxml)
    return environment

if __name__ == '__main__':
    import torch
    import random
    #############################################################
    ################### Gym Version Checking ####################
    #############################################################
    assert has_gym, 'cannot perform match testing without gym'
    tested_gym_version = '0.15.3'
    if gym.__version__ != tested_gym_version:
        print(f'This code was tested to produce identical ' +
              f'values to gym=={tested_gym_version}.')

    #############################################################
    ############## Unit Testing Utility Functions ###############
    #############################################################
    class PrintTime(object):
        def __init__(self, p_str):
            self.st_time = None
            self.p_str = p_str

        def __enter__(self):
            print(f'Starting {self.p_str}')
            self.st_time = time.time()
            return self

        def __exit__(self, type, value, traceback):
            end_time = time.time()
            print(f'Finished {self.p_str}')
            print(f'Total Time: {end_time - self.st_time}')
            print('-'*20)

    def printout(intro, np_arr, intro_min_len = 18):
        intro = intro + ' = '
        intro = intro + ' ' * max(0, intro_min_len-len(intro))
        print(intro + str(np_arr.reshape(-1)).replace('\n', '\n' + ' '*len(intro)))
        print('-'*intro_min_len)

    class TypicalActor(torch.nn.Module):
        def __init__(self, observation_dim, action_dim):
            super().__init__()
            self.fc1 = torch.nn.Linear(observation_dim, 64, bias=True).double()
            self.fc2 = torch.nn.Linear(64, 64, bias=True).double()
            self.fc3_mu = torch.nn.Linear(64, action_dim, bias=True).double()

        def forward(self, x):
            """
            Takes observation vector x and returns a vector mu and a vector std.
            x: state observation
            mu: mean of action distribution
            std: standard deviation of action distribution
            """
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            mu = self.fc3_mu(x)

            return mu

    #############################################################
    ######### Registering Non-locally Rewarded Pendulums ########
    #############################################################
    from gym.envs.registration import register
    import sys
    sys.path.insert(-1, '..')
    for v in range(1, 9):
        entry_point = f'opt.nlrpendulum:NLRPendulum_v{v}'
        register(id=f'NLRPendulum-v{v}', entry_point=entry_point)

    #############################################################
    ################ Performing Matching Checks #################
    #############################################################

    a = [('Pendulum', PendulumCRoller, 10, 0),
         ('NLRPendulum', NLRPendulumCRoller, 10, 1),
         ('InvertedPendulum', InvertedPendulumCRoller, 500, 2),
         ('InvertedDoublePendulum', InvertedDoublePendulumCRoller, 500, 2),
         ('HalfCheetah', HalfCheetahCRoller, 10, 2),
         ('Swimmer', SwimmerCRoller, 10, 2),
         ('Hopper', HopperCRoller, 100, 2),
         ('Walker2d', Walker2dCRoller, 500, 2),
         ('Ant', AntCRoller, 10, 2),
         ('Humanoid', HumanoidCRoller, 100, 2),
         ('HumanoidStandup', HumanoidStandupCRoller, 1, 2),
         ('Reacher', ReacherCRoller, 100, 2),
         ]

    for envname, CRollerCLS, n_trajs, gymver in a:
        env_cpp = make(envname, options=dict())
        env_py = gym.make(f'{envname}-v{gymver}')

        outer_loop_rate = env_cpp.options['outer_loop_rate']
        time_before_reset = env_cpp.options['time_before_reset']
        obs_dim = env_cpp.obs_dim
        act_dim = env_cpp.act_dim
        n_steps = int(outer_loop_rate * time_before_reset)

        ################### Creating an Initial Policy ######################
        # Performing global seeding to make sure the actor is reproducible.
        np.random.seed(12345)
        random.seed(12345)
        torch.manual_seed(12345)

        policy = TypicalActor(observation_dim=obs_dim, action_dim=act_dim)
        with torch.no_grad():
            policy.fc3_mu.weight.data *= 0.1
            policy.fc3_mu.bias.data *= 0.1

        ################### Running the C++ Trajectory ######################
        env_cpp.seed(12345)
        env_cpp.set_torch_policy(policy)
        with PrintTime(f'{n_trajs} C++ {envname} Rollout') as print_obj:
            greedy_outdict = env_cpp.greedy(n_trajs, n_steps)
        s_cpp = greedy_outdict['observations'][-1]
        a_cpp = greedy_outdict['actions'][-1]
        r_cpp = greedy_outdict['rewards'][-1]

        ################### Running the Python Trajectory ###################
        np.random.seed(12345) # SimInterface uses np.random as RNG
        random.seed(12345)
        torch.manual_seed(12345)
        env_py.seed(12345)

        with PrintTime(f'{n_trajs} Python {envname} Rollouts') as print_obj:
            for _ in range(n_trajs):
                s_py = np.zeros((n_steps, obs_dim))
                a_py = np.zeros((n_steps, act_dim))
                r_py = np.zeros(n_steps)
                s_next = env_py.reset()
                for t in range(n_steps):
                    s_py[t,:] = s_next
                    with torch.no_grad():
                        a_py[t,:] = policy(torch.from_numpy(s_next)).numpy()
                    s_next, r_py[t], done, info = env_py.step(a_py[t,:])
                    if done:
                        break

        ################# Comparting Python & C++ Trajectories ##################
        def max_diff(cpp_arr, py_arr):
            return np.abs(cpp_arr - py_arr).max()
        def mean_abs_diff(cpp_arr, py_arr):
            return np.abs(cpp_arr - py_arr).mean()
        print(f'Maximum       Observation Difference: %0.10f' % max_diff(s_cpp, s_py))
        print(f'Maximum       Action      Difference: %0.10f' % max_diff(a_cpp, a_py))
        print(f'Maximum       Reward      Difference: %0.10f' % max_diff(r_cpp, r_py))
        print(f'Mean Absolute Observation Difference: %0.10f' % mean_abs_diff(s_cpp, s_py))
        print(f'Mean Absolute Action      Difference: %0.10f' % mean_abs_diff(a_cpp, a_py))
        print(f'Mean Absolute Reward      Difference: %0.10f' % mean_abs_diff(r_cpp, r_py))
        print(('='*120 + '\n') * 3, end='')

    #############################################################
    ############## Instantiating a Toy C++ Roller ###############
    #############################################################
    import sys
    sys.exit(0)

    cleg_dir = dirname(dirname(abspath(__file__)))

    policy = TypicalActor(observation_dim=1, action_dim=1)
    with torch.no_grad():
        policy.fc1.weight.data *= 0.0
        policy.fc2.weight.data *= 0.0
        policy.fc3_mu.weight.data *= 0.0
        policy.fc1.bias.data *= 0.0
        policy.fc2.bias.data *= 0.0
        policy.fc3_mu.bias.data *= 0.0
        policy.fc3_mu.bias.data += 1.

    toylibpath = '../opt/tests/cmp_py_cpp/set6/libRollout_toy.so'
    toy_roller = ToyCRoller(lib_path=toylibpath, options=dict())
    toy_roller.seed(12345)
    traj_num = 10
    n_steps = 200
    gamma = 0.99

    print('')
    print('#############################################################')
    print('########### Rolling Out Toy Greedy Trajectories ############')
    print('#############################################################')
    toy_roller.set_torch_policy(policy)
    with PrintTime(f'{traj_num} Rollouts!') as print_obj:
        greedy_lite_outdict = toy_roller.greedy_lite(traj_num, n_steps, gamma)
    with np.printoptions(precision=5, suppress=True, threshold=20, linewidth=100):
        for key, val in greedy_lite_outdict.items():
            printout(key, val)

    #############################################################
    ############## Instantiating a Leg C++ Roller ###############
    #############################################################

    lib_path = '../opt/tests/cmp_py_cpp/set1/libRollout.so'
    pyxml_file = f'../opt/tests/cmp_py_cpp/leg.xml'
    if not exists(lib_path):
        lib_path =  cleg_dir + '/opt/tests/cmp_py_cpp/set1/libRollout.so'
    leg_roller = LegCRoller(lib_path=lib_path, options=dict(outer_loop_rate=4000, xml_file=pyxml_file))

    leg_roller.seed(12345)
    np_random = leg_roller.np_random

    traj_num = 10
    n_steps = 8000
    obs_dim = leg_roller.obs_dim
    act_dim = leg_roller.act_dim

    # Loading the MLP agent from a file
    agent_path = '../opt/tests/cmp_py_cpp/set1/1_pd4k_actor'
    if not exists(agent_path):
        agent_path = cleg_dir + '/opt/tests/cmp_py_cpp/set1/1_pd4k_actor'
    policy = TypicalActor(observation_dim=obs_dim, action_dim=act_dim)
    policy.load_state_dict(torch.load(agent_path))
    leg_roller.set_torch_policy(policy)
    leg_roller.set_policy_calls -= 1 # Just not to mess up the rollout counts

    print('#############################################################')
    print('##################### Testing the MLP #######################')
    print('#############################################################')

    input_num = n_steps
    mlp_input = np_random.randn(input_num, obs_dim).astype(np.float64)
    tch_output = policy(torch.from_numpy(mlp_input)).detach().numpy()

    print(f'Starting {input_num} C++ MLP Inferences!')
    st_time = time.time()
    cpp_output = leg_roller.infer_mlp(mlp_input)
    end_time = time.time()
    print(f'Finished {input_num} C++ MLP Inferences!')
    print(f'Total Inference Time: {end_time - st_time}')

    with np.printoptions(precision=5, suppress=True, threshold=20, floatmode='fixed'):
        for i in range(min(input_num, 5)):
            print(f'Test Case {i}:')
            inp_str = '  ' + repr(mlp_input[i])
            inp_str = inp_str + ' ' * max(0, 50-len(inp_str))
            print(inp_str + '   === C++    MLP ===>     ' + repr(cpp_output[i]))
            print(inp_str + '   === Torch  MLP ===>     ' + repr(tch_output[i]))
            is_close_ = np.allclose(cpp_output[i], tch_output[i])
            print('Result: ' + ('Success!' if is_close_ else 'Failure!'))
            print('-'*20)

    is_close_ = np.allclose(cpp_output, tch_output)
    print('Overall Result: ' + ('Success!' if is_close_ else 'Failure!'))

    print('')
    print('#############################################################')
    print('############### Rolling Out Vine Trajectories ###############')
    print('#############################################################')
    gamma = 1.0 - 1.0 / n_steps
    expl_steps = 1
    reset_times = np_random.randint(low=0, high=n_steps, size=(traj_num,), dtype=np.int32)
    expl_noise = np_random.randn(traj_num, expl_steps, act_dim).astype(np.float64)
    leg_roller.set_torch_policy(policy)
    with PrintTime(f'{2*traj_num} Rollouts!') as print_obj:
        vine_outdict = leg_roller.vine(traj_num, n_steps, gamma, expl_steps,
                                       reset_times, expl_noise)
    with np.printoptions(precision=5, suppress=True, threshold=20, linewidth=100):
        for key, val in vine_outdict.items():
            printout(key, val)

    print('')
    print('#############################################################')
    print('########### Rolling Out Lite Greedy Trajectories ############')
    print('#############################################################')
    leg_roller.set_torch_policy(policy)
    with PrintTime(f'{traj_num} Rollouts!') as print_obj:
        greedy_lite_outdict = leg_roller.greedy_lite(traj_num, n_steps, gamma)
    with np.printoptions(precision=5, suppress=True, threshold=20, linewidth=100):
        for key, val in greedy_lite_outdict.items():
            printout(key, val)

    print('')
    print('#############################################################')
    print('############ Rolling Out Stochastic Trajectories ############')
    print('#############################################################')
    noise_std = 0.1
    expl_noise = np_random.randn(traj_num, n_steps, act_dim).astype(np.float64) * noise_std
    leg_roller.set_torch_policy(policy)
    with PrintTime(f'{traj_num} Rollouts with N(0,{noise_std}) Noise!') as print_obj:
        sto_outdict = leg_roller.stochastic(traj_num, n_steps, expl_noise)
    sto_outdict['return'] = sto_outdict['rewards'].sum(axis=1)
    with np.printoptions(precision=5, suppress=True, threshold=20, linewidth=100):
        for key, val in sto_outdict.items():
            printout(key, val)

    print('')
    print('#############################################################')
    print('############## Rolling Out Greedy Trajectories ##############')
    print('#############################################################')
    leg_roller.set_torch_policy(policy)
    with PrintTime(f'{traj_num} Rollouts!') as print_obj:
        greedy_outdict = leg_roller.greedy(traj_num, n_steps)
    greedy_outdict['return'] = greedy_outdict['rewards'].sum(axis=1)
    with np.printoptions(precision=5, suppress=True, threshold=20, linewidth=100):
        for key, val in greedy_outdict.items():
            printout(key, val)

    note_msg = ('Note: If the numbers above indicate a poor performing agent, it may \n'
                '      be because of a mismatch in the simulation interface options\n'
                '      used for compiling libRollout.so. This agent was trained with a \n'
                '      PD Loop running at 4KHz inner and outer loop rates. See \n'
                '          opt/tests/binding/defs.hpp\n'
                '      for an example set of compatible interface options for this agent.')
    if greedy_outdict['return'].mean() < -25000:
        print(note_msg)

    print('')
    print('#############################################################')
    print('############ Comparing C++ & Python Trajectories ############')
    print('#############################################################')

    tst_basepath = abspath('../opt/tests/cmp_py_cpp')
    if not exists(tst_basepath):
        tst_basepath =  cleg_dir + '/opt/tests/cmp_py_cpp'
    import json
    import random
    import sys
    from unittest.mock import patch
    sys.path.append(tst_basepath) # to load leg_dbg

    from leg_dbg import SimInterface, GymEnv
    from noise import MeasuredNoiseGenerator

    def make_py_cpp_envs(lib_path, outer_loop_rate, time_before_reset,
                         jumping_obs, do_jump, do_obs_noise, add_act_sm_r,
                         rew_type):
        opts_json_path = f'{tst_basepath}/agent_interface_options.json'
        pyxml_file = f'{tst_basepath}/leg.xml'

        cleg_roller_opts = dict()
        cleg_roller_opts['theta_hip_init'] = [-50 * np.pi / 180] * 2
        cleg_roller_opts['theta_knee_init'] = [-100 * np.pi / 180] * 2
        cleg_roller_opts['omega_hip_init'] = [0, 0]
        cleg_roller_opts['omega_knee_init'] = [0, 0]
        cleg_roller_opts['pos_slider_init'] = [0.4, 0.4]
        cleg_roller_opts['vel_slider_init'] = [0.0, 0.0]
        cleg_roller_opts['jump_time_bounds'] = [1.35, 1.35]
        cleg_roller_opts['outer_loop_rate'] = outer_loop_rate
        cleg_roller_opts['jumping_obs'] = jumping_obs
        cleg_roller_opts['add_act_sm_r'] = add_act_sm_r
        cleg_roller_opts['xml_file'] = pyxml_file
        cleg_roller_opts['inner_loop_rate'] = 4000
        cleg_roller_opts['ground_contact_type'] = 'compliant'
        cleg_roller_opts['time_before_reset'] = time_before_reset
        cleg_roller_opts['do_jump'] = do_jump
        cleg_roller_opts['do_obs_noise'] = do_obs_noise
        cleg_roller_opts['contact_obs'] = False
        cleg_roller_opts['slurm_job_file'] = None
        cleg_roller_opts['stand_reward'] = rew_type
        cleg_roller = LegCRoller(lib_path=lib_path, options=cleg_roller_opts)

        with open(opts_json_path, 'r') as infile:
            interface_options = json.load(infile)
        interface_options.update(cleg_roller_opts)

        interface_metadata = {}

        sim_interface = SimInterface(options=interface_options,
                                     metadata=interface_metadata)
        env_py = GymEnv(sim_interface)
        return env_py, cleg_roller

    for set_id in (1,):
        time_before_reset = 2
        do_obs_noise = False
        add_act_sm_r = False
        rew_type = 'dense'
        if set_id == 1:
            outer_loop_rate = 4000
            agent_path = f'{tst_basepath}/set1/1_pd4k_actor'
            lib_path = f'{tst_basepath}/set1/libRollout.so'
            jumping_obs = False
            do_jump = False
            print(f'\n**********************************************************')
            print(f'--> Set 1) Testing 4KHz Inner & Outer Loop Drop Agent')
        elif set_id == 2:
            outer_loop_rate = 500
            # This agent was downloaded from
            # https://github.com/compdyn/robo-rl/blob/cl11_drop_curric/mujoco_leg/agents/
            # cl11_500Hz_40cm_0_1-1457972-2021-02-07/agent_actor
            agent_path = f'{tst_basepath}/set2/2_pd500HzOuter_drop_actor'
            lib_path = f'{tst_basepath}/set2/libRollout.so'
            jumping_obs = True
            do_jump = False
            print(f'\n**********************************************************')
            print(f'--> Set 2) Testing 500 Hz Outer & 4KHz Inner Loop Drop Agent')
        elif set_id == 3:
            outer_loop_rate = 100
            # This agent was downloaded from
            # https://github.com/compdyn/robo-rl/blob/cl06_nonCompliantJump/mujoco_leg/
            # agents/cl03_curric_10067888-10068530-2019-12-24/agent_actor
            agent_path = f'{tst_basepath}/set3/3_jump_pd100HzOuter_actor'
            lib_path = f'{tst_basepath}/set3/libRollout.so'
            jumping_obs = True
            do_jump = True
            print(f'\n**********************************************************')
            print(f'--> Set 3) Testing 100 Hz Outer & 4KHz Inner Loop Jump Agent')
        elif set_id == 4:
            outer_loop_rate = 100
            # This agent was downloaded from
            # https://github.com/compdyn/robo-rl/blob/cl06_nonCompliantJump/mujoco_leg/
            # agents/cl03_curric_10067888-10068530-2019-12-24/agent_actor
            agent_path = f'{tst_basepath}/set4/3_jump_pd100HzOuter_actor'
            lib_path = f'{tst_basepath}/set4/libRollout.so'
            jumping_obs = True
            do_jump = True
            do_obs_noise = True
            print(f'\n*******************************************************************')
            print(f'--> Set 4) Testing Noised 100 Hz Outer & 4KHz Inner Loop Jump Agent')
        elif set_id == 5:
            outer_loop_rate = 4000
            agent_path = f'{tst_basepath}/set5/1_pd4k_actor'
            lib_path = f'{tst_basepath}/set5/libRollout.so'
            jumping_obs = False
            do_jump = False
            add_act_sm_r = True
            rew_type = 'simplified'
            print(f'\n**********************************************************')
            print(f'--> Set 5) Testing 4KHz Inner & Outer Loop Drop Agent ' +
                  'with action smoothness reward')
        else:
            raise Exception(f'Unknown set_id: {set_id}')

        env_py, leg_roller = make_py_cpp_envs(lib_path, outer_loop_rate, time_before_reset,
                                              jumping_obs, do_jump, do_obs_noise, add_act_sm_r,
                                              rew_type)

        n_steps = int(outer_loop_rate * time_before_reset)
        obs_dim = leg_roller.obs_dim
        act_dim = leg_roller.act_dim

        policy = TypicalActor(observation_dim=obs_dim, action_dim=act_dim)
        policy.load_state_dict(torch.load(agent_path), strict=False)
        ################### Running the C++ Trajectory ######################
        leg_roller.seed(12345)
        leg_roller.set_torch_policy(policy)

        with PrintTime(f'One C++ Leg Rollout') as print_obj:
            greedy_outdict = leg_roller.greedy(1, n_steps)
        s_cpp = greedy_outdict['observations'][0]
        a_cpp = greedy_outdict['actions'][0]
        r_cpp = greedy_outdict['rewards'][0]

        ################### Running the Python Trajectory ###################
        np.random.seed(12345) # SimInterface uses np.random as RNG
        random.seed(12345)
        torch.manual_seed(12345)

        def patched_mngen_init(self):
            self.i = leg_roller.r_init.noise_index[0] - 2

        with PrintTime(f'One Python Leg Rollout') as print_obj:
            s_py = np.zeros((n_steps, obs_dim))
            a_py = np.zeros((n_steps, act_dim))
            r_py = np.zeros(n_steps)
            with patch.object(MeasuredNoiseGenerator, '__init__', patched_mngen_init):
                s_next = env_py.reset()
            for t in range(n_steps):
                s_py[t,:] = s_next
                with torch.no_grad():
                    a_py[t,:] = policy(torch.from_numpy(s_next)).numpy()
                s_next, r_py[t], done, info = env_py.step(a_py[t,:])
                if done:
                    break

        ################# Comparting Python & C++ Trajectories ##################
        def max_diff(cpp_arr, py_arr):
            return np.abs(cpp_arr - py_arr).max()
        def mean_abs_diff(cpp_arr, py_arr):
            return np.abs(cpp_arr - py_arr).mean()
        print(f'Maximum       Observation Difference: %0.10f' % max_diff(s_cpp, s_py))
        print(f'Maximum       Action      Difference: %0.10f' % max_diff(a_cpp, a_py))
        print(f'Maximum       Reward      Difference: %0.10f' % max_diff(r_cpp, r_py))
        print(f'Mean Absolute Observation Difference: %0.10f' % mean_abs_diff(s_cpp, s_py))
        print(f'Mean Absolute Action      Difference: %0.10f' % mean_abs_diff(a_cpp, a_py))
        print(f'Mean Absolute Reward      Difference: %0.10f' % mean_abs_diff(r_cpp, r_py))

    print('')
    print('Note: All the previous tests used pre-compiled .so files from opt/tests.')
    print('')
    print('#############################################################')
    print('##### Waiting for dm_control to Spit Out Closing Errors! ####')
    print('#############################################################')
