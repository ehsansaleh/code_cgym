import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class NLRPendulum(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g=10.0, max_ep_len=200, des_T=0.5, des_th=np.pi/2, des_amp=0.1):
        self.max_speed=8
        self.max_torque=80.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None
        self.des_T = des_T
        self.des_th = des_th
        self.des_amp = des_amp

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.max_ep_len = max_ep_len
        self.th_mem = np.full(self.max_ep_len+1, np.nan, dtype=np.float64)
        self.omega_mem = np.full(self.max_ep_len+1, np.nan, dtype=np.float64)
        self.act_mem = np.full(self.max_ep_len, np.nan, dtype=np.float64)
        self.r_mem = np.full(self.max_ep_len, np.nan, dtype=np.float64)
        self.t = 0

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])

        self.t = self.t + 1
        self.act_mem[self.t - 1] = u
        self.th_mem[self.t] = self.state[0]
        self.omega_mem[self.t] = self.state[1]
        done = (self.t == (self.max_ep_len))
        self.out_info = dict()
        self.out_info['theta_rot'] = th - np.pi
        self.r_mem[self.t - 1] = r = self.get_reward(done)

        return self._get_obs(), r, done, self.out_info

    def get_reward(self, done):
        if done:
            s = self.th_mem - np.pi
            n = s.shape[0]
            freq = np.fft.fftfreq(n, self.dt)
            s_fft = np.fft.fft(s)
            sp2 = np.abs(s_fft)**2
            #This is the hacky part
            sp2 = sp2[freq>=0]
            freq = freq[freq>=0]
            sp2[1:] *= 2.
            sp = np.sqrt(sp2)/n

            # finding the reward
            dc_ = s_fft[0].real/n

            T_des = self.des_T # Default: 0.5
            des_th = self.des_th # Default: np.pi/2
            des_amp = self.des_amp # Default: 0.1

            gamma = 0.99
            r_coeff = 20. * (1. - gamma ** self.max_ep_len) / ((1.-gamma) * (gamma**self.max_ep_len))
            f_des = 1./T_des
            f_max = freq.max()

            sp2_ac = sp2[1:]
            sp_ac_normalized = sp2_ac / (sp2_ac.sum() + 1e-6)
            f_rew_profile = (freq <= f_des).astype(np.float)
            f_rew_profile += (freq >= (1.2 * f_des)).astype(np.float)
            f_rew_profile = -1. * f_rew_profile[1:]
            sp_ac_size = np.sqrt(sp2[1:].sum()) / n

            good_ac_reward = np.sum(sp_ac_normalized * f_rew_profile) * 0.1
            dc_reward = -1. * (np.abs(dc_ - des_th)/np.pi)
            bad_ac_reward = -1. * np.clip(sp_ac_size/des_amp - 1., 0., 10.)

            # Not really necessary, but for sake of unique optimality
            bad_ac_reward += -0.00000001 * np.clip(1. - sp_ac_size/des_amp, 0., 1.)

            self.out_info['ac_spectrum_reward'] = good_ac_reward * r_coeff
            self.out_info['ac_size_reward'] = bad_ac_reward * r_coeff
            self.out_info['dc_reward'] = dc_reward * r_coeff

            r = sum(val for key,val in self.out_info.items() if key.endswith('_reward'))
            return r
        else:
            self.out_info['ac_spectrum_reward'] = 0.
            self.out_info['ac_size_reward'] = 0.
            self.out_info['dc_reward'] = 0.
            return 0.

    def reset(self):
        high = np.array([np.pi + 0.1,  0.02])
        low  = np.array([np.pi - 0.1, -0.02])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        self.t = 0
        self.th_mem.fill(np.nan)
        self.omega_mem.fill(np.nan)
        self.act_mem.fill(np.nan)
        self.r_mem.fill(np.nan)
        self.th_mem[0] = self.state[0]
        self.omega_mem[0] = self.state[1]

        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-1.4,1.4,-1.4,1.4)

            for radian_ in np.linspace(0., np.pi/2., num=7):
                radius_ = 1.
                hour_ = rendering.make_polyline([(0,0), (radius_ * np.cos(radian_-np.pi/2), radius_ * np.sin(radian_-np.pi/2))])
                hour_.set_color(0,0,0)
                hour_.set_linewidth(1.)
                self.viewer.add_geom(hour_)

            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)

            #fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            fname = f'{os.path.dirname(gym.__file__)}/envs/classic_control/assets/clockwise.png'
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/(2*self.max_torque), np.abs(self.last_u/self.max_torque)/2)
            #self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class NLRPendulum_v1(NLRPendulum):
    def __init__(self):
        super().__init__(g=10.0, max_ep_len=200, des_T=0.50, des_th=np.pi*2./12., des_amp=0.2)

class NLRPendulum_v2(NLRPendulum):
    def __init__(self):
        super().__init__(g=10.0, max_ep_len=200, des_T=0.60, des_th=np.pi*2./12., des_amp=0.2)

class NLRPendulum_v3(NLRPendulum):
    def __init__(self):
        super().__init__(g=10.0, max_ep_len=200, des_T=0.50, des_th=np.pi*3./12., des_amp=0.2)

class NLRPendulum_v4(NLRPendulum):
    def __init__(self):
        super().__init__(g=10.0, max_ep_len=200, des_T=0.40, des_th=np.pi*2./12., des_amp=0.2)

class NLRPendulum_v5(NLRPendulum):
    def __init__(self):
        super().__init__(g=10.0, max_ep_len=200, des_T=0.50, des_th=np.pi*3./12., des_amp=0.5)

class NLRPendulum_v6(NLRPendulum):
    def __init__(self):
        super().__init__(g=10.0, max_ep_len=200, des_T=0.50, des_th=np.pi*4./12., des_amp=0.2)

class NLRPendulum_v7(NLRPendulum):
    def __init__(self):
        super().__init__(g=10.0, max_ep_len=200, des_T=0.50, des_th=np.pi*5./12., des_amp=0.2)

class NLRPendulum_v8(NLRPendulum):
    def __init__(self):
        super().__init__(g=10.0, max_ep_len=200, des_T=0.50, des_th=np.pi*6./12., des_amp=np.pi*2./12.)
