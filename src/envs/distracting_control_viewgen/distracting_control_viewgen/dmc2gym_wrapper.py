from time import time
from gym import spaces
from dm_env import StepType

class DMC2Gym:
    """Convert a DMC environment to a Gym environment"""

    def __init__(self, dmc_env, channels_first=False):
        """Initializes a new DMC2Gym wrapper
        Args:
            dmc_env (DMCEnv): The DMC environment to convert.
        """
        self._channels_first = channels_first
        dmc_obs_spec = dmc_env.observation_spec()['pixels']
        shape = dmc_obs_spec.shape
        if channels_first==True:
            shape = (dmc_obs_spec.shape[2],dmc_obs_spec.shape[0],dmc_obs_spec.shape[1])
        self._observation_space = spaces.Box(
            shape=shape,
            dtype=dmc_obs_spec.dtype,
            low=0,
            high=255
        )
        dmc_act_spec = dmc_env.action_spec()
        # import pdb; pdb.set_trace()
        self._action_space = spaces.Box(
            shape=dmc_act_spec.shape,
            dtype=dmc_act_spec.dtype,
            low=dmc_act_spec.minimum,
            high=dmc_act_spec.maximum
        )
        self._dmc_env = dmc_env
        self._max_episode_steps = 1000

    def step(self, action):
        time_step = self._dmc_env.step(action)
        obs = time_step.observation
        reward = time_step.reward
        discount = time_step.discount
        if time_step.last():
            done = True
        else:
            done = False
        if self._channels_first:
            obs['pixels'] = obs['pixels'].transpose(2, 0, 1).copy()
        return obs['pixels'], reward, done, {'discount': discount}

    def reset(self):
        time_step = self._dmc_env.reset()
        if self._channels_first:
            time_step.observation['pixels'] = time_step.observation['pixels'].transpose(2, 0, 1).copy()
        return time_step.observation['pixels']

    def render(self):
        return self._dmc_env.render()

    def observation_spec(self):
        return self._observation_space

    def action_spec(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space