import copy
from dm_control.rl import control
import numpy as np
import math


class DistractingCameraEnv(control.Environment):
  def __init__(self,
               env,
               camera_id,
               difficulty,
               fov=50,
               is_shake=False,
               is_moving=False,
               vel=0.001,
               seed=None):
    
    self._env = env
    self._camera_id = camera_id
    self.step_count = 0
    self.difficulty = difficulty
    self._fov = fov
    self._is_shake = is_shake
    self._is_moving = is_moving
    self.delta_angle = self.difficulty * np.pi / 3 
    self.origin_cam_pos = self._env.env.env.env.env.env.sim.model.cam_pos[self._camera_id][:2]
    self.cam_z = self._env.env.env.env.env.env.sim.model.cam_pos[self._camera_id][2] 
    self.target_pos = np.array([1.5237, 0.3])
    self.r = np.sqrt(np.sum(np.square(self.origin_cam_pos - self.target_pos)))
    self.origin_angle = math.atan2(self.origin_cam_pos[1] - self.target_pos[1], self.origin_cam_pos[0] - self.target_pos[0])    

  def setup_camera(self):
    scale = 1
    delta_z = 0
    if self._is_moving:
      cycle = 100
      x = (self.step_count % cycle) / cycle
      if 0 <= x <= 0.5:
          scale = 1 - 4 * x          
      elif 0.5 < x <= 1:
          scale = -3 + 4 * x

      if 0 <= x <= 0.25:
          delta_z = 1.2 * x       
      elif 0.25 < x <= 0.5:
          delta_z = 0.6 - 1.2 * x
      elif 0.5 < x <= 0.75:
          delta_z = 1.2 * x - 0.6
      elif 0.75 < x <= 1:
          delta_z = 1.2 - 1.2 * x
    
    self.angle = self.origin_angle + scale * self.delta_angle
    x2 = self.target_pos[0] + self.r * math.cos(self.angle)
    y2 = self.target_pos[1] + self.r * math.sin(self.angle)
    self.cam_pos = np.array([x2, y2, self.cam_z + delta_z])
    
  def reset(self):
    self.step_count = 0
    self.setup_camera()
    self._apply()
    obs = self._env.reset()
    return obs

  def step(self, action):
    self.step_count += 1
    self.setup_camera()
    self._apply()
    a, b, c, d = self._env.step(action)
    return a, b, c, d

  def _apply(self):
    if self._is_shake:
      noise = np.random.normal(loc=0, scale=0.4, size=(3,))
      noise = np.clip(noise, -0.07, +0.07)
      self._env.env.env.env.env.env.sim.model.cam_pos[self._camera_id] = self.cam_pos + noise
    else: 
      self._env.env.env.env.env.env.sim.model.cam_pos[self._camera_id] = self.cam_pos
    if self._fov != 50:
      self._env.env.env.env.env.env.sim.model.cam_fovy[self._camera_id] = self._fov

  def __getattr__(self, attr):
    if hasattr(self._env, attr):
      return getattr(self._env, attr)
    raise AttributeError("'{}' object has no attribute '{}'".format(
        type(self).__name__, attr))
  