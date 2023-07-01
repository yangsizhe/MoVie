import copy
import numpy as np
import math


class DistractingCameraEnv():  
  def __init__(self,
               env,
               camera_id,
               difficulty,
               fov=45,
               is_shake=False,
               is_moving=False,
              ):
    
    self._env = env
    self._camera_id = camera_id
    self.step_count = 0
    self.difficulty = difficulty
    self._fov = fov
    self._is_shake = is_shake
    self._is_moving = is_moving
    self.delta_angle = self.difficulty * np.pi / 3 
    self.origin_cam_pos = self._env.env.sim.model.cam_pos[self._camera_id][:2]
    self.cam_z = self._env.env.sim.model.cam_pos[self._camera_id][2] 
    self.target_pos = np.array([0.1794, -0.1072])  # TODO
    self.r = np.sqrt(np.sum(np.square(self.origin_cam_pos - self.target_pos)))
    self.origin_angle = math.atan2(self.origin_cam_pos[1] - self.target_pos[1], self.origin_cam_pos[0] - self.target_pos[0])    

  def setup_camera(self):
    scale = 1
    if self._is_moving:
      cycle = 100
      x = (self.step_count % cycle) / cycle
      if 0 <= x <= 0.5:
          scale = 1 - 4 * x
      elif 0.5 < x <= 1:
          scale = -3 + 4 * x
    self.angle = self.origin_angle + scale * self.delta_angle
    x2 = self.target_pos[0] + self.r * math.cos(self.angle)
    y2 = self.target_pos[1] + self.r * math.sin(self.angle)
    self.cam_pos = np.array([x2, y2, self.cam_z])
    
  def reset(self):
    """Reset the camera state. """
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
      noise = np.random.normal(loc=0, scale=0.04, size=(3,))
      noise = np.clip(noise, -0.07, +0.07)
      self._env.env.sim.model.cam_pos[self._camera_id] = self.cam_pos + noise
    else: 
      self._env.env.sim.model.cam_pos[self._camera_id] = self.cam_pos
    if self._fov != 45:
      self._env.env.sim.model.cam_fovy[self._camera_id] = self._fov

  # Forward property and method calls to self._env.
  def __getattr__(self, attr):
    if hasattr(self._env, attr):
      return getattr(self._env, attr)
    raise AttributeError("'{}' object has no attribute '{}'".format(
        type(self).__name__, attr))
  