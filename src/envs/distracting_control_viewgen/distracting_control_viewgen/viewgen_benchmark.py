import numpy as np
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from distracting_control_viewgen import camera_no_random_adroit, camera_no_random, camera_no_random_xarm
from dm_control import suite as dm_control_suite
from dm_control.suite.wrappers import pixels
from distracting_control_viewgen import dmc2gym_wrapper
import gym


def get_camera_params(domain_name, scale, is_dynamic=False, is_phi=True, is_theta=True, is_r=True, is_theta_roll=True):
    return dict(
        vertical_delta=np.pi / 6 * scale if is_theta else 0.,
        horizontal_delta=np.pi / 4 * scale if is_phi else 0.,
        # Limit camera to -90 / 90 degree rolls.
        roll_delta=np.pi / 4. * scale if is_theta_roll else 0.,
        vel_std=.1 * scale if is_dynamic else 0.,
        max_vel=.4 * scale if is_dynamic else 0.,
        roll_std=np.pi / 300 * scale if is_dynamic else 0.,
        max_roll_vel=np.pi / 50 * scale if is_dynamic else 0.,
        # Allow the camera to zoom in at most 50%.
        max_zoom_in_percent=.5 * scale if is_r else 0.,
        # Allow the camera to zoom out at most 200%.
        max_zoom_out_percent=1.5 * scale if is_r else 0.,
        limit_to_upper_quadrant='reacher' not in domain_name,
    )


def distraction_wrap_dmc(env, domain_name, difficulty, is_dynamic=False, fov=0, is_moving=False, is_phi=True, is_theta=True, is_r=True, is_theta_roll=True, is_shake=False):
    camera_kwargs = get_camera_params(domain_name=domain_name, scale=difficulty, is_dynamic=is_dynamic, is_phi=is_phi, is_theta=is_theta, is_r=is_r, is_theta_roll=is_theta_roll)
    return camera_no_random.DistractingCameraEnv(env, camera_id=0, fov=fov, is_shake=is_shake, is_moving=is_moving, **camera_kwargs)

def distraction_wrap_adroit(env, difficulty, fov=0, is_shake=False, is_moving=False):
    return camera_no_random_adroit.DistractingCameraEnv(env, camera_id=2, difficulty=difficulty, fov=fov, is_shake=is_shake, is_moving=is_moving)

def distraction_wrap_xarm(env, difficulty, fov=0, is_shake=False, is_moving=False):
    return camera_no_random_xarm.DistractingCameraEnv(env, camera_id=0, difficulty=difficulty, fov=fov, is_shake=is_shake, is_moving=is_moving)

def make_env(domain_name, task_name, difficulty, fov=0, is_shake=False, is_moving=False, is_dynamic=False, is_phi=True, is_theta=True, is_r=False, is_theta_roll=False, seed=0, frame_stack=3):
    if domain_name == 'adroit':
        import sys; sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/../../mj_envs')
        env_id = task_name.split("-", 1)[-1] + "-v0"
        env = gym.make(env_id)
        env = distraction_wrap_adroit(env, difficulty=difficulty, fov=fov, is_shake=is_shake, is_moving=is_moving)
    elif domain_name == 'xarm':
        from xarm.wrappers import make_env
        env = make_env(
            domain_name='xarm',
            task_name=task_name,
            seed=seed,
            episode_length=50,
            n_substeps=20,
            frame_stack=frame_stack,
            image_size=84,
            cameras=['third_person'],  # ['third_person', 'first_person']
            observation_type='image', 
            action_space='xy' if task_name=='reach' or task_name=='push' else 'xyz',  # Reach, Push: 'xy'.  Pegbox, Hammerall: 'xyz'
        )
        env = distraction_wrap_xarm(env, difficulty=difficulty, fov=fov, is_shake=is_shake, is_moving=is_moving)
    else:
        env = dm_control_suite.load(domain_name, task_name)
        env = distraction_wrap_dmc(env, domain_name, difficulty=difficulty, fov=fov, is_shake=is_shake, is_moving=is_moving, is_dynamic=is_dynamic, is_phi=is_phi, is_theta=is_theta, is_r=is_r, is_theta_roll=is_theta_roll)
    return env


def make_gym_env(domain_name, task_name, difficulty, is_dynamic=False, is_phi=True, is_theta=True, is_r=False, is_theta_roll=False, image_height=256, image_width=256, channels_first=False):
    env = dm_control_suite.load(domain_name, task_name)
    env = distraction_wrap_dmc(env, domain_name, difficulty=difficulty, is_dynamic=is_dynamic, is_phi=is_phi, is_theta=is_theta, is_r=is_r, is_theta_roll=is_theta_roll)
    env = pixels.Wrapper(env, render_kwargs={'camera_id': 0, 'height': image_height, 'width': image_width})
    env = dmc2gym_wrapper.DMC2Gym(env, channels_first=channels_first)

    return env

