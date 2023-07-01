from algorithm.tdmpc import TDMPC
from algorithm import helper as h
from cfg import parse_adaptation_cfg
from pathlib import Path
from algorithm.tdmpc import TDMPC
import os
import imageio
import pandas as pd
from omegaconf import OmegaConf
from env import make_env_via_viewgen_type
import numpy as np
import torch
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from copy import deepcopy
from tqdm import tqdm
import wandb
import random

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

class ReplayBuffer():
    def __init__(self, capacity, obs_shape, action_shape):
        self.capacity = capacity
        self.obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device='cuda')
        self.next_obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device='cuda')
        self.actions = torch.zeros((capacity, *action_shape), dtype=torch.float32, device='cuda')
        self.idx = 0
        self.size = 0
        
    def add(self, obs, action, next_obs):
        self.obs[self.idx] = torch.tensor(obs, dtype=torch.float32, device='cuda')
        self.next_obs[self.idx] = torch.tensor(next_obs, dtype=torch.float32, device='cuda')
        self.actions[self.idx] = action.clone().detach()
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        idxs = torch.randint(low=0, high=self.size, size=(batch_size,))
        return self.obs[idxs], self.actions[idxs], self.next_obs[idxs]

class VideoRecorder(object):
    def __init__(self, dir_name, height=100, width=100, camera_id=0, fps=25, domain_name='not_xarm'):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []
        self.domain_name = domain_name

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env=None, obs=None):
        if self.enabled:
            if self.domain_name == 'xarm':
                frame = env.env.sim.render(
                        	mode="offscreen",
                        	width=self.width,
                        	height=self.height,
                        	camera_name="third_person",
                    	)[::-1,:,:]
            else:
                frame = env.render(
            	    mode='rgb_array',
            	    height=self.height,
            	    width=self.width,
            	    camera_id=self.camera_id
        		)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
            wandb.log({'eval_video': wandb.Video(np.stack(self.frames).transpose(0, 3, 1, 2), fps=self.fps, format='mp4')})
	    
	    
def evaluate(env, agent, buffer, num_episodes, ssl_task, video, cfg=None):
	episode_rewards, episode_success = [], []
	ep_agent = deepcopy(agent) 
	num_total_step = 0

	ep_agent.model.train()
	for i in tqdm(range(num_episodes)):

		video.init(enabled=True)
		obs = env.reset()		
		done = False
		episode_reward = 0
		step = 0
		
		while not done:
			ep_agent.model.train(False)  # TODO
			action = ep_agent.plan(obs, eval_mode=True, step=1e10, t0=step==0) 
			ep_agent.model.train(True)
			next_obs, reward, done, info = env.step(action.cpu().numpy())
			episode_reward += reward
			buffer.add(obs, action, next_obs)

			if ssl_task != 'no':
				for j in range(int(h.linear_schedule(cfg.multi_update_schedule, 1000*i+step))):
					batch_obs, batch_action, batch_next_obs = buffer.sample(cfg.batch_size_eval)               
					ep_agent.adapt_via_ssl(batch_obs, batch_next_obs, batch_action, cfg, ssl_task, j + step * int(h.linear_schedule(cfg.multi_update_schedule, 1000*i+step)))

			video.record(env)
			obs = next_obs
			step += 1
			num_total_step += 1
		video.save(f'{i}.mp4')
		episode_rewards.append(episode_reward)
		episode_success.append(int(info.get('success', 0)) + int(info.get('is_success', 0))) 
		wandb.log({'episode_reward': episode_reward})
		wandb.log({'episode_success': int(info.get('success', 0)) + int(info.get('is_success', 0))})
	wandb.log({'average_reward': np.mean(episode_rewards)}) 
	wandb.log({'average_success_rate': np.mean(episode_success)}) 
	return np.mean(episode_rewards), np.nanmean(episode_success)


def main_viewgen_exp(cfg):
	set_seed(cfg.seed)
	import wandb

	os.makedirs(cfg.eval_dir, exist_ok=True)

	# video
	video_dir = os.path.join(cfg.eval_dir, 'video')
	os.makedirs(video_dir, exist_ok=True)
	video = VideoRecorder(video_dir, height=448, width=448, domain_name=cfg.task.replace('-', '_').split('_', 1)[0])

	viewgen_difficulty = cfg.difficulty
	assert torch.cuda.is_available(), 'must have cuda enabled'

	env = make_env_via_viewgen_type(cfg, cfg.viewgen_type)

	agent = TDMPC(cfg)
	agent.load(fp=f'{cfg.checkpoints_path}/{cfg.task}/model.pt')
	agent.model.adaptation_init(cfg)
        
	reward, success_rate = 0, 0
	wandb.init(project=cfg.wandb_project.replace('/', ',').replace(" ", ""),
		entity=cfg.wandb_entity,
		group=f'{cfg.exp_disc}'.replace('/', ',').replace(" ", ""),
        name=str(cfg.seed),
		config=OmegaConf.to_container(cfg, resolve=True))
	buffer = ReplayBuffer(cfg.buffer_size, (cfg.frame_stack*3, 84, 84), env.action_space.shape)                
	reward, success_rate = evaluate(env=env, agent=agent, buffer=buffer, num_episodes=cfg.num_episodes, ssl_task=cfg.ssl_task, video=video, cfg=cfg)
	print(f'{cfg.viewgen_type},{cfg.ssl_task}###  ', 'reward: ', reward, 'success_rate: ', success_rate)
	wandb.finish()
        
		
if __name__ == '__main__':
	cfg = parse_adaptation_cfg(Path().cwd() / 'cfgs')		
	main_viewgen_exp(cfg)
