import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import algorithm.helper as h
from algorithm.transform import TransformNet_STN_PerImage, TransformNet_STN1


class InvFunction(nn.Module):
    """MLP for inverse dynamics model."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(2*obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, h, h_next):
        joint_h = torch.cat([h, h_next], dim=1)
        return self.trunk(joint_h)
    

class TOLD(nn.Module):
	"""Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._encoder = h.enc(cfg)
		
		self._dynamics = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
		self._reward = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)
		self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
		self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
		self.apply(h.orthogonal_init)
		for m in [self._reward, self._Q1, self._Q2]:
			m[-1].weight.data.fill_(0)
			m[-1].bias.data.fill_(0)

	def track_q_grad(self, enable=True):
		"""Utility function. Enables/disables gradient tracking of Q-networks."""
		for m in [self._Q1, self._Q2]:
			h.set_requires_grad(m, enable)

	def h(self, obs):
		"""Encodes an observation into its latent representation (h)."""
		return self._encoder(obs)

	def next(self, z, a):
		"""Predicts next latent state (d) and single-step reward (R)."""
		x = torch.cat([z, a], dim=-1)
		return self._dynamics(x), self._reward(x)

	def pi(self, z, std=0):
		"""Samples an action from the learned policy (pi)."""
		mu = torch.tanh(self._pi(z))
		if std > 0:
			std = torch.ones_like(mu) * std
			return h.TruncatedNormal(mu, std).sample(clip=0.3)
		return mu

	def Q(self, z, a):
		"""Predict state-action value (Q)."""
		x = torch.cat([z, a], dim=-1)
		return self._Q1(x), self._Q2(x)

	def adaptation_init(self, cfg):
		self._transforms = [TransformNet_STN_PerImage(cfg.frame_stack*3).cuda(), TransformNet_STN1(cfg.num_channels).cuda()]
		self.transform_optims = [torch.optim.Adam(self._transforms[0].parameters(), lr=cfg.lr_transforms_stn[0]), 
			   torch.optim.Adam(self._transforms[1].parameters(), lr=cfg.lr_transforms_stn[1]), ]

		self._encoder = nn.Sequential(
			list(self._encoder.children())[0],
			self._transforms[0],
			list(self._encoder.children())[1],
			list(self._encoder.children())[2],
			self._transforms[1],
			list(self._encoder.children())[3],
			list(self._encoder.children())[4],
			list(self._encoder.children())[5],
			list(self._encoder.children())[6],
			list(self._encoder.children())[7],
			list(self._encoder.children())[8],
			list(self._encoder.children())[9],
			list(self._encoder.children())[10]
			)
		
		self.encoder_optim = torch.optim.Adam(self._encoder.parameters(), lr=cfg.lr_encoder)

		encoder_param = []
		for name, param in self._encoder.named_parameters():
			if not ('localization' in name or 'fc_loc' in name):
				encoder_param.append(param)
				print(name)
		self.encoder_no_stn_optim = torch.optim.Adam(encoder_param, lr=cfg.lr_encoder)	

		self.ssl_models = {}
		self.ssl_models['inv'] = InvFunction(cfg.latent_dim, cfg.action_shape[0], 512).cuda()
		self.inv_optim =  torch.optim.Adam(self.ssl_models['inv'].parameters(), lr=1e-5)


class TDMPC():
	"""Implementation of TD-MPC learning + inference."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda')
		self.std = h.linear_schedule(cfg.std_schedule, 0)
		self.model = TOLD(cfg).cuda()
		self.model_target = deepcopy(self.model)
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
		self.aug = h.RandomShiftsAug(cfg)
		self.model.eval()
		self.model_target.eval()

	def state_dict(self):
		"""Retrieve state dict of TOLD model, including slow-moving target network."""
		return {'model': self.model.state_dict(),
				'model_target': self.model_target.state_dict()}

	def save(self, fp):
		"""Save state dict of TOLD model to filepath."""
		torch.save(self.state_dict(), fp)
	
	def load(self, fp):
		"""Load a saved state dict from filepath into current agent."""
		d = torch.load(fp)
		self.model.load_state_dict(d['model'])
		self.model_target.load_state_dict(d['model_target'])

	@torch.no_grad()
	def estimate_value(self, z, actions, horizon):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(horizon):
			z, reward = self.model.next(z, actions[t])
			G += discount * reward
			discount *= self.cfg.discount
		G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
		return G

	@torch.no_grad()
	def plan(self, obs, eval_mode=False, step=None, t0=True):
		"""
		Plan next action using TD-MPC inference.
		obs: raw input observation.
		eval_mode: uniform sampling and action noise is disabled during evaluation.
		step: current time step. determines e.g. planning horizon.
		t0: whether current step is the first step of an episode.
		"""
		# Seed steps
		if step < self.cfg.seed_steps and not eval_mode:
			return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

		# Sample policy trajectories
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
		horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
		num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
		if num_pi_trajs > 0:
			pi_actions = torch.empty(horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
			z = self.model.h(obs).repeat(num_pi_trajs, 1)
			for t in range(horizon):
				pi_actions[t] = self.model.pi(z, self.cfg.min_std)
				z, _ = self.model.next(z, pi_actions[t])

		# Initialize state and parameters
		z = self.model.h(obs).repeat(self.cfg.num_samples+num_pi_trajs, 1)
		mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
		std = 2*torch.ones(horizon, self.cfg.action_dim, device=self.device)
		if not t0 and hasattr(self, '_prev_mean'):
			mean[:-1] = self._prev_mean[1:]

		# Iterate CEM
		for i in range(self.cfg.iterations):
			actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device), -1, 1)
			if num_pi_trajs > 0:
				actions = torch.cat([actions, pi_actions], dim=1)

			# Compute elite actions
			value = self.estimate_value(z, actions, horizon).nan_to_num_(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score /= score.sum(0)
			_mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
			_std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
			_std = _std.clamp_(self.std, 2)
			mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

		# Outputs
		score = score.squeeze(1).cpu().numpy()
		actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
		self._prev_mean = mean
		mean, std = actions[0], _std[0]
		a = mean
		if not eval_mode:
			a += std * torch.randn(self.cfg.action_dim, device=std.device)
		return a

	def update_pi(self, zs):
		"""Update policy using a sequence of latent states."""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)

		# Loss is a weighted sum of Q-values
		pi_loss = 0
		for t,z in enumerate(zs):
			a = self.model.pi(z, self.cfg.min_std)
			Q = torch.min(*self.model.Q(z, a))
			pi_loss += -Q.mean() * (self.cfg.rho ** t)

		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.pi_optim.step()
		self.model.track_q_grad(True)
		return pi_loss.item()

	@torch.no_grad()
	def _td_target(self, next_obs, reward):
		"""Compute the TD-target from a reward and the observation at the following time step."""
		next_z = self.model.h(next_obs)
		td_target = reward + self.cfg.discount * \
			torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std)))
		return td_target

	def update(self, replay_buffer, step):
		"""Main update function. Corresponds to one iteration of the TOLD model learning."""
		obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		self.std = h.linear_schedule(self.cfg.std_schedule, step)
		self.model.train()

		# Representation
		z = self.model.h(self.aug(obs))
		zs = [z.detach()]

		consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
		att_loss = 0
		for t in range(self.cfg.horizon):

			# Predictions
			Q1, Q2 = self.model.Q(z, action[t])
			z, reward_pred = self.model.next(z, action[t])
			with torch.no_grad():
				next_obs = self.aug(next_obses[t])
				next_z = self.model_target.h(next_obs)
				td_target = self._td_target(next_obs, reward[t])
			zs.append(z.detach())

			# Losses
			rho = (self.cfg.rho ** t)
			consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)
			reward_loss += rho * h.mse(reward_pred, reward[t])
			value_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))
			priority_loss += rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target)) 

		# Optimize model
		total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
					 self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
					 self.cfg.value_coef * value_loss.clamp(max=1e4) 
		weighted_loss = (total_loss.squeeze(1) * weights).mean()
		weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
		weighted_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)

		self.optim.step()
		replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

		# Update policy + target network
		pi_loss = self.update_pi(zs)
		if step % self.cfg.update_freq == 0:
			h.ema(self.model, self.model_target, self.cfg.tau)

		self.model.eval()
		return {'consistency_loss': float(consistency_loss.mean().item()),
				'reward_loss': float(reward_loss.mean().item()),
				'value_loss': float(value_loss.mean().item()),
				'pi_loss': pi_loss,
				'total_loss': float(total_loss.mean().item()),
				'weighted_loss': float(weighted_loss.mean().item()),
				'grad_norm': float(grad_norm)}
	
	
	def adapt_via_ssl(self, obs, next_obs, action, cfg, ssl_task, adaptation_step):
		z = self.model.h(obs)
		z_next = self.model.h(next_obs)
		z_next_pred, reward_pred = self.model.next(z, action)
		dynamics_loss = F.mse_loss(z_next, z_next_pred)

		action_pred = self.model.ssl_models['inv'](z, z_next)
		idm_loss = F.mse_loss(action, action_pred)

		ssl_loss = 0
		
		if ssl_task == 'idm':
			ssl_loss += idm_loss

			self.model.inv_optim.zero_grad()
			self.model.encoder_optim.zero_grad()
			self.model.transform_optims[0].zero_grad()
			self.model.transform_optims[1].zero_grad()

			ssl_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
			ssl_loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)

			self.model.inv_optim.step()
			self.model.encoder_optim.step()
			self.model.transform_optims[0].step()
			self.model.transform_optims[1].step()

			return ssl_loss.item()

		elif ssl_task == 'dynamics':
			ssl_loss += dynamics_loss

			self.model.encoder_optim.zero_grad()
			self.model.transform_optims[0].zero_grad()
			self.model.transform_optims[1].zero_grad()

			ssl_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
			ssl_loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)

			self.model.encoder_optim.step()
			self.model.transform_optims[0].step()
			self.model.transform_optims[1].step()

			return ssl_loss.item()
			
		elif ssl_task == 'dynamics_encoder_only':
			ssl_loss += dynamics_loss
			self.model.encoder_no_stn_optim.zero_grad()
	
			ssl_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
			ssl_loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
	
			self.model.encoder_no_stn_optim.step()
			
			return ssl_loss.item()

