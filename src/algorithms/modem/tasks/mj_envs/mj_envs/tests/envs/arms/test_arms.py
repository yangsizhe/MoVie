import gym
import mj_envs.envs.arms  # noqa
import numpy
import pickle
import pytest


ENVIRONMENT_IDS = (
    "FrankaReachFixed-v0",
    "FrankaReachRandom-v0",
    "FrankaPushFixed-v0",
    "FrankaPushRandom-v0",
    "FetchReachFixed-v0",
    "FetchReachRandom-v0",
)


@pytest.mark.parametrize("environment_id", ENVIRONMENT_IDS)
def test_serialize_deserialize(environment_id):
    input_seed = 123
    env1 = gym.make(environment_id, seed=input_seed)
    obs_dict_1 = env1.get_obs_dict(env1.env.sim)
    reward_dict_1 = env1.get_reward_dict(obs_dict_1)
    assert len(obs_dict_1) > 0
    assert len(reward_dict_1) > 0
    obs = env1.env.get_obs()
    assert len(obs) > 0
    infos1 = env1.env.get_env_infos()
    assert len(infos1) > 0
    assert env1.get_input_seed() == input_seed

    env1.reset()

    env2 = pickle.loads(pickle.dumps(env1))
    env2.reset()
    assert env2.get_input_seed() == input_seed

    assert env1.get_input_seed() == env2.get_input_seed(), {
        env1.get_input_seed(),
        env2.get_input_seed(),
    }
    assert env1.action_space == env2.action_space, (
        env1.action_space,
        env2.action_space,
    )
    assert (env1.get_obs() == env2.get_obs()).all(), (env1.get_obs(), env2.get_obs())

    obs_dict_2 = env2.get_obs_dict(env2.env.sim)
    reward_dict_2 = env2.get_reward_dict(obs_dict_2)
    infos2 = env2.env.get_env_infos()
    assert len(obs_dict_1) == len(obs_dict_2), (obs_dict_1, obs_dict_2)
    assert len(reward_dict_1) == len(reward_dict_2), (reward_dict_1, reward_dict_2)
    assert len(infos1) == len(infos2), (infos1, infos2)

    env1.env.step(numpy.zeros(env1.env.sim.model.nu))
    env2.env.step(numpy.zeros(env2.env.sim.model.nu))
