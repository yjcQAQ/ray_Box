import ray
import ray.rllib.agents.ppo as ppo
from box2DEnv import Box2DEnv
from ray.tune.registry import register_env


def env_creator(env_config):
    return Box2DEnv(env_config['mat'])  # return an env instance


ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
preprocessor_pref = 'rllib'
register_env("my_env", env_creator)
trainer = ppo.PPOTrainer(config={"env_config": {"mat": [[1, 0, 0], [0, 0, -1]]}}, env='my_env')
# 加载训练完的节点
# trainer.restore('/Users/4paradigm/ray_results/PPO_CartPole-v0_2021-07-14_21-27-59dy1d906l/checkpoint_000001')
trainer.train()
