import time
import ray
import ray.rllib.agents.ppo as ppo
from boxenv import BoxEnv
from ray.tune.registry import register_env

def env_creator(env_config):
    return BoxEnv(env_config['map'])  # return an env instance

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
preprocessor_pref = 'rllib'
register_env("my_env", env_creator)
trainer = ppo.PPOTrainer(config={"env_config": {"map": [[1, 0, 0], [0, 0, -1]]}}, env='my_env')
# 加载训练完的节点
# trainer.restore('./PPO_CartPole-v0_2021-07-14_21-27-59dy1d906l/checkpoint_000001')

# 训练10次
for i in range(10):
    trainer.train()
env = BoxEnv([[1, 0, 0], [0, 0, -1]])

for i_episode in range(10):
    observation = env.reset()
    for t in range(1000):
        env.render()
        time.sleep(0.2)
        action = trainer.compute_action(observation)

        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            env.render()
            time.sleep(1)
            break
env.close()