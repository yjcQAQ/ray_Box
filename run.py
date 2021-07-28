import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
import gym
from ray.rllib.env.wrappers.atari_wrappers import is_atari, \
    get_wrapper_by_cls, MonitorEnv, NoopResetEnv, ClipRewardEnv, \
    FireResetEnv, EpisodicLifeEnv, MaxAndSkipEnv, WarpFrame, FrameStack, \
    FrameStackTrajectoryView, ScaledFloatFrame, wrap_deepmind
from ray.rllib.utils.deprecation import deprecation_warning

# from ray.rllib.env.wrappers import atari_wrappers
# env = gym.make('SpaceInvaders-v0')
# atari_wrappers.wrap_deepmind(env)
# env=atari_wrappers.wrap_deemind(env)
# env=atari_wrappers.wrap_deepmind(env)
from ray.rllib.env.wrappers.atari_wrappers import is_atari, \
    get_wrapper_by_cls, MonitorEnv, NoopResetEnv, ClipRewardEnv, \
    FireResetEnv, EpisodicLifeEnv, MaxAndSkipEnv, WarpFrame, FrameStack, \
    FrameStackTrajectoryView, ScaledFloatFrame, wrap_deepmind
from ray.rllib.utils.deprecation import deprecation_warning




ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
preprocessor_pref='rllib'
trainer = ppo.PPOTrainer(config=config,env='SpaceInvaders-v0')

# trainer.restore('/Users/4paradigm/ray_results/PPO_CartPole-v0_2021-07-14_21-27-59dy1d906l/checkpoint_000001')
# trainer.train()
# Can optionally call trainer.restore(path) to load a checkpoint.
# trainer.save('/Users/4paradigm/result')
# trainer.restore('/Users/4paradigm/ray_results/PPO_2021-07-15_16-21-05/PPO_CartPole-v0_9984c_00000_0_2021-07-15_16-21-05/checkpoint_000010/checkpoint-10')
# for i in range(100):
#    # Perform one iteration of training the policy with PPO
#    result = trainer.train()
#    print(pretty_print(result))
#
#    if i % 10 == 0:
#        checkpoint = trainer.save()
#        print("checkpoint saved at", checkpoint)
#
# # Also, in case you have trained a model outside of ray/RLlib and have created
# # an h5-file with weight values in it, e.g.
# # my_keras_model_trained_outside_rllib.save_weights("model.h5")
# # (see: https://keras.io/models/about-keras-models/)
#
# # ... you can load the h5-weights into your Trainer's Policy's ModelV2
# # (tf or torch) by doing:
# trainer.import_model("my_weights.h5")
# # NOTE: In order for this to work, your (custom) model needs to implement
# # the `import_from_h5` method.
# # See https://github.com/ray-project/ray/blob/master/rllib/tests/test_model_imports.py
# # for detailed examples for tf- and torch trainers/models
# agent = ppo.PPOTrainer(config=config, env='CartPole-v0')

env = gym.make('SpaceInvaders-v0')
env = wrap_deepmind(env)
for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        env.render()
        print(observation)
        action = trainer.compute_action(observation)
        # action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

