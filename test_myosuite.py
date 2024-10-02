import myosuite, deprl
from myosuite.envs.myo.myochallenge import *
from myosuite.utils import gym
from tqdm import tqdm_notebook as tqdm

env = gym.make("myoChallengeOslRunFixed-v0")
from stable_baselines3 import PPO

# model = PPO("MlpPolicy", env, verbose=0)
# model.learn(total_timesteps=100)


# # evaluate policy
# all_rewards = []
# for _ in tqdm(range(100)): # 5 random targets
#   ep_rewards = []
#   done = False
#   obs,_ = env.reset()
#   done = False
#   for _ in range(100):
#       obs = env.observation_space.sample()
#       # get the next action from the policy
#       action, _ = model.predict(obs, deterministic=True)
#       # take an action based on the current observation
#       obs, reward, done, _, info = env.step(action)
#       ep_rewards.append(reward)
#   all_rewards.append(np.sum(ep_rewards))
# print(f"Average reward: {np.mean(all_rewards)} over 5 episodes")


for ep in range(5):
    print(f'Episode: {ep} of 5')
    state = env.reset()
    timestep = 0
    while timestep < 2000:
        # action = env.action_space.sample()       
        obs = env.observation_space.sample()
        action, _ = model.predict(obs, deterministic=True)
        # uncomment if you want to render the task
        env.mj_render()
        next_state, reward, done,_,info = env.step(action)
        state = next_state
        if done: 
            break
        timestep += 1
env.close()