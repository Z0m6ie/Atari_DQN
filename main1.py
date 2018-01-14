import sys
import gym
from gym import wrappers
from agent1 import Agent


batch_size = 32
episodes = sys.argv[1] if len(sys.argv) > 1 else 10000
env_name = sys.argv[2] if len(sys.argv) > 2 else "Breakout-v0"

episodes = int(episodes)
env_name = env_name
D = 84 * 84


env = gym.make(env_name)

env = wrappers.Monitor(env, env_name, force=True)

agent = Agent(env.observation_space.shape, env.action_space.n)

for i_episodes in range(episodes):
    State = env.reset()
    state_for_stack = agent.RGBprocess(State)
    state = agent.stack(state_for_stack, state_for_stack)
    totalreward = 0
    done = False
    while not done:
        if i_episodes % 50 == 0:
            env.render()
        action = agent.act(state)
        new_state, reward, done, info = env.step(action)
        new_state_p = agent.RGBprocess(new_state)
        new_state_dif = agent.stack(new_state_p, state_for_stack)
        agent.remember(state, action, reward, new_state_dif, done)
        state = new_state_dif
        state_for_stack = new_state_p
        totalreward += reward
    agent.memory_replay(batch_size)
    if done:
        print("{} episode, score = {}\n".format(i_episodes + 1, totalreward))
        agent.save_model()
agent.f.close()
env.close()
gym.upload(env_name, api_key='sk_WRCITkqmTJKYB9hvBk5tPA')
