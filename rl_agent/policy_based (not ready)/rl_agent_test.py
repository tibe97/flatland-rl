from collections import deque
from collections import namedtuple

import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from reinforcement_learning.dddqn_policy import DDDQNPolicy
from reinforcement_learning.ppo_agent import PPOPolicy

dddqn_param_nt = namedtuple('DDDQN_Param', ['hidden_size', 'buffer_size', 'batch_size', 'update_every', 'learning_rate',
                                            'tau', 'gamma', 'buffer_min_size', 'use_gpu'])
dddqn_param = dddqn_param_nt(hidden_size=128,
                             buffer_size=1000,
                             batch_size=64,
                             update_every=10,
                             learning_rate=1.e-3,
                             tau=1.e-2,
                             gamma=0.95,
                             buffer_min_size=0,
                             use_gpu=False)


def cartpole(use_dddqn=False):
    eps = 1.0
    eps_decay = 0.99
    min_eps = 0.01
    training_mode = True

    env = gym.make("CartPole-v1")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    if not use_dddqn:
        policy = PPOPolicy(observation_space, action_space, False)
    else:
        policy = DDDQNPolicy(observation_space, action_space, dddqn_param)
    episode = 0
    checkpoint_interval = 20
    scores_window = deque(maxlen=100)

    writer = SummaryWriter()

    while True:
        episode += 1
        state = env.reset()
        policy.reset(env)
        handle = 0
        tot_reward = 0

        policy.start_episode(train=training_mode)
        while True:
            # env.render()
            policy.start_step(train=training_mode)
            action = policy.act(handle, state, eps)
            state_next, reward, terminal, info = env.step(action)
            policy.end_step(train=training_mode)
            tot_reward += reward
            # reward = reward if not terminal else -reward
            reward = 0 if not terminal else -1
            policy.step(handle, state, action, reward, state_next, terminal)
            state = np.copy(state_next)
            if terminal:
                break

        policy.end_episode(train=training_mode)
        eps = max(min_eps, eps * eps_decay)
        scores_window.append(tot_reward)
        if episode % checkpoint_interval == 0:
            print('\rEpisode: {:5}\treward: {:7.3f}\t avg: {:7.3f}\t eps: {:5.3f}\t replay buffer: {}'.format(episode,
                                                                                                              tot_reward,
                                                                                                              np.mean(
                                                                                                                  scores_window),
                                                                                                              eps,
                                                                                                              len(
                                                                                                                  policy.memory)))
        else:
            print('\rEpisode: {:5}\treward: {:7.3f}\t avg: {:7.3f}\t eps: {:5.3f}\t replay buffer: {}'.format(episode,
                                                                                                              tot_reward,
                                                                                                              np.mean(
                                                                                                                  scores_window),
                                                                                                              eps,
                                                                                                              len(
                                                                                                                  policy.memory)),
                  end=" ")

        writer.add_scalar("CartPole/value", tot_reward, episode)
        writer.add_scalar("CartPole/smoothed_value", np.mean(scores_window), episode)
        writer.flush()


if __name__ == "__main__":
    cartpole()
