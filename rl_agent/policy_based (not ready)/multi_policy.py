import numpy as np
from flatland.envs.rail_env import RailEnv

from reinforcement_learning.policy import Policy
from reinforcement_learning.ppo_agent import PPOPolicy
from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent


class MultiPolicy(Policy):
    def __init__(self, state_size, action_size, n_agents, env):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.loss = 0
        self.deadlock_avoidance_policy = DeadLockAvoidanceAgent(env, action_size, False)
        self.ppo_policy = PPOPolicy(state_size + action_size, action_size)

    def load(self, filename):
        self.ppo_policy.load(filename)
        self.deadlock_avoidance_policy.load(filename)

    def save(self, filename):
        self.ppo_policy.save(filename)
        self.deadlock_avoidance_policy.save(filename)

    def step(self, handle, state, action, reward, next_state, done):
        action_extra_state = self.deadlock_avoidance_policy.act(handle, state, 0.0)
        action_extra_next_state = self.deadlock_avoidance_policy.act(handle, next_state, 0.0)

        extended_state = np.copy(state)
        for action_itr in np.arange(self.action_size):
            extended_state = np.append(extended_state, [int(action_extra_state == action_itr)])
        extended_next_state = np.copy(next_state)
        for action_itr in np.arange(self.action_size):
            extended_next_state = np.append(extended_next_state, [int(action_extra_next_state == action_itr)])

        self.deadlock_avoidance_policy.step(handle, state, action, reward, next_state, done)
        self.ppo_policy.step(handle, extended_state, action, reward, extended_next_state, done)

    def act(self, handle, state, eps=0.):
        action_extra_state = self.deadlock_avoidance_policy.act(handle, state, 0.0)
        extended_state = np.copy(state)
        for action_itr in np.arange(self.action_size):
            extended_state = np.append(extended_state, [int(action_extra_state == action_itr)])
        action_ppo = self.ppo_policy.act(handle, extended_state, eps)
        self.loss = self.ppo_policy.loss
        return action_ppo

    def reset(self, env: RailEnv):
        self.ppo_policy.reset(env)
        self.deadlock_avoidance_policy.reset(env)

    def test(self):
        self.ppo_policy.test()
        self.deadlock_avoidance_policy.test()

    def start_step(self, train):
        self.deadlock_avoidance_policy.start_step(train)
        self.ppo_policy.start_step(train)

    def end_step(self, train):
        self.deadlock_avoidance_policy.end_step(train)
        self.ppo_policy.end_step(train)
