from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv, RailEnvActions

from reinforcement_learning.policy import HybridPolicy
from reinforcement_learning.ppo_agent import PPOPolicy
from utils.agent_action_config import map_rail_env_action
from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent


class DeadLockAvoidanceWithDecisionAgent(HybridPolicy):

    def __init__(self, env: RailEnv, state_size, action_size, learning_agent):
        print(">> DeadLockAvoidanceWithDecisionAgent")
        super(DeadLockAvoidanceWithDecisionAgent, self).__init__()
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.learning_agent = learning_agent
        self.dead_lock_avoidance_agent = DeadLockAvoidanceAgent(self.env, action_size, False)
        self.policy_selector = PPOPolicy(state_size, 2)

        self.memory = self.learning_agent.memory
        self.loss = self.learning_agent.loss

    def step(self, handle, state, action, reward, next_state, done):
        select = self.policy_selector.act(handle, state, 0.0)
        self.policy_selector.step(handle, state, select, reward, next_state, done)
        self.dead_lock_avoidance_agent.step(handle, state, action, reward, next_state, done)
        self.learning_agent.step(handle, state, action, reward, next_state, done)
        self.loss = self.learning_agent.loss

    def act(self, handle, state, eps=0.):
        select = self.policy_selector.act(handle, state, eps)
        if select == 0:
            return self.learning_agent.act(handle, state, eps)
        return self.dead_lock_avoidance_agent.act(handle, state, -1.0)

    def save(self, filename):
        self.dead_lock_avoidance_agent.save(filename)
        self.learning_agent.save(filename)
        self.policy_selector.save(filename + '.selector')

    def load(self, filename):
        self.dead_lock_avoidance_agent.load(filename)
        self.learning_agent.load(filename)
        self.policy_selector.load(filename + '.selector')

    def start_step(self, train):
        self.dead_lock_avoidance_agent.start_step(train)
        self.learning_agent.start_step(train)
        self.policy_selector.start_step(train)

    def end_step(self, train):
        self.dead_lock_avoidance_agent.end_step(train)
        self.learning_agent.end_step(train)
        self.policy_selector.end_step(train)

    def start_episode(self, train):
        self.dead_lock_avoidance_agent.start_episode(train)
        self.learning_agent.start_episode(train)
        self.policy_selector.start_episode(train)

    def end_episode(self, train):
        self.dead_lock_avoidance_agent.end_episode(train)
        self.learning_agent.end_episode(train)
        self.policy_selector.end_episode(train)

    def load_replay_buffer(self, filename):
        self.dead_lock_avoidance_agent.load_replay_buffer(filename)
        self.learning_agent.load_replay_buffer(filename)
        self.policy_selector.load_replay_buffer(filename + ".selector")

    def test(self):
        self.dead_lock_avoidance_agent.test()
        self.learning_agent.test()
        self.policy_selector.test()

    def reset(self, env: RailEnv):
        self.env = env
        self.dead_lock_avoidance_agent.reset(env)
        self.learning_agent.reset(env)
        self.policy_selector.reset(env)

    def clone(self):
        return self
