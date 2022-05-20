from flatland.envs.rail_env import RailEnv

from reinforcement_learning.dddqn_policy import DDDQNPolicy
from reinforcement_learning.policy import LearningPolicy, DummyMemory
from reinforcement_learning.ppo_agent import PPOPolicy


class MultiDecisionAgent(LearningPolicy):

    def __init__(self, state_size, action_size, in_parameters=None):
        print(">> MultiDecisionAgent")
        super(MultiDecisionAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.in_parameters = in_parameters
        self.memory = DummyMemory()
        self.loss = 0

        self.ppo_policy = PPOPolicy(state_size, action_size, use_replay_buffer=False, in_parameters=in_parameters)
        self.dddqn_policy = DDDQNPolicy(state_size, action_size, in_parameters)
        self.policy_selector = PPOPolicy(state_size, 2)


    def step(self, handle, state, action, reward, next_state, done):
        self.ppo_policy.step(handle, state, action, reward, next_state, done)
        self.dddqn_policy.step(handle, state, action, reward, next_state, done)
        select = self.policy_selector.act(handle, state, 0.0)
        self.policy_selector.step(handle, state, select, reward, next_state, done)

    def act(self, handle, state, eps=0.):
        select = self.policy_selector.act(handle, state, eps)
        if select == 0:
            return self.dddqn_policy.act(handle, state, eps)
        return self.policy_selector.act(handle, state, eps)

    def save(self, filename):
        self.ppo_policy.save(filename)
        self.dddqn_policy.save(filename)
        self.policy_selector.save(filename)

    def load(self, filename):
        self.ppo_policy.load(filename)
        self.dddqn_policy.load(filename)
        self.policy_selector.load(filename)

    def start_step(self, train):
        self.ppo_policy.start_step(train)
        self.dddqn_policy.start_step(train)
        self.policy_selector.start_step(train)

    def end_step(self, train):
        self.ppo_policy.end_step(train)
        self.dddqn_policy.end_step(train)
        self.policy_selector.end_step(train)

    def start_episode(self, train):
        self.ppo_policy.start_episode(train)
        self.dddqn_policy.start_episode(train)
        self.policy_selector.start_episode(train)

    def end_episode(self, train):
        self.ppo_policy.end_episode(train)
        self.dddqn_policy.end_episode(train)
        self.policy_selector.end_episode(train)

    def load_replay_buffer(self, filename):
        self.ppo_policy.load_replay_buffer(filename)
        self.dddqn_policy.load_replay_buffer(filename)
        self.policy_selector.load_replay_buffer(filename)

    def test(self):
        self.ppo_policy.test()
        self.dddqn_policy.test()
        self.policy_selector.test()

    def reset(self, env: RailEnv):
        self.ppo_policy.reset(env)
        self.dddqn_policy.reset(env)
        self.policy_selector.reset(env)

    def clone(self):
        multi_descision_agent = MultiDecisionAgent(
            self.state_size,
            self.action_size,
            self.in_parameters
        )
        multi_descision_agent.ppo_policy = self.ppo_policy.clone()
        multi_descision_agent.dddqn_policy = self.dddqn_policy.clone()
        multi_descision_agent.policy_selector = self.policy_selector.clone()
        return multi_descision_agent
