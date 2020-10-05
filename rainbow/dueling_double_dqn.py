import copy
import os
import random
from collections import namedtuple, deque, Iterable, defaultdict
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader, Batch, Data

from model import DQN

BUFFER_SIZE = int(1e4)  # replay buffer size
#BATCH_SIZE = 512  # minibatch size
BATCH_SIZE = 128
GAMMA = 0.99  # discount factor 0.99
TAU = 1e-3  # for soft update of target parameters
LR = 0.001  # learning rate 0.5e-4 works
UPDATE_EVERY = 10  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Available device is " + str(device))


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, obs_builder, agent_messages={}, double_dqn=True):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.state_size = state_size
        self.double_dqn = double_dqn
        self.action_size = action_size
        # Q-Network
        self.qnetwork_local = DQN(state_size).to(device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer("fc", BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.obs_builder = obs_builder

    def save(self, filename):
        print("AGENT: Saving local and target networks")
        torch.save(self.qnetwork_local.state_dict(), filename + "_local.pth")
        torch.save(self.qnetwork_target.state_dict(), filename + "_target.pth")

    def load(self, filename):
        print("filename is: " + filename)
        if os.path.exists(filename + "_local.pth"):
            try:
                if device == torch.device("cpu"):
                    print("Loading local model on cpu")
                    self.qnetwork_local.load_state_dict(torch.load(filename + "_local.pth", map_location=torch.device('cpu')))
                else:
                    self.qnetwork_local.load_state_dict(torch.load(filename + "_local.pth"))
                print("Weights for model {} have been loaded!".format(filename+"_local.pth"))
            except Exception as e:
                print("Weights for model {} couldn't be loaded!".format(filename+ "_local.pth"))
                print(str(e))
        if os.path.exists(filename + "_target.pth"):
            try:
                if device == torch.device("cpu"):
                    print("Loading target model on cpu")
                    self.qnetwork_local.load_state_dict(torch.load(filename + "_target.pth", map_location=torch.device('cpu')))
                else:
                    self.qnetwork_target.load_state_dict(torch.load(filename + "_target.pth"))
                print("Weights for model {} have been loaded!".format(filename + "_target.pth"))
            except Exception as e:
                print("Weights for model {} couldn't be loaded!".format(filename + "_target.pth"))
                print(str(e))

    def step(self, state, reward, next_state, done, deadlock, train=True):
        # Save experience in replay memory
        self.memory.add(state, reward, next_state, done, deadlock)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                if train:
                    self.learn(experiences, GAMMA)

    
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state: current state as graph observation object
            eps (float): epsilon, for epsilon-greedy action selection

        Returns
        =======
            pair of (path, value)
        """
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        batch = state.to(device)
        batch_list = batch.to_data_list()
        self.qnetwork_local.eval()
        agents_best_path_values = dict()
        with torch.no_grad():
            out = self.qnetwork_local(batch.x, batch.edge_index)
            out_mapped = defaultdict(lambda: defaultdict(list))
            for i, res in enumerate(out):
                batch_index = batch.batch[i]
                handle, path = batch_list[batch_index].graph_info
                out_mapped[handle][(path[0], path[1])].append(res)
        for handle in out_mapped.keys():
            paths = []
            for path in out_mapped[handle].keys():
                paths.append([path, out_mapped[handle][path][0]])
            if random.random() > eps:
                best_path = max(paths, key=lambda item:item[1])
                agents_best_path_values.update({handle: best_path})
            else:
                random_index = random.choice(np.arange(len(paths)))
                agents_best_path_values.update({handle: paths[random_index]})
            """
            for path in state.keys():
                node_features, graph_edges = state[path].node_features, state[path].graph_edges
                path_values.append([path, self.qnetwork_local(node_features, graph_edges)])
            """
        self.qnetwork_local.train()

        return agents_best_path_values
       



    def learn(self, experiences, gamma):

        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): 
            gamma (float): discount factor
        """
        
        states, rewards, next_states, dones, deadlocks = experiences

        Q_expected = self.compute_Q_values(states, "local")

        # Choose best paths with local network, then compute value with target network
        selected_next_states = []
        for i, state in enumerate(next_states):
            if not deadlocks[i]:
                preprocessed_next_state = self.obs_builder.preprocess_agent_obs(state, 0) # 0 is just a placeholder value
                path_values = self.act(preprocessed_next_state, eps=0) # Choose path to take at the current switch        
                next_state = state["partitioned"][path_values[0][0]]
            else: 
                next_state = state["partitioned"][0]
            selected_next_states.append(next_state)
         
        # Double DQN
        Q_targets_next = self.compute_Q_values(selected_next_states, "target")
        
        rewards = torch.tensor(rewards).to(device)
        dones = torch.tensor([1 if done else 0 for done in dones]).to(device)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


    def compute_Q_values(self, states, net):
        network = self.qnetwork_local if net=="local" else self.qnetwork_target
        network.train()
        batch_list = []
        for i, state in enumerate(states):
            data = Data(x=state["node_features"], edge_index=state["graph_edges"])
            data.new_to_old_map = state["new_to_old_map"]
            batch_list.append(data)
        batch = Batch.from_data_list(batch_list).to(device)

        # Get expected Q values from local model
        out_q_expected = self.qnetwork_local(batch.x, batch.edge_index).flatten()
        out_mapped = defaultdict(list)
        for i, res in enumerate(out_q_expected):
            batch_index = int(batch.batch[i])
            out_mapped[batch_index].append(res)
        Q_expected = []
        for i in out_mapped.keys():
            Q_expected.append(max(out_mapped[i]))
        Q_expected = torch.stack(Q_expected)
        return Q_expected

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def get_q_values(self, state):
        """
        Used for debugging.
        :param state: 
        :return: 
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        return q_values
        
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, network_type, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.network_type = network_type
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "reward", "next_state", "done", "deadlock"])

    def add(self, state, reward, next_state, done, deadlock):
        """Add a new experience to memory."""
        if state and next_state and reward:
            e = self.experience(state, reward, next_state, done, deadlock)
            self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        """
        states = torch.from_numpy(self.__v_stack_impr([e.state for e in experiences if e is not None])) \
            .float().to(device)
        rewards = torch.from_numpy(self.__v_stack_impr([e.reward for e in experiences if e is not None])) \
            .float().to(device)
        next_states = torch.from_numpy(self.__v_stack_impr([e.next_state for e in experiences if e is not None])) \
            .float().to(device)
        dones = torch.from_numpy(self.__v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(device)
        """
        states = [e.state for e in experiences if e is not None]
        rewards = [e.reward for e in experiences if e is not None]
        next_states = [e.next_state for e in experiences if e is not None]
        dones = [e.done for e in experiences if e is not None]
        deadlocks = [e.deadlock for e in experiences if e is not None]


        return states, rewards, next_states, dones, deadlocks
        
        

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    ''' 
    This same function is used for states, actions, rewards etc, so the parameter 'states' doesn't contain states all the time
    and for this reason output can have different shapes.
    In any case returns a batch of experience according to the specified batch_size.
    '''
'''
    def __v_stack_impr(self, values):
        #sub_dim = len(values[0][0]) if isinstance(values[0], Iterable) else 1
        # values are actually states (not actions, or rewards...)
        if self.network_type == 'fc':
            sub_dim = len(values[0][0]) if isinstance(values[0], Iterable) else 1
            np_values = np.reshape(np.array(values), (len(values), sub_dim))
            return np_values
            
        elif self.network_type == 'conv':
            if isinstance(values[0], Iterable):
                sub_dim = len(values[0][0])
                # Create a 1d array of states and reshape it into (batch_size, in_channels, view_width, view_height)
                # 'states' is a list containing batch_size arrays of shape (1, in_channels, view_width, view_height)
                np_values = np.reshape(np.array(values), (len(values), sub_dim, 15,  30)) # TODO add param env_width env_height
            else:  # values are actions or rewards...
                sub_dim = 1
                # Create a 1d array of values and reshape it into (batch_size, in_channels)
                np_values = np.reshape(np.array(values), (len(values), sub_dim))

        return np_values
'''