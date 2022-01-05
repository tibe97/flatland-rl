import copy
import os
import random
from collections import namedtuple, deque, Iterable, defaultdict
from datetime import datetime
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data
from prioritized_memory import Memory
import pdb
# import torchsummary

from model import DQN_action, DQN_value, GAT_action, GAT_value, FC_action

BUFFER_SIZE = int(1e4)  # replay buffer size
# BATCH_SIZE = 512  # minibatch size
GAMMA = 0.99  # discount factor 0.99
TAU = 1e-3  # for soft update of target parameters
# LR = 0.001  # learning rate 0.5e-4 works
UPDATE_EVERY = 15  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Available device is " + str(device))


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, args, state_size, obs_builder):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.args = args
        self.state_size = state_size
        # Q-Network
        #self.qnetwork_value_local = DQN_value(state_size).to(device)
        self.qnetwork_value_local = GAT_value(12, 10, 1, args.gat_layers, args.dropout_rate, 0.3, args.attention_heads, args.flow, args.batch_norm).to(device)
        self.qnetwork_value_target = copy.deepcopy(self.qnetwork_value_local)
        
        #self.qnetwork_action = DQN_action(state_size).to(device)
        #self.qnetwork_action = GAT_action(14, 10, 2, args.gat_layers, args.dropout_rate, 0.3, args.attention_heads, args.flow, args.batch_norm).to(device)
        self.qnetwork_action = FC_action(3, 32, 12, 2).to(device)
        
        self.learning_rate = args.learning_rate
        self.optimizer_value = optim.Adam(self.qnetwork_value_local.parameters(), lr=self.learning_rate)
        self.optimizer_action = optim.Adam(self.qnetwork_action.parameters(), lr=self.learning_rate)
        self.evaluation_mode = False
        # Replay memory
        #self.memory = ReplayBuffer("fc", BUFFER_SIZE, args.batch_size)
        self.memory = Memory(BUFFER_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.obs_builder = obs_builder
        self.batch_size = args.batch_size
        self.use_stop_action = args.use_stop_action
     

    def eval(self):
        self.evaluation_mode = True
        self.qnetwork_value_local.eval()
        self.qnetwork_value_target.eval()
        self.qnetwork_action.eval()

    def train(self):
        self.evaluation_mode = False
        self.qnetwork_value_local.train()
        self.qnetwork_value_target.train()
        self.qnetwork_action.train()

    def save(self, filename):
        print("AGENT: Saving local and target networks")
        torch.save(self.qnetwork_value_local.state_dict(), filename + "_value_local.pth")
        torch.save(self.qnetwork_value_target.state_dict(), filename + "_value_target.pth")
        torch.save(self.qnetwork_action.state_dict(), filename + "_action.pth")
        #print(self.qnetwork_local.conv1.mlp[0].weight)

    def load(self, filename):
        print("filename is: " + filename)
        if os.path.exists(filename + "_value_local.pth"):
            try:
                if device == torch.device("cpu"):
                    print("Loading local model on cpu")
                    self.qnetwork_value_local.load_state_dict(torch.load(
                        filename + "_value_local.pth", map_location=torch.device('cpu')))
                    #print(self.qnetwork_local.conv1.mlp[0].weight)
                else:
                    self.qnetwork_value_local.load_state_dict(
                        torch.load(filename + "_value_local.pth"))
                print("Weights for model {} have been loaded!".format(
                    filename+"_value_local.pth"))
            except Exception as e:
                print("Weights for model {} couldn't be loaded!".format(
                    filename + "_value_local.pth"))
                print(str(e))
        if os.path.exists(filename + "_value_target.pth"):
            try:
                if device == torch.device("cpu"):
                    print("Loading target model on cpu")
                    self.qnetwork_value_target.load_state_dict(torch.load(
                        filename + "_value_target.pth", map_location=torch.device('cpu')))
                else:
                    self.qnetwork_value_target.load_state_dict(
                        torch.load(filename + "_value_target.pth"))
                print("Weights for model {} have been loaded!".format(
                    filename + "_value_target.pth"))
            except Exception as e:
                print("Weights for model {} couldn't be loaded!".format(
                    filename + "_value_target.pth"))
                print(str(e))
        if os.path.exists(filename + "_action.pth"):
            try:
                if device == torch.device("cpu"):
                    print("Loading target model on cpu")
                    self.qnetwork_action.load_state_dict(torch.load(
                        filename + "_action.pth", map_location=torch.device('cpu')))
                else:
                    self.qnetwork_action.load_state_dict(
                        torch.load(filename + "_action.pth"))
                print("Weights for model {} have been loaded!".format(
                    filename + "_action.pth"))
            except Exception as e:
                print("Weights for model {} couldn't be loaded!".format(
                    filename + "_action.pth"))
                print(str(e))

    def step(self, state, reward, next_state, done, deadlock, mean_field, next_q_value, ep=0, train=True):
        # Save experience in replay memory
        # Logarithmic scaling
        
        reward = np.log10(abs(reward)+1) * np.sign(reward)

        if np.isnan(reward):
            return None
        if reward is None:
            return None
        
        #self.memory.add(state, reward, next_state, done, deadlock)
        self._append_sample(state, reward, next_state, mean_field, next_q_value, done, deadlock)

        '''
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            #print("Dones: {}, deadlock: {}, other: {}".format(len(self.memory.memory_dones), len(self.memory.memory_deadlocks), len(self.memory.memory_other)))
            if (len(self.memory.memory_dones) > BATCH_SIZE//10*2 and len(self.memory.memory_deadlocks) > BATCH_SIZE//10*3 and len(self.memory.memory_other) > BATCH_SIZE//10*5) or (len(self.memory.memory_dones) > BATCH_SIZE//2 and len(self.memory.memory_other) > BATCH_SIZE//2):
                experiences = self.memory.sample()
                if train:
                    return self.learn(experiences, GAMMA, ep)
        '''
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if self.memory.tree.n_entries >= 1000:
                return self.learn(GAMMA, ep)
      
        return None
        
    def act(self, state, mean_field, eps=0.0, eval=True):
        """ Returns for each agent the path to take and whether to stop or go.
            2 separate networks (we should try to unify)

        Params
        ======
            state: current state as graph observation object
            eps (float): epsilon, for epsilon-greedy action selection

        Returns
        =======
            pair of (path, value)
        """
        # the batch groups together the nodes of different paths
        batch = state.to(device)
        batch_list = batch.to_data_list()
        agents_best_path_values = dict()
        if eval:
            self.qnetwork_value_local.eval()
            with torch.no_grad():
                out_value = self.qnetwork_value_local(batch.x, batch.edge_index) 
        else:
            # switch from qnetwork value local to target to implement Fixed Q-learning
            #out_value = self.qnetwork_value_local(batch.x, batch.edge_index) 
            out_value = self.qnetwork_value_target(batch.x, batch.edge_index)

        #out_action = self.qnetwork_action(batch.x, batch.edge_index)
        action_x = torch.cat([out_value, mean_field.repeat(out_value.shape[0], 1)], dim=1)
        out_action = self.qnetwork_action(action_x) # qvalue + mean_field
        #print("action_x: {}, action_out: {}".format(action_x, out_action))

        out_mapped = defaultdict(lambda: defaultdict(list))
        for i, res in enumerate(zip(out_value, out_action)): # RES = (node_value, (value_stop, value_go))
            batch_index = batch.batch[i] # batch array tells you to which graph each feature belongs to 
            handle, path = batch_list[batch_index].graph_info
            out_mapped[handle][(path[0], path[1])].append(res) # path[0] = path_ID, path[1] = path_variation (through the switch)

        for handle in out_mapped.keys():
            paths = []
            action_prob = None
            for path in out_mapped[handle].keys():
                if self.use_stop_action: # current node where the agent is represents STOP action
                    paths.append([path, out_mapped[handle][path][0]])
                else: 
                    if path[0] != 0: # Skip current node, i.e. skip STOP action
                        paths.append([path, out_mapped[handle][path][0]])   
                action_prob = out_mapped[handle][(0,0)][0]
            if self.evaluation_mode: # test time
                best_path = max(paths, key=lambda item: item[1][0])
                m = Categorical(action_prob[1])
                action = m.sample()
                log_prob = m.log_prob(action)
                agents_best_path_values.update({handle: [best_path[0], action.item(), log_prob, best_path[1][0], m.probs[action]]})
            else:
                if random.random() >= eps:
                    best_path = max(paths, key=lambda item: item[1][0])
                    m = Categorical(action_prob[1])
                    action = m.sample()
                    log_prob = m.log_prob(action)
                    agents_best_path_values.update({handle: [best_path[0], action.item(), log_prob, best_path[1][0], m.probs[action]]})
                else:
                    random_index = random.choice(np.arange(len(paths)))
                    random_path = paths[random_index]
                    m = Categorical(action_prob[1])
                    action = m.sample()
                    log_prob = m.log_prob(action)
                    agents_best_path_values.update({handle: [random_path[0], action.item(), log_prob, random_path[1][0], m.probs[action]]})
           
        self.qnetwork_value_local.train()
        self.qnetwork_action.train()

        return agents_best_path_values
    
    def _append_sample(self, state, reward, next_state, mean_field, next_q_value, done, deadlock):
        '''
            Compute error for prioritized experience buffer and append to buffer
        '''
        with torch.no_grad():
            Q_expected = self.compute_Q_values([state], [mean_field], "local")

        # Choose best paths with local network, then compute value with target network
        if not deadlock and not done:
            #preprocessed_next_state = self.obs_builder.preprocess_agent_obs(
            #    next_state, 0)  # 0 is just a placeholder value for the agent (we don't care which one here)     
            # Choose path to take at the current switch
            #path_values = self.act(preprocessed_next_state, eps=0)
            #S_value = path_values[0][3] # selected node value, which is the path with max value
            #Q_target = reward + GAMMA * S_value
            # MULTI AGENT
            Q_target = next_q_value
        else: # if agent is done or is in deadlock, we don't need to compute any value. Value is reward at the end
            Q_target = torch.tensor([reward]).to(device) # append random value,it won't be considered

        error = abs(Q_expected - Q_target).cpu().numpy()

        self.memory.add(error, (state, reward, next_state, mean_field.cpu(), next_q_value.cpu(), done, deadlock))
        

    def learn(self, gamma, ep):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]):
            gamma (float): discount factor
        """
        print("------LEARNING------\n")
        batch, idxs, is_weights = self.memory.sample(self.batch_size)
        batch = np.array(batch).transpose()
        states = list(batch[0])
        rewards = list(batch[1])
        next_states = list(batch[2])
        mean_fields = list(batch[3])
        next_q_values = list(batch[4])
        dones = list(batch[5])
        deadlocks = list(batch[6])
        
        Q_expected = self.compute_Q_values(states, mean_fields, "local")
   
        # Choose best paths with local network, then compute value with target network
        #selected_next_states = []
        Q_targets_next = next_q_values
        '''
        for i, state in enumerate(next_states):
            if not deadlocks[i] and not dones[i]:
                preprocessed_next_state = self.obs_builder.preprocess_agent_obs(
                    state, 0)  # 0 is just a placeholder value for the agent (we don't care which one here)
                # Choose path to take at the current switch
                path_values = self.act(preprocessed_next_state, eps=0, eval=False)
                S_value = path_values[0][3] # selected node value, which is the path with max value
                Q_targets_next.append(S_value)
            else: # if agent is done or is in deadlock, we don't need to compute any value. Value is reward at the end
                Q_targets_next.append(torch.tensor([rewards[i]])) # append random value,it won't be considered
        '''
            
        # Double DQN
        Q_targets_next = [target.to(device) for target in Q_targets_next]
        Q_targets_next = torch.stack(Q_targets_next).squeeze()

        '''
        rewards = torch.tensor(rewards).to(device)
        # convert true/false in integers
        dones = torch.tensor([1 if done else 0 for done in dones]).to(device)
        deadlocks = torch.tensor([1 if deadlock else 0 for deadlock in deadlocks]).to(device)
        '''
        rewards = torch.tensor(rewards).to(device)
        dones = torch.tensor([1 if done else 0 for done in dones]).to(device)
        deadlocks = torch.tensor([1 if deadlock else 0 for deadlock in deadlocks]).to(device)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones) * (1 - deadlocks))


        errors = torch.abs(Q_expected - Q_targets).data.cpu().numpy()
        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])
    
        self.optimizer_value.zero_grad()
        # Compute loss
        #loss = F.mse_loss(Q_expected, Q_targets)
        #loss = F.smooth_l1_loss(Q_expected, Q_targets) # Huber loss
        Q_targets = Q_targets.type(torch.cuda.FloatTensor) 
        loss = (torch.FloatTensor(is_weights).to(device) * F.mse_loss(Q_targets, Q_expected)).mean()
        loss_to_return = loss.item() 
        loss.backward()
        # Clip gradients - https://stackoverflow.com/questions/47036246/dqn-q-loss-not-converging
        for param in self.qnetwork_value_local.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer_value.step()
        '''
        for param in self.qnetwork_value_local.parameters():
            print(param.data)
        '''
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_value_local, self.qnetwork_value_target, TAU)
        
        return loss_to_return

    def learn_actions(self, log_probs, ending_step, dones, max_steps, ep, gamma=1.0):
        '''
            Network to learn STOP/GO action for each reachable path.
            Compute the rewards to assign to each trajectory (between -1 and 1).
            If agent reaches destination, the reward is inversely proportional to time taken.
            Faster the agent, higher the reward.
            -1 if agent in deadlock or doesn't finish.
        '''
        for a in range(len(log_probs)):
            policy_loss = []
            if dones[a]:
                agent_reward = (max_steps-ending_step[a])/max_steps
            else:
                agent_reward = -1
            for log_prob in log_probs[a]:
                policy_loss.append(-log_prob * agent_reward)
        policy_loss = torch.stack(policy_loss).sum() / len(log_probs)

        self.optimizer_action.zero_grad()
        policy_loss.backward()
        self.optimizer_action.step()


    def compute_Q_values(self, states, mean_fields, net):
        ''' 
            Used at learning time to compute values of states/paths.
            The value of each single path is represented by the value of the first node, so the first output
            for that graph. 
        '''
        network = self.qnetwork_value_local if net == "local" else self.qnetwork_value_target
        network.train()
        batch_list = []
        
        # create batch with all the states
        for i, state in enumerate(states):
            data = Data(x=state["node_features"],
                        edge_index=state["graph_edges"]).to(device)
            #new_x = torch.cat([data.x, mean_fields[i].to(device).repeat(data.x.shape[0], 1)], dim=1)
            #data.x = new_x
            data.new_to_old_map = state["new_to_old_map"]
            batch_list.append(data)
        batch = Batch.from_data_list(batch_list).to(device)

        # Get expected Q values from local or target model
        out_q_expected = network(batch.x, batch.edge_index).flatten()
        
        out_mapped = defaultdict(list)
        # TODO: Da sistemare, è buggato. per ora non scelgo il max tra i nodi vicini ma il max tra tutti i nodi
        # Inoltre, per q_expected devo prendere solo il nodo attuale, ma per q_next devo prendere il max tra i vicini.
        # SCEMO!
        for i, res in enumerate(out_q_expected):
            batch_index = int(batch.batch[i])
            out_mapped[batch_index].append(res)
        Q_expected = []
        for i in out_mapped.keys():
            Q_expected.append(out_mapped[i][0])
        Q_expected = torch.stack(Q_expected)
        return Q_expected.type(torch.cuda.FloatTensor)

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
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

   

class ReplayBuffer:
    """
        Fixed-size buffer to store experience tuples.
        TODO: implement experience replay
    """

    def __init__(self, network_type, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.network_type = network_type
        self.memory_dones = deque(maxlen=buffer_size) # save done experiences
        self.memory_deadlocks = deque(maxlen=buffer_size) # save deadlock experiences
        self.memory_other = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        # self.experience = namedtuple("Experience", field_names=["state", "reward", "next_state", "done", "deadlock"])

    def add(self, state, reward, next_state, done, deadlock):
        """Add a new experience to memory."""
        if state and next_state and reward is not None:
            # e = self.experience(state, reward, next_state, done, deadlock)
            reward = reward
            e = (state, reward, next_state, done, deadlock)
            if done:
                self.memory_dones.append(e)
            elif deadlock: 
                self.memory_deadlocks.append(e)
                
            else:
                self.memory_other.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        states = []
        rewards = []
        next_states = []
        dones = []
        deadlocks = []

        if len(self.memory_dones) > self.batch_size//10*2 and len(self.memory_deadlocks) > self.batch_size//10*3 and len(self.memory_other) > self.batch_size//10*5:
            dones_size = self.batch_size//10*2
            deadlocks_size = self.batch_size//10*3
            others_size = self.batch_size - dones_size - deadlocks_size
        else: # we are possibly training only 1 agent
            dones_size = self.batch_size//2
            deadlocks_size = 0
            others_size = self.batch_size - dones_size

        experiences = random.sample(self.memory_dones, k=dones_size)
        
        states += [e[0] for e in experiences if e is not None]
        rewards += [e[1] for e in experiences if e is not None]
        next_states += [e[2] for e in experiences if e is not None]
        dones += [e[3] for e in experiences if e is not None]
        deadlocks += [e[4] for e in experiences if e is not None]

        experiences = random.sample(self.memory_deadlocks, k=deadlocks_size)
        
        states += [e[0] for e in experiences if e is not None]
        rewards += [e[1] for e in experiences if e is not None]
        next_states += [e[2] for e in experiences if e is not None]
        dones += [e[3] for e in experiences if e is not None]
        deadlocks += [e[4] for e in experiences if e is not None]

        experiences = random.sample(self.memory_other, k=others_size)
        
        states += [e[0] for e in experiences if e is not None]
        rewards += [e[1] for e in experiences if e is not None]
        next_states += [e[2] for e in experiences if e is not None]
        dones += [e[3] for e in experiences if e is not None]
        deadlocks += [e[4] for e in experiences if e is not None]

        return states, rewards, next_states, dones, deadlocks

    def load_memory(self, memory_path):
        try:
            with open(memory_path+"_dones.pickle", 'rb') as pickle_file:
                self.memory_dones = pickle.load(pickle_file)
            with open(memory_path+"_deadlocks.pickle", 'rb') as pickle_file:
                self.memory_deadlocks = pickle.load(pickle_file)
            with open(memory_path+"_other.pickle", 'rb') as pickle_file:
                self.memory_other = pickle.load(pickle_file)
            print("Replay memory loaded")
        except Exception as e:
            print("Impossible to load replay memory.")
            print(e)
        


    def save_memory(self, memory_path):
        try:
            with open(memory_path+"_dones.pickle", 'wb') as pickle_file:
                pickle.dump(self.memory_dones, pickle_file)
            with open(memory_path+"_deadlocks.pickle", 'wb') as pickle_file:
                pickle.dump(self.memory_deadlocks, pickle_file)
            with open(memory_path+"_other.pickle", 'wb') as pickle_file:
                pickle.dump(self.memory_other, pickle_file)
            print("Replay memory saved")
        except Exception as e:
            print("Impossible to save replay memory")
            print(e)

 
