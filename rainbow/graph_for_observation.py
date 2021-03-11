import collections
from typing import Optional, List, Dict, Tuple
import queue
import numpy as np
from collections import defaultdict, namedtuple, deque
import math
import torch
import matplotlib.pyplot as plt
import seaborn as sns;
import logging
import itertools
from copy import copy
from sklearn import preprocessing

from flatland.core.env import Environment
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
from flatland.utils.ordered_set import OrderedSet
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env import RailEnvNextAction, RailEnvActions
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.envs.rail_trainrun_data_structures import Waypoint
# , get_action_for_move
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_, get_k_shortest_paths, get_new_position_for_action, get_shortest_paths, get_action_for_move
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position, distance_on_rail, position_to_coordinate

import torch.utils.data as D
from torch_geometric.data import Data
from torch_geometric.data import DataLoader, Batch
import torch.nn.functional as F
import wandb

IntersectionBranch = namedtuple("IntersectionBranch", "track_id, orientation")

class EpisodeController():
    def __init__(self, env, agent, max_steps):
        super(EpisodeController, self).__init__()
        self.env = env
        self.rl_agent = agent
        self.scores_window = deque(maxlen=100)
        self.done_window = deque(maxlen=100)
        self.score = 0
        self.max_steps = max_steps
        self.path_values_buffer = [] # to compute mean path value for debugging
        self.agent_obs = [None] * self.env.get_num_agents()
        self.agent_obs_buffer = [None] * self.env.get_num_agents()  # updated twice
        self.agent_action_buffer = defaultdict(list)
        # some switches are long so we have to accumulate reward or penalty
        self.acc_rewards = defaultdict(lambda: 0)
        # how many timesteps remaining to complete current action
        self.agents_speed_timesteps = [0] * self.env.get_num_agents()
        self.agent_at_switch = [False] * self.env.get_num_agents()
        self.agent_path_obs_buffer = [None] * self.env.get_num_agents()
        # Used to check if agent with fractionary speed could move at current cell
        self.agent_old_speed_data = {}
        self.agent_done_removed = [False] * self.env.get_num_agents()
        self.update_values = [False] * self.env.get_num_agents()
        self.agents_in_deadlock = [False] * self.env.get_num_agents()
        self.epoch_loss = []
        self.rewards_buffer = [list()] * self.env.get_num_agents()
        self.log_probs_buffer = [list()] * self.env.get_num_agents()
        self.probs_buffer = [list()] * self.env.get_num_agents()
        self.stop_go_buffer = [list()] * self.env.get_num_agents()
        self.agent_ending_timestep = [self.max_steps] * self.env.get_num_agents()
        self.num_agents_at_switch = 0 # number of agents at switches
        self.stats = defaultdict(list)
        self.epoch_mean_loss = None

    def reset(self):
        self.score = 0
        self.path_values_buffer = [] # to compute mean path value for debugging
        self.agent_obs = [None] * self.env.get_num_agents()
        self.agent_obs_buffer = [None] * self.env.get_num_agents()  # updated twice
        self.agent_action_buffer = defaultdict(list)
        # some switches are long so we have to accumulate reward or penalty
        self.acc_rewards = defaultdict(lambda: 0)
        # how many timesteps remaining to complete current action
        self.agents_speed_timesteps = [0] * self.env.get_num_agents()
        self.agent_at_switch = [False] * self.env.get_num_agents()
        self.agent_path_obs_buffer = [None] * self.env.get_num_agents()
        # Used to check if agent with fractionary speed could move at current cell
        self.agent_old_speed_data = {}
        self.agent_done_removed = [False] * self.env.get_num_agents()
        self.update_values = [False] * self.env.get_num_agents()
        self.agents_in_deadlock = [False] * self.env.get_num_agents()
        self.epoch_loss = []
        self.rewards_buffer = [list()] * self.env.get_num_agents()
        self.log_probs_buffer = [list()] * self.env.get_num_agents()
        self.probs_buffer = [list()] * self.env.get_num_agents()
        self.stop_go_buffer = [list()] * self.env.get_num_agents()
        self.agent_ending_timestep = [self.max_steps] * self.env.get_num_agents()
        self.num_agents_at_switch = 0 # number of agents at switches
        self.stats = defaultdict(list)
        self.epoch_mean_loss = None
    
    def is_episode_done(self):
        return self.agent_done_removed.count(True) == self.env.get_num_agents()

    def compute_agent_action(self, handle, info, eps):
        ''' 
            If agent is arriving at switch we need to compute the next path to reach in advance.
            The agent computes a sequence of actions because the switch could be composed of more cells.
            Each action (e.g. MOVE_FORWARD) could take more than 1 timestep if speed is not 1.
            Each action has a fixed TIME_STEPS value for each agent, computed as 1/agent_speed.
            At each environment step we subtract 1 from the remaining timestep for a certain action of an agent.
            When we finish all the actions we have reached the new node, here the only allowed action is MOVE_FORWARD
            until we reach the next switch.
        '''
        next_action = 0
        agent = self.env.agents[handle]

        # Compute agent position
        if agent.status == RailAgentStatus.ACTIVE:
            agent_position = agent.position
        elif agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_position = agent.initial_position

        if agent.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
            logging.debug("Agent {} is done".format(handle))
        else:
            logging.debug("Agent {} at position {}, fraction {}, status {}".format(
                handle, agent.position, agent.speed_data["position_fraction"], agent.status))


        # If the agent arrives at a switch
        if info['action_required'][handle] and self.agent_at_switch[handle] and agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.READY_TO_DEPART] and agent.malfunction_data["malfunction"] == 0:
            # if we finish previous action (action may take more than 1 timestep)
            if self.agents_speed_timesteps[handle] == 0:
                # if we arrive before a switch we compute the next path to reach and the actions required
                # (this is due to the switch being eventually composed of more cells)
                # We are about to enter the switch
                if len(self.agent_action_buffer[handle]) == 0:
                    # Check that dict is not empty
                    assert self.agent_obs[handle]["partitioned"]
                    obs_batch = self.env.obs_builder.preprocess_agent_obs(self.agent_obs[handle], handle)
                    # Choose path to take at the current switch
                    path_values = self.rl_agent.act(obs_batch, eps=eps)
                    self.log_probs_buffer[handle].append(path_values[handle][2])
                    self.probs_buffer[handle].append(path_values[handle][4])
                    self.stop_go_buffer[handle].append(path_values[handle][1])
                    railenv_action = self.env.obs_builder.choose_railenv_actions(handle, path_values[handle])
                    self.agent_action_buffer[handle] = railenv_action
                    # as state to save we take the path (with its children) chosen by agent
                    self.agent_path_obs_buffer[handle] = self.agent_obs[handle]["partitioned"][path_values[handle][0]]
                    logging.debug("Agent {} choses path {} with value {} at position {}. Num actions to take: {}".format(
                        handle, path_values[handle][0][0], path_values[handle][3], agent.position, len(self.agent_action_buffer[handle])))
                    self.path_values_buffer.append(path_values[handle][3]) # for debug 
                    
                    logging.debug(
                        "Agent {} actions: {}".format(handle, railenv_action))
                next_action = self.agent_action_buffer[handle].pop(0)
                
                logging.debug("Agent {} at: {}. Action is: {}. Speed: {}. Fraction {}. Remaining actions: {}. SpeedTimesteps: {}".format(
                    handle, agent.position, next_action, agent.speed_data["speed"], agent.speed_data["position_fraction"], len(self.agent_action_buffer[handle]), self.agents_speed_timesteps[handle]))
                # if agent has to stop, do it for 1 timestep
                if (next_action == RailEnvActions.STOP_MOVING):
                    self.agents_speed_timesteps[handle] = 1
                    self.env.obs_builder.agent_requires_obs.update({handle: True})
                else:
                    # speed is a fractionary value between 0 and 1
                    self.agents_speed_timesteps[handle] = int(round(1 / info["speed"][handle]))
        # if agent is not at switch just go straight
        elif agent.status != RailAgentStatus.DONE_REMOVED:  
            next_action = 0
            if agent.status == RailAgentStatus.READY_TO_DEPART or (not agent.moving and agent.malfunction_data["malfunction"] == 0):
                valid_move_actions = get_valid_move_actions_(agent.direction, agent_position, self.env.rail)
                # agent could be at switch, so more actions possible
                assert len(valid_move_actions) >= 1
                next_action = valid_move_actions.popitem()[0].action

        return next_action
    
    def save_experience_and_train(self, a, action, reward, next_obs, done, step, args, ep):
        '''
            In the first part we perform an agent step (save experience and possibly learn) only if agent 
            was able to move (no agent blocked his action).
        '''
        agent = self.env.agents[a]
        if not self.agent_done_removed[a]:
            logging.debug("Agent {} at position {}, fraction {}, speed Timesteps {}, reward {}".format(a, agent.position, agent.speed_data["position_fraction"], self.agents_speed_timesteps[a], self.acc_rewards[a]))

        self.score += reward / self.env.get_num_agents()  # Update score

        # if agent didn't move do nothing: agent couldn't perform action because another agent
        # occupied next cell or agent's action was STOP
        if self.env.obs_builder.agent_could_move(a, action, self.agent_old_speed_data[a]):
            # update replay memory
            self.acc_rewards[a] += reward
            if ((self.update_values[a] and agent.speed_data["position_fraction"] == 0) or agent.status == RailAgentStatus.DONE_REMOVED) and not self.agent_done_removed[a]:
                logging.debug("Update=True: agent {}".format(a))
                # next state is the complete state, with all the possible path choices
                if len(next_obs) > 0 and self.agent_path_obs_buffer[a] is not None:
                    # if agent reaches target
                    if agent.status == RailAgentStatus.DONE_REMOVED or agent.status == RailAgentStatus.DONE:
                        self.agent_done_removed[a] = True
                        self.acc_rewards[a] = args.done_reward
                        self.agent_ending_timestep[a] = step
                        logging.debug("Agent {} DONE! It has been removed and experience saved with reward of {}!".format(a, self.acc_rewards[a]))
                    else: 
                        logging.debug("Agent reward is {}".format(self.acc_rewards[a]))
                    # step saves experience tuple and can perform learning (every T time steps)
                    step_loss = self.rl_agent.step(self.agent_path_obs_buffer[a], self.acc_rewards[a], next_obs, self.agent_done_removed[a], self.agents_in_deadlock[a], ep=ep)
                    
                    # save stats
                    if step_loss is not None:
                        self.epoch_loss.append(step_loss)
                    if self.agent_done_removed[a]:
                        self.rewards_buffer[a].append(0)
                    else:
                        self.rewards_buffer[a].append(self.acc_rewards[a])
                    self.acc_rewards[a] = 0
                    self.update_values[a] = False
                    
                    
            if len(next_obs) > 0:
                # prepare agent obs for next timestep
                self.agent_obs[a] = next_obs.copy()

            if self.agent_at_switch[a]:
                # we descrease timestep if agent is performing actions at switch
                self.agents_speed_timesteps[a] -= 1
                
            """
                We want to optimize computation of observations only when it's needed, i.e. before 
                making a decision, to accelerate simulation.
                We update the dictionary AGENT_REQUIRED_OBS to tell the ObservationBuilder for which agent to compute obs.
                We compute observations only in these cases:
                1. Agent is entering switch (obs for last cell of current path): we need obs to evaluate which path
                    to take next
                2. Agent is exiting a switch (obs for new cell of new path): we compute the obs because we could immediately
                    meet another switch (track section only has 1 cell), so we need the observation in buffer
                3. Agent is about to finish: we compute obs to save the experience tuple
            """
            if agent.status == RailAgentStatus.ACTIVE:
                # Compute when agent is about to enter a switch and when it's about to leave a switch
                # PURPOSE: to compute observations only when needed, i.e. before a switch and after, also before and
                # after making an action that leads to the target of an agent
                if not self.agent_at_switch[a]:
                    agent_pos = agent.position
                    assert self.env.obs_builder.get_track(agent_pos) != -2
                    if self.env.obs_builder.is_agent_entering_switch(a) and agent.speed_data["position_fraction"] == 0:
                        logging.debug("Agent {} arrived at 1 cell before switch".format(a))
                        self.agent_at_switch[a] = True
                        self.agents_speed_timesteps[a] = 0
                        # env.obs_builder.agent_requires_obs.update({a: False})
                    elif self.env.obs_builder.is_agent_2_steps_from_switch(a):
                        self.env.obs_builder.agent_requires_obs.update({a: True})
                        self.update_values[a] = True
                    if self.env.obs_builder.is_agent_about_to_finish(a):
                        self.env.obs_builder.agent_requires_obs.update({a: True})
                else:  # Agent at SWITCH. In the step before reaching target path we want to make sure to compute the obs
                    # in order to update the replay memory. We need to be careful if the agent can't reach new path because of another agent blocking the cell.
                    # when agent speed is 1 we reach the target in 1 step
                    if len(self.agent_action_buffer[a]) == 1 and agent.speed_data["speed"] == 1:
                        # agent_next_action = agent_action_buffer[a][0]
                        # assert env.obs_builder.is_agent_exiting_switch(a, agent_next_action)
                        #update_values[a] = True
                        self.env.obs_builder.agent_requires_obs.update({a: True})

                    # if speed is less than 1, we need more steps to reach target. So only compute obs if doing last step
                    elif len(self.agent_action_buffer[a]) == 0:
                        if self.env.obs_builder.get_track(agent.position) == -2 and agent.speed_data["speed"] < 1 and np.isclose(agent.speed_data["speed"] + agent.speed_data["position_fraction"], 1, rtol=1e-03):
                            # same check as "if" condition
                            assert self.agents_speed_timesteps[a] > 0
                            #update_values[a] = True
                            self.env.obs_builder.agent_requires_obs.update({a: True})
                        else:
                            if self.env.obs_builder.get_track(agent.position) != -2:
                                if self.env.obs_builder.is_agent_entering_switch(a):
                                    assert len(next_obs) > 0
                                    logging.debug("Agent {} just exited switch and ALREADY entering another one".format(a))
                                    self.agent_obs_buffer[a] = next_obs.copy()
                                    self.update_values[a] = True
                                else:
                                    logging.debug("Agent {} is not at switch anymore".format(a))
                                    self.agent_at_switch[a] = False
                                    self.agents_speed_timesteps[a] = 0
                                    self.agent_obs_buffer[a] = next_obs.copy()
                                if self.env.obs_builder.is_agent_about_to_finish(a):
                                    self.env.obs_builder.agent_requires_obs.update(
                                        {a: True})

        else:  # agent did not move. Check if it stopped on purpose or it's in deadlock
            if action == RailEnvActions.STOP_MOVING:
                self.agents_speed_timesteps[a] -= 1
                self.env.obs_builder.agent_requires_obs.update({a: True})
            else:
                logging.debug("Agent {} cannot move at position {}, fraction {}".format(
                    a, agent.position, agent.speed_data["position_fraction"]))
                # check if agent is in deadlock
                if self.env.obs_builder.is_agent_in_deadlock(a) and not self.agents_in_deadlock[a]: # agent just got in deadlock
                    self.env.obs_builder.agent_requires_obs.update({a: True})
                    logging.debug("Agent {} in DEADLOCK saved as experience with reward of {}".format(
                        a, self.acc_rewards[a]))
                    if len(next_obs) > 0 and self.agent_path_obs_buffer[a] is not None:
                        self.agent_obs_buffer[a] = next_obs
                        self.acc_rewards[a] = args.deadlock_reward
                        self.agents_in_deadlock[a] = True
                        step_loss = self.rl_agent.step(self.agent_path_obs_buffer[a], self.acc_rewards[a], self.agent_obs_buffer[a], done, self.agents_in_deadlock[a], ep=ep)
                        if step_loss is not None:
                            self.epoch_loss.append(step_loss)
                        self.env.obs_builder.agent_requires_obs.update({a: False})
                    logging.debug("Agent {} is in DEADLOCK, accum. reward: {}, required_obs: {}".format(a, self.acc_rewards[a], self.env.obs_builder.agent_requires_obs[a]))
        self.agent_old_speed_data.update({a: agent.speed_data.copy()})

    def print_episode_stats(self, ep, args, eps, step):
        # Metrics
        num_agents_done = 0  # Num of agents that reached their target
        num_agents_in_deadlock = 0
        num_agents_not_started = 0
        num_agents_in_deadlock_at_switch = 0
        env_done = 0
        for a in range(self.env.get_num_agents()):
            if self.env.agents[a].status in [RailAgentStatus.DONE_REMOVED, RailAgentStatus.DONE]:
                num_agents_done += 1
            elif self.env.agents[a].status == RailAgentStatus.READY_TO_DEPART:
                num_agents_not_started += 1
            elif self.env.obs_builder.is_agent_in_deadlock(a):
                num_agents_in_deadlock += 1
                if self.env.obs_builder.get_track(self.env.agents[a].position) == -2:
                    num_agents_in_deadlock_at_switch += 1
        if num_agents_done == self.env.get_num_agents():
            env_done = 1
        

        self.scores_window.append(self.score / self.max_steps)  # Save most recent score
        self.done_window.append(env_done)
        if len(self.epoch_loss) > 0:
            self.epoch_mean_loss = (sum(self.epoch_loss)/(len(self.epoch_loss)))

        


        # Print training results info
        episode_stats = '\rEp: {}\t {} Agents on ({},{}).\t Ep score {:.3f}\tAvg Score: {:.3f}\t Env Dones so far: {:.2f}%\t Done Agents in ep: {:.2f}%\t In deadlock {:.2f}%(at switch {})\n\t\t Not started {}\t Eps: {:.2f}\tEP ended at step: {}/{}\tMean state_value: {}\t Epoch avg_loss: {}\n'.format(
            ep,
            self.env.get_num_agents(), 
            args.width, 
            args.height,
            self.score,
            np.mean(self.scores_window),
            100 * np.mean(self.done_window),
            100 * (num_agents_done/args.num_agents),
            100 * (num_agents_in_deadlock/args.num_agents),
            (num_agents_in_deadlock_at_switch),
            num_agents_not_started,
            eps,
            step+1,
            self.max_steps,
            (sum(self.path_values_buffer).detach().numpy()/len(self.path_values_buffer)),
            self.epoch_mean_loss)
        print(episode_stats, end=" ")

    def retrieve_wandb_log(self):
        wandb_log_dict = {"Learning rate value": self.rl_agent.optimizer_value.param_groups[0]['lr'], 
                    "Learning rate action": self.rl_agent.optimizer_action.param_groups[0]['lr']}
        if self.epoch_mean_loss is not None:
            wandb_log_dict.update({"mean_loss": self.epoch_mean_loss})

        wandb_log_dict.update({"action_probs": wandb.Histogram(np.array([prob.detach().numpy() for agent_probs in self.probs_buffer for prob in agent_probs]))})
        wandb_log_dict.update({"stop_go_action": wandb.Histogram(np.array([action for agent_actions in self.stop_go_buffer for action in agent_actions]))})
        wandb_log_dict.update({"node_values": wandb.Histogram(np.array(self.path_values_buffer))})
        wandb_log_dict.update({"episode_rewards": wandb.Histogram(self.rewards_buffer)})
        return wandb_log_dict


class GraphObservation(ObservationBuilder):
    '''
    When using Pytorch Geometric for Deep Learning on graphs, the observation becomes the nodes features and
    the adjacency matrix (see PyG documentation). Apparently no way to selectively classificy specific nodes (or yes?).
    Maybe using directed edge we only obtain the output of the desired node, which is the node representing the path.
    So we should pass in a subgraph for performance reasons.
    '''

    def __init__(self, depth):
        super(GraphObservation, self).__init__()
        # self.bfs_depth = bfs_depth
        self.depth = depth
        self.track_map = [] # matrix used to build the graph of nodes
        self.switches = []
        self.switch_paths_dict = {}
        self.path_switches_dict = {}
        self.agents_messages = {}
        # dict of bools to indicate whether an agent needs observation
        self.agent_requires_obs = {} # to optimize computing obs to speed up learning
        self.intersections_dict = defaultdict(list)
        self.track_to_intersection_dict = defaultdict(list)
        self.agents_path_at_switch = dict() # store the agents path at switch to compute target track section

    def _initGraphObservation(self):
        self.track_map = []
        self.switches = []
        self.switch_paths_dict = {}
        self.path_switches_dict = {}
        self.intersections_dict = defaultdict(list)
        self.track_to_intersection_dict = defaultdict(list)
        track_map, switches = self._build_track_map()
        self.track_map = track_map
        self.switches = switches
        self.agents_messages = {}
        self.agent_requires_obs = {}
        switch_paths_dict, path_switches_dict = self._build_graph_edges()
        self.switch_paths_dict = switch_paths_dict
        self.path_switches_dict = path_switches_dict
        self.agents_path_at_switch = dict()


    def get(self, handle: int) -> {}:
        obs = {}
        if len(self.agent_requires_obs) == 0 or self.agent_requires_obs[handle]:
            agent_pos = self.env.agents[handle].position
            if agent_pos is None: 
                if self.env.agents[handle].status == RailAgentStatus.READY_TO_DEPART:
                    agent_pos = self.env.agents[handle].initial_position
                else: # agent done
                    agent_pos = self.env.agents[handle].target 
            if self.track_map[agent_pos[0], agent_pos[1]] != -2: # not at switch
                unified, partitioned = self._get_graph_observation(
                    depth=self.depth, handle=handle)
                obs = {
                    "unified": unified,
                    "partitioned": partitioned
                }
        return obs

    def get_many(self, handles: Optional[List[int]] = None) -> {}:
        # self._initGraphObservation()
        obs = defaultdict(dict)
        for agent in handles:
            obs[agent] = self.get(agent)
        return obs

    def reset(self):
        self._initGraphObservation()
        return True

    def set_env(self, env: Environment):
        super().set_env(env)
        # self._initGraphObservation()

    def _build_graph_edges(self):
        '''
        Build the edges of the graph starting from the switch positions and the track map.
        For each switch, study the transition bitmap and add an edge for each possible transition
        linking 2 paths.

        OUTPUT:
            - SWITCH_PATHS_DICT: dictionary where the key is the switch position and the
                                 value is the set of all the linked tracks, identified by ID
            - PATH_SWITCHES_DICT: key is the path and value is the set of linked switches
                                  (each path is delimited by 2 switches, except deadends)
        '''
        switch_paths_dict = defaultdict(
            set)  # for each switch, store the adjacent paths
        path_switches_dict = defaultdict(set)
    

        for switch in self.switches:
            orientations = self._get_cell_orientations(switch)
            for orientation in orientations:
                # we need to skip -2 values because these are still part of switches
                reachable_paths, _ = self._get_reachable_paths(
                    switch, orientation)
                for path in reachable_paths:
                    path_id, _, _ = path
                    path_switches_dict[path_id].add(switch)
                    switch_paths_dict[switch].add(path_id)

        return switch_paths_dict, path_switches_dict

    def _build_track_map(self):
        """
        Assign to each cell of the railway map the ID corresponding to the track section
        Switches -> -2
        Intersections -> -1
        """

        bfs_queue = []  # store switches found at the end of a track section
        track_map = np.zeros((self.env.height, self.env.width))
        switches = []
        current_track_ID = 1  # progressive ID to assign to a track section
        # we start from a cell belonging to the railway, so just take position of agent 0
        handle = 0
        agent = self.env.agents[handle]

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            current_cell_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            current_cell_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            current_cell_position = agent.target
            done = True
        else:
            return None

        current_direction = agent.direction
        # before starting, explore current path in both directions
        track_map[current_cell_position[0]
                  ][current_cell_position[1]] = current_track_ID

        branch_directions = self._get_cell_orientations(current_cell_position)

        for current_direction in branch_directions:
            tmp_track_map, switch_position, last_is_dead_end, intersections = self._explore_path(current_cell_position,
                                                                                                 current_direction,
                                                                                                 current_track_ID,
                                                                                                 track_map)
            bfs_queue += intersections
            track_map = tmp_track_map
            if not last_is_dead_end:
                # add unexplored switch to the queue
                bfs_queue.append((switch_position, "switch"))
                switches.append(switch_position)

        # If first explored cell is a switch
        counts = np.count_nonzero(track_map == 1)
        if counts == 0:
            current_track_ID = 0

        while len(bfs_queue) > 0:
            switch_position, switch_type = bfs_queue.pop(0)
            branch_directions = self._get_cell_branch_directions(
                switch_position)
            intersection_branch_explored = False
            for branch_direction in branch_directions:
                new_path_cell = get_new_position(
                    switch_position, branch_direction)
                # track is not explored yet
                if track_map[new_path_cell[0]][new_path_cell[1]] == 0:
                    if switch_type == "intersection":
                        if not intersection_branch_explored:
                            current_track_ID += 1  # update track ID
                            # we find the already assigned orientations to compute the missing ones (4 in total, 2 for each path)
                            if len(self.intersections_dict[switch_position]) != 1:
                                print("Intersection_dict entry has {} elements, not 1!".format(len(self.intersections_dict[switch_position])))
                                raise Exception("INTERSECTION_DICT error")
                            already_found_orientations = self.intersections_dict[switch_position][0].orientation
                            missing_orientations = set([0, 1, 2, 3]) - set(already_found_orientations)
                            assert len(missing_orientations) == 2
                            self.intersections_dict[switch_position].append(IntersectionBranch(current_track_ID, list(missing_orientations)))
                            intersection_branch_explored = True
                    else:
                        current_track_ID += 1  # update track ID
                    tmp_track_map, adj_switch, last_is_dead_end, intersections = self._explore_path(new_path_cell,
                                                                                                    branch_direction, current_track_ID, track_map)
                    track_map = tmp_track_map
                    bfs_queue += intersections  # add intersections
                    if not last_is_dead_end and not (adj_switch in switches):
                        # add new found switch at the end of the path
                        bfs_queue.append((adj_switch, "switch"))
                        switches.append(adj_switch)
                elif track_map[new_path_cell[0]][new_path_cell[1]] == -1:
                    try: 
                        self.resolve_intersection_cell(new_path_cell, branch_direction)
                    except: # if the current direction is not explored yet
                        if switch_type == "intersection":
                            if not intersection_branch_explored:
                                current_track_ID += 1  # update track ID
                                # we find the already assigned orientations to compute the missing ones (4 in total, 2 for each path)
                                if len(self.intersections_dict[switch_position]) != 1:
                                    print("Intersection_dict entry has {} elements, not 1!".format(len(self.intersections_dict[switch_position])))
                                    raise Exception("INTERSECTION_DICT error")
                                already_found_orientations = self.intersections_dict[switch_position][0].orientation
                                missing_orientations = set([0, 1, 2, 3]) - set(already_found_orientations)
                                assert len(missing_orientations) == 2
                                self.intersections_dict[switch_position].append(IntersectionBranch(current_track_ID, list(missing_orientations)))
                                intersection_branch_explored = True
                        else:
                            current_track_ID += 1  # update track ID
                        tmp_track_map, adj_switch, last_is_dead_end, intersections = self._explore_path(new_path_cell,
                                                                                                        branch_direction, current_track_ID, track_map)
                        track_map = tmp_track_map
                        bfs_queue += intersections  # add intersections
                        if not last_is_dead_end and not (adj_switch in switches):
                            # add new found switch at the end of the path
                            bfs_queue.append((adj_switch, "switch"))
                            switches.append(adj_switch)


        self.track_map = track_map.astype(int)

        for key in self.intersections_dict.keys():
            if len(self.intersections_dict[key]) == 1:
                new_intersection_branch = self._complete_intersections_dict(key)
                if new_intersection_branch is not None:
                    self.intersections_dict[key].append(new_intersection_branch)
                else:
                    # both directions lead to a switch, so we have a single cell between two switches
                    # We assign a new track ID to this cell/path
                    current_track_ID += 1
                    missing_orientations = list(set([0, 1, 2, 3]) - set(self.intersections_dict[key][0].orientation))
                    self.intersections_dict[key].append(IntersectionBranch(current_track_ID, missing_orientations))
            assert len(self.intersections_dict[key]) != 0
            
            # Create a dict that maps each track section to the switches on their way
            for branch in self.intersections_dict[key]:
                self.track_to_intersection_dict[branch.track_id].append(key)
    
        
                

        return track_map.astype(int), switches
    
    def _complete_intersections_dict(self, key, prev_orient=None):
        '''
        Returns the missing IntersectionBranch of intersection
        Example: we have found a intersection cell (-1 in TRACK_MAP) but we only know one path crossing it.
        So we need to find the other path crossing the intersection.
        '''
        if prev_orient is not None:
            missing_orientations = [prev_orient]
        else:
            missing_orientations = list(set([0, 1, 2, 3]) - set(self.intersections_dict[key][0].orientation))
        for orientation in missing_orientations:
            valid_move_action = list(get_valid_move_actions_(orientation, key, self.env.rail).items())[0][0]
            if self.get_track(valid_move_action[1]) not in [-2, -1]:
                return IntersectionBranch(self.get_track(valid_move_action[1]), missing_orientations)
            elif self.get_track(valid_move_action[1]) == -1:
                try:
                    track_id = self.resolve_intersection_cell(valid_move_action[1], orientation)
                except:
                    logging.debug("Can't resolve for intersection {} with orientation: {}, missing_orientations: {}".format(key, orientation, missing_orientations))
                    res = self._complete_intersections_dict(valid_move_action[1], prev_orient=orientation)
                    if not res is None:
                        return res
            elif self.get_track(valid_move_action[1]) == 0:
                raise Exception("Branch from intersection hasn't been explored")
        

    def _explore_path(self, position, direction, track_ID, track_map):
        """
        Given agent handle, current position, and direction, explore that path until a new branching point is found.
        Add each cell of this path to the corresponding track.
        :param handle: agent id
        :param position: agent position as cell
        :param direction: agent direction
        :param track_ID: value to assign to the cells
        :return
        """

        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        last_is_switch = False
        last_is_dead_end = False
        last_is_terminal = False  # wrong cell or cycle
        visited = OrderedSet()
        intersections = []  # intersections met during exploration. Ignore them and add in the end
        while True:

            if (position[0], position[1], direction) in visited:
                last_is_terminal = True
                break
            visited.add((position[0], position[1], direction))

            '''
            # If the target node is encountered, pick that as node. Also, no further branching is possible.
            if np.array_equal(position, self.env.agents[handle].target):
                last_is_target = True
                break
            '''
            cell_transitions = self.env.rail.get_transitions(
                *position, direction)
            cell_transitions_bitmap = bin(
                self.env.rail.get_full_transitions(*position))
            total_transitions = cell_transitions_bitmap.count("1")

            if total_transitions <= 2:
                # Check if dead-end (1111111111111111), or if we can go forward along direction
                if total_transitions == 1:
                    track_map[position[0]][position[1]] = track_ID
                    last_is_dead_end = True
                    break

                if not last_is_dead_end:
                    # Keep walking through the tree along `direction`
                    # convert one-hot encoding to 0,1,2,3

                    track_map[position[0]][position[1]] = track_ID
                    direction = np.argmax(cell_transitions)
                    position = get_new_position(position, direction)

            elif total_transitions > 2:
                # Diamond crossing represented with -1
                # diamond crossing cell is in common to 2 track sections
                if int(cell_transitions_bitmap, 2) == int('1000010000100001', 2):
                    track_map[position[0]][position[1]] = -1
                    intersections.append((position, "intersection"))
                    self.intersections_dict[position].append(IntersectionBranch(track_ID, [direction, (direction+2)%4]))
                    direction = np.argmax(cell_transitions)
                    position = get_new_position(position, direction)
                else:  # switch found
                    track_map[position[0], position[1]] = -2
                    last_is_switch = True
                    break

            elif total_transitions == 0:
                # Wrong cell type, but let's cover it and treat it as a dead-end, just in case
                print("WRONG CELL TYPE detected in tree-search (0 transitions possible) at cell", position[0],
                      position[1], direction)
                last_is_terminal = True
                break
        # Out of while loop - a branching point was found

        return track_map, position, last_is_dead_end, intersections

    def _get_cell_orientations(self, position):
        '''
        Cell orientations are the allowed orientations (0,1,2,3) at a given cell.
        These orientations are the result of entering this cell from different directions.

        '''
        initial_position_transitions_bitmap_string = str(bin(
            self.env.rail.get_full_transitions(*position)))[2:]
        while (len(initial_position_transitions_bitmap_string) < 16):
            initial_position_transitions_bitmap_string = "0" + \
                initial_position_transitions_bitmap_string
        transition_list = [int(d)
                           for d in initial_position_transitions_bitmap_string]
        # transition bitmap as a matrix to determine the possible transitions
        transition_matrix = np.reshape(transition_list, (4, 4))
        # only take columns indexes
        cell_orientations = list(set(np.where(transition_matrix == 1)[0]))
        return cell_orientations

    def _get_cell_branch_directions(self, position):
        '''
        Branch direction are the directions at a given cell along which we reach another track cell
        '''
        initial_position_transitions_bitmap_string = str(bin(
            self.env.rail.get_full_transitions(*position)))[2:]
        while (len(initial_position_transitions_bitmap_string) < 16):
            initial_position_transitions_bitmap_string = "0" + \
                initial_position_transitions_bitmap_string
        transition_list = [int(d)
                           for d in initial_position_transitions_bitmap_string]
        # transition bitmap as a matrix to determine the possible transitions
        transition_matrix = np.reshape(transition_list, (4, 4))
        # only take columns indexes
        cell_orientations = list(set(np.where(transition_matrix == 1)[1]))
        return cell_orientations

    def get_track(self, position):
        """
        INPUT
            position: cell position
            track_map: map where the cells represent the track they belong to
        OUTPUT
            track_ID: ID of the track containing the cell specified by its position (tuple of coordinates)
        """
        return self.track_map[position[0]][position[1]]

    

    def _find_next_switch_from_position(self, position, orientation):
        '''
        Given a position and a orientation of the agent, find the next reachable switch and return the position,
        orientation of the agent at that switch and whether it's deadend
        '''
        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        last_is_switch = False
        last_is_dead_end = False
        last_is_terminal = False  # wrong cell or cycle
        last_is_target = False  # target was reached
        visited = OrderedSet()

        while True:

            if (position[0], position[1], orientation) in visited:
                last_is_terminal = True
                break
            visited.add((position[0], position[1], orientation))

            '''
            # If the target node is encountered, pick that as node. Also, no further branching is possible.
            if np.array_equal(position, self.env.agents[handle].target):
                last_is_target = True
                break
            '''

            cell_transitions_bitmap = bin(
                self.env.rail.get_full_transitions(*position))
            cell_transitions = self.env.rail.get_transitions(
                *position, orientation)
            num_transitions = np.count_nonzero(cell_transitions)
            total_transitions = cell_transitions_bitmap.count("1")

            if total_transitions <= 2:
                # Check if dead-end (1111111111111111), or if we can go forward along direction
                if total_transitions == 1:
                    last_is_dead_end = True
                    break

                if not last_is_dead_end:
                    # Keep walking through the tree along `direction`
                    # convert one-hot encoding to 0,1,2,3
                    orientation = np.argmax(cell_transitions)
                    position = get_new_position(position, orientation)

            elif total_transitions > 2:
                if int(cell_transitions_bitmap, 2) == int('1000010000100001', 2):
                    direction = np.argmax(cell_transitions)
                    position = get_new_position(position, direction)
                else:
                    assert self.track_map[position[0], position[1]] == -2
                    last_is_switch = True
                    break

        return position, orientation, last_is_dead_end


    def _get_reachable_paths(self, switch_position: Tuple[int, int], current_orientation: int, explored_cells=None):
        '''
        Given a switch and the orientation, return the next reachable tracks with their ID
        Paths are represented by their track ID and progressive index, because the same track section could be reached
        through more ways.
        '''
        if explored_cells is None:
            explored_cells = []
        reachable_paths = []
        path_dict = defaultdict(list) # dict with switch cells that lead to the track
        orientation = current_orientation
        valid_move_actions = get_valid_move_actions_(orientation, switch_position, self.env.rail)
        for a in valid_move_actions:
            new_direction = a[2]
            new_track_id = self.get_track(a[1])
            if new_track_id == -2:
                if a[1] not in explored_cells:
                    new_explored_cells = explored_cells.copy()
                    new_explored_cells.append(switch_position)
                    reachable_paths_tmp, paths_dict_tmp = self._get_reachable_paths(
                        a[1], new_direction, new_explored_cells)
                    reachable_paths += reachable_paths_tmp
                    for key in paths_dict_tmp.keys():
                        for branch_waypoints in paths_dict_tmp[key]:
                            path_dict[key].append([Waypoint(switch_position, current_orientation)]+branch_waypoints)
            elif new_track_id==0:
                raise Exception("Error: empty cell can't be reached")
            else:
                if new_track_id == -1: # Intersection, have to handle which path to consider
                    new_track_id = self.resolve_intersection_cell(a[1], new_direction)
                reachable_paths.append(
                    (new_track_id, new_direction, a[1]))
                
                path_dict[new_track_id].append([Waypoint(switch_position, current_orientation), Waypoint(a[1], new_direction)])
        return reachable_paths, path_dict

    #def _get_path_dict_progessive_index(self, path, )
    


    def resolve_intersection_cell(self, position, orientation):
        intesection_branches = self.intersections_dict[position]
        for branch in intesection_branches:
            if orientation in branch.orientation:
                return branch.track_id
        raise Exception("Cannot resolve for intersection cell!")

    def _get_graph_observation(self, depth: int, handle: int, consider_joining_paths=False, include_root: bool = True, symmetric_edges: bool = False):
        '''
        We build the representation feature for each node/track in the computation graph.
        We represent track sections (rail sections delimited by switches) as nodes of the graph.
        Edges are then the possible transitions from one track section in a particular direction to another track section.
        Each track section is associated with a ID, used in OBS_GRAPH to build the graph.
        The cells belonging to a certain track are noted with the same ID.

        We use a value-based approach for reinforcement learning, where each action is mapped to the selection of
        certain path represented by track ID, which is directly reachable from the current one

        When using pytorch_geometric there is no chance to selectively compute node features
        only for single nodes (our agents), so in order to avoid expensive and useless
        computations we do graph convolutions on a subgraph. We need to compute the subgraph, given a certain depth, for 
        each agent when required

        Parameters
        ----------
        depth :  depth of the graph convolution
        handle : ID of the considered agent
        track_map : matrix of integers indicating to which track ID a cell belongs to
        switch_paths_dict : for each switch a list of joining tracks
        path_switch_dict : for each path a list of delimiting switches
        consider_joining_paths : if True, also take into account the paths not reachable from a switch,
        given a certain direction
        include_root: if True, also include the current track in the computation graph
        symmetric_edges: if True, add symmetric edges, i.e. if (a,b) in Edges -> (b,a) also in Edges

        Returns
        -------
        node_features : Node feature matrix with shape [num_nodes, num_node_features]
        graph_edges: Graph connectivity in COO format with shape [2, num_edges] and type torch.long

        '''
        node_features = defaultdict(list)
        # Add lists of 2 elements (node index) to represent edges
        graph_edges = []
        # Afterwards, call "graph_edges.t().contiguous()" for standard format
        # edges for single states (each path from root node is a possible state)
        fist_layer_paths_dict = dict()
        partitioned_graph_edges = defaultdict(list)
        partitioned_computation_graph = defaultdict(lambda: defaultdict(set))
        partitioned_node_features = defaultdict(lambda: defaultdict(list))
        reachable_nodes = []
        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_position = agent.position
        elif agent.status == RailAgentStatus.DONE_REMOVED:
            agent_position = agent.target

        current_direction = agent.direction
        current_track = self.get_track(agent_position)
        if current_track == -1:  # agent is at an intersection
            current_track = self.resolve_intersection_cell(agent_position, current_direction)
        elif current_track == -2 and self.is_agent_in_deadlock(handle):  # agent is at switch
            # TODO: find a way to implement this observation
            # print("Agent {} start position at switch. Need to handle this case".format(handle))
            return {}, {}
            
        # store the nodes at each level of computation graph
        computation_graph = defaultdict(set)
        tracks_queue = []  # list of tuples (TRACK_ID, SWITCH_ORIGIN)
        next_switch, switch_orientation, last_is_deadend = self._find_next_switch_from_position(
            agent_position, current_direction)
        while last_is_deadend:
            rail_env_next_action = list(get_valid_move_actions_(current_direction,
                                                                agent_position,
                                                                self.env.rail)
                                        .items())[0][0]
            agent_position = rail_env_next_action.next_position
            current_direction = rail_env_next_action.next_direction
            next_switch, switch_orientation, last_is_deadend = self._find_next_switch_from_position(
                agent_position, current_direction)
            if last_is_deadend:
                logging.debug("Last is again deadend")

        node = (current_track, next_switch, switch_orientation, current_track)
        tracks_queue.append(node)

        root_node = (current_track, current_direction, agent_position)
        start_index = 0
        if include_root:
            computation_graph[start_index].add(root_node)

        for layer in range(start_index, depth):
            layer_queue = []  # accumulate all nodes for this layer

            # node is a position tuple (x,y) belonging to a track section
            for node in tracks_queue:
                # Current_direction = orientation
                current_track, next_switch, current_direction, root_track = node
                if layer == start_index+1:
                    # retrieve indexes for PARTITIONED_GRAPH_EDGES (first paths IDs from root node)
                    root_track = (current_track, 0)

                if consider_joining_paths:
                    reachable_nodes = list(
                        self.switch_paths_dict[next_switch].copy().remove(current_track))
                else:
                    reachable_nodes, paths_dict = self._get_reachable_paths(
                        next_switch, current_direction)
                    if layer==0:
                        fist_layer_paths_dict = paths_dict

                if include_root:
                    l = layer+1
                    if l < depth:
                        computation_graph[l] = computation_graph[l] | set(
                            reachable_nodes)

                if layer > start_index:
                    partitioned_computation_graph[root_track][layer] = partitioned_computation_graph[root_track][layer] | set(
                        reachable_nodes)

                for track in reachable_nodes:
                    # compute nodes for next layer
                    new_track_id, new_direction, track_cell = track
                    if layer+1 < depth:
                        graph_edges.append([current_track, new_track_id])
                        if symmetric_edges:
                            graph_edges.append([new_track_id, current_track])
                    if layer == start_index:  # initialize partition_computation_graph
                        root_track = (new_track_id, 0)
                        partitioned_computation_graph[root_track][layer].add(
                            track)
                    if layer > start_index:  # skip for the current track where the agent is.
                        partitioned_graph_edges[root_track].append(
                            [current_track, new_track_id])
                        if symmetric_edges:
                            partitioned_graph_edges[root_track].append(
                                [new_track_id, current_track])

                    if layer+1 < depth:  # we stop accumulating in queue if it's the last layer of depth
                        new_next_switch, orientation, _ = self._find_next_switch_from_position(
                            track_cell, new_direction)
                        new_direction = orientation

                        if new_next_switch is not None:
                            # new direction becomes orientation
                            layer_queue.append(
                                (new_track_id, new_next_switch, new_direction, root_track))

            tracks_queue = list(set(layer_queue))
        
        partitioned_node_features_tmp = defaultdict(lambda: defaultdict(list))
        computed_features = dict()
        # Check computation graph and compute node features
        for layer in range(depth):
            for track in computation_graph[layer]:
                track_id, current_orientation, origin_switch = track
                switch_cells = None
                if layer == 1: # Consider the switch cells for the first path
                    for i, switch_branch in enumerate(fist_layer_paths_dict[track_id]):
                        switch_cells = list(map(lambda x: x.position, switch_branch))
                        node_features[(track_id, i)] = self._compute_node_features(
                            handle, track_id, current_orientation, origin_switch, switch_cells)
                        computed_features.update({(track_id, i): node_features[(track_id, i)]})
                else:
                    node_features[(track_id, 0)] = self._compute_node_features(
                        handle, track_id, current_orientation, origin_switch)
                    computed_features.update({(track_id, 0): node_features[(track_id, 0)]})
                
            for path in partitioned_computation_graph.keys():
                for track in partitioned_computation_graph[path][layer]:
                    track_id, current_orientation, origin_switch = track
                    track_features = []
                    if layer > 0:
                        track_features = self._compute_node_features(
                            handle, track_id, current_orientation, origin_switch)
                        partitioned_node_features_tmp[path][(track_id, 0)] = track_features
                        computed_features.update({(track_id, 0): partitioned_node_features_tmp[path][(track_id, 0)]})
        # Layer 0
        for path in partitioned_node_features_tmp.keys():
            for track in partitioned_computation_graph[path][0]:
                track_id, current_orientation, origin_switch = track
                assert track_id == path[0]
                track_features = []
                switch_cells = None
                for i, switch_branch in enumerate(fist_layer_paths_dict[track_id]):
                    switch_cells = list(map(lambda x: x.position, switch_branch))
                    track_features = self._compute_node_features(
                        handle, track_id, current_orientation, origin_switch, switch_cells)
                    partitioned_node_features[(track_id, i)][(track_id, i)] = track_features
                    computed_features.update({(track_id, i): partitioned_node_features[(track_id, i)][(track_id, i)]})
                    if i > 0:
                        partitioned_graph_edges[(track_id, i)] = partitioned_graph_edges[(track_id, 0)]
                    for k,v in partitioned_node_features_tmp[path].items():
                        partitioned_node_features[(track_id, i)][k] = copy(v)

        # Normalize data in computed features and then reassign the normalized features to the nodes
        computed_features_tensor = []
        for k,v in computed_features.items():
            computed_features_tensor.append(torch.FloatTensor(v))
           
        # computed_features_tensor = torch.FloatTensor(computed_features_tensor)
        # loader = D.DataLoader(computed_features_tensor, batch_size=len(computed_features_tensor), num_workers=1)
        # data = next(iter(loader))
        # minmax_scaler = preprocessing.MinMaxScaler()
        # minmax_scaler.fit(data)
        '''
        means = []
        stds = []
        for c in range(data.shape[1]):
            means.append(data[:,c].mean())
            stds.append(data[:,c].std())
        data_normalized = F.normalize(data, dim=0)
        '''
        # scaled_features = minmax_scaler.transform(data)


        # Now remap the track indexes from old IDs to new ones
        # (ID are the row of the node features in the graph feature matrix)
        old_ids = list(node_features.keys())
        new_ids = list(range(len(old_ids)))
        old_to_new_map = dict(zip(old_ids, new_ids))
        new_to_old_map = dict(zip(new_ids, old_ids))
        # Map the new node indexes to the edges
        new_mapped_graph_edges = [] # Considering multiple ways to reach same track
        for e in graph_edges:
            e0_list = list(filter(lambda x: x[0]==e[0], old_ids))
            e1_list = list(filter(lambda x: x[0]==e[1], old_ids))
            new_mapped_graph_edges += list(itertools.product(e0_list, e1_list))

        new_graph_edges = [[old_to_new_map[e[0]],
                            old_to_new_map[e[1]]] for e in new_mapped_graph_edges]
        assert len(new_graph_edges) > 0
        new_node_features = []
        for new_node_index in range(len(old_ids)):
            new_node_features.append(node_features[new_to_old_map[new_node_index]])
            

        # new_node_features = torch.FloatTensor(minmax_scaler.transform(torch.FloatTensor(new_node_features)))
        new_node_features = torch.FloatTensor(torch.FloatTensor(new_node_features) * 0.1)
        new_graph_edges = torch.LongTensor(new_graph_edges).t().contiguous()
        observation = {
            "node_features": new_node_features,
            "graph_edges": new_graph_edges,
            "new_to_old_map": new_to_old_map
        }
        # do the same for partitioned graph
        partitioned_observation = defaultdict(dict)
        partitioned_observation[(0, 0)] = observation
        for path in partitioned_node_features.keys():  # for each subgraph
            old_ids = list(partitioned_node_features[path].keys())
            # first element of node features must be the track section
            assert path == old_ids[0]
            new_ids = list(range(len(old_ids)))
            old_to_new_map = dict(zip(old_ids, new_ids))
            new_to_old_map = dict(zip(new_ids, old_ids))
            # Map the new node indexes to the edges
            new_mapped_graph_edges = [] # Considering multiple ways to reach same track
            for e in partitioned_graph_edges[path]:
                e0_list = list(filter(lambda x: x[0]==e[0], old_ids))
                e1_list = list(filter(lambda x: x[0]==e[1], old_ids))
                new_mapped_graph_edges += list(itertools.product(e0_list, e1_list))

            new_graph_edges = [[old_to_new_map[e[0]], old_to_new_map[e[1]]]
                               for e in new_mapped_graph_edges]
            assert len(new_graph_edges) > 0
            new_node_features = []
            for new_node_index in range(len(old_ids)):
                new_node_features.append(
                    partitioned_node_features[path][new_to_old_map[new_node_index]])

            #new_node_features = torch.FloatTensor(minmax_scaler.transform(torch.FloatTensor(new_node_features)))
            new_node_features = torch.FloatTensor(torch.FloatTensor(new_node_features) * 0.1)
            new_graph_edges = torch.LongTensor(
                new_graph_edges).t().contiguous()
            partitioned_observation[path] = {
                "node_features": new_node_features,
                "graph_edges": new_graph_edges,
                "new_to_old_map": new_to_old_map
            }

        return observation, partitioned_observation

    def _are_agents_same_direction(self, position1, orientation1, position2, orientation2):
        next_switch_from_position1 = self._find_next_switch_from_position(
            position1, orientation1)[0]
        next_switch_from_position2 = self._find_next_switch_from_position(
            position2, orientation2)[0]
        if next_switch_from_position1 == next_switch_from_position2:
            return True
        return False

    def _compute_node_features(self, handle: int, track_ID: int, orientation: int, path_cell: Tuple[int, int], switch_cells=None):
        '''
        Given a track section and an agent, compute the node representation for that track w.r.t. to the
        agent's information
        PROBLEM: What if there is no intersection? what should the default value be? 0 could mean that we
        are at the intersection point because the distance is null. So should we even consider the distance?
        Or just take 1 or 0 values?
        POSSIBLE SOLUTIONS:
        • Fill with -1 where the "distance" is not available (for example if the feature is "distance to intersection" but
        there is no intersection in this track, we put a -1 instead of 0)

        origin_switch = if we are computing features for a track section where our agent is not present, we
        need to compute a valid direction in order to compute the shortest path. So we use the information
        about the switch we come from to compute this direction. This information is given by how we explore the graph
        '''
        track_map = self.track_map.copy()

        agents = self.env.agents
        agent = agents[handle]
        agent_target = agent.target
        # DIRECTION argument can be different if agent is not on this track
        agent_orientation = agent.direction
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_position = agent.position
        elif agent.status in [RailAgentStatus.DONE_REMOVED, RailAgentStatus.DONE]:
            agent_position = agent.target


        track_length = np.count_nonzero(track_map == track_ID) # doesn't count intersections
        target_distance = 0  # distance to the agent's target from this track section
        num_agents_same_dir = 0  # agents on this track in the same direction
        num_agents_opp_dir = 0
        avg_speed_agents_same_dir = 0
        # another_agent = -1 # if other agents are on this track, the mean of distances is considered
        target_on_track = 0  # if agent's target is on this track, compute distance
        another_target = 0  # if another agent's target is on this track
        num_intersections = 0  # if there is an intersection on this track take the distance
        # if agent has malfunctions, take remaining number of blocking steps
        malfunction_agent = 0
        malfunction_other = 0  # maximum blocking steps of all agents on this section
        slowest_speed = 0  # slowest speed of all agents in the same direction on this track
        agent_not_started = 0  # num agents not started yet
        agent_speed = 0  # speed of current agent
        # remaining cells to complete this track section, before reaching switch.
        # If the agent is not on this track, consider the path to reach this track
        
        node_degree = 0  
        dead_end = 0  # 1 if track section has a dead end

        distance_map = self.env.distance_map.get()

        num_switches = len(self.path_switches_dict[track_ID])
        if num_switches == 1:
            dead_end = 1

        # list of cells belonging to this track
        track_cells = list(zip(*np.where(track_map == track_ID)))
        # add intersections present in this track section
        intersections = self.track_to_intersection_dict[track_ID]
        num_intersections = len(intersections)
        track_cells += intersections

        agent_on_this_track = True if (agent_position in track_cells) else False

        next_switch_from_origin_switch = self._find_next_switch_from_position(
                path_cell, orientation)[0]
        # node degree
        node_degree = len(
            self.switch_paths_dict[next_switch_from_origin_switch])
        # TARGET_DISTANCE
        if agent_on_this_track:
            target_distance = distance_map[(handle, *agent_position, agent_orientation)]
            if target_distance == float('inf'):
                target_distance = np.count_nonzero(track_map != 0) # penalize if shortest path can't be computed
            #print("target distance: {}".format(target_distance))
            '''
            shortest_path = get_k_shortest_paths(
                self.env, agent_position, agent_orientation, agent_target)
            if len(shortest_path) > 0:
                target_distance = len(shortest_path[0])
            else:
                target_distance = np.count_nonzero(track_map != 0)
            '''
            malfunction_agent = agent.malfunction_data["malfunction"]
            agent_speed = agent.speed_data["speed"] if agent.moving else 0
           
        else:
            # count from the middle cell of the track
            # compute the cell in the middle
            track_cells_ordered = track_cells
            track_cells_ordered.sort(key=lambda x: x[0])
            mid_cell = track_cells_ordered[len(track_cells)//2]
            mid_cell_orientations = self._get_cell_orientations(mid_cell)
            # which orientation to take? coherent with the exploration direction, the that leads to the same switch
            for mid_cell_orientation in mid_cell_orientations:
                next_switch_from_mid_cell = self._find_next_switch_from_position(
                    mid_cell, mid_cell_orientation)[0]
                if next_switch_from_mid_cell == next_switch_from_origin_switch:
                    orientation = mid_cell_orientation
                    break
            
            target_distance = distance_map[(handle, *mid_cell, orientation)]
            if target_distance == float('inf'):
                target_distance = np.count_nonzero(track_map != 0) # penalize if shortest path can't be computed
            #print("target distance: {}".format(target_distance))

            '''
            shortest_path = get_k_shortest_paths(
                self.env, mid_cell, orientation, agent_target)  # agent_direction could not work
            if len(shortest_path) > 0:
                shortest_path = shortest_path[0]
                target_distance = len(shortest_path)
            else:
                logging.debug("Error: shortest path from mid cell to target does not exist")
                logging.debug("Computing node features for track {}".format(track_ID))
                logging.debug("Shortest path is: {}".format(shortest_path))
                logging.debug("mid_cell: {}, track_id: {}".format(
                    mid_cell, self.get_track(mid_cell)))
                logging.debug("agent_target: {}, track_id: {}".format(
                    agent_target, self.get_track(agent_target)))
                target_distance = np.count_nonzero(track_map != 0) # penalize if shortest path can't be computed
            '''
            
        agents_blocking_steps = []
        agents_speeds = []
        for a in agents:
            if agent.status == RailAgentStatus.READY_TO_DEPART:
                if a.initial_position in track_cells:
                    agent_not_started += 1
            elif agent.status == RailAgentStatus.ACTIVE:
                if a.position in track_cells:
                    agents_blocking_steps.append(
                        a.malfunction_data["malfunction"])
                    if (agent_on_this_track):
                        tmp_position = agent_position
                        tmp_orientation = agent_orientation
                    else:
                        tmp_position = mid_cell
                        tmp_orientation = mid_cell_orientation
                    if self._are_agents_same_direction(a.position, a.direction, tmp_position, tmp_orientation):
                        # NUM_AGENTS_SAME_DIR
                        num_agents_same_dir += 1
                        agents_speeds.append(a.speed_data["speed"])
                    else:
                        # NUM_AGENTS_OPP_DIR
                        num_agents_opp_dir += 1
                if a.position and (a.target in track_cells) and (not a == agent):
                    another_target += 1
        # TARGET_ON_TRACK
        if agent_target in track_cells:
            target_on_track = 1

        # SLOWEST_SPEED
        slowest_speed = min(agents_speeds) if len(
            agents_speeds) > 0 else slowest_speed
        # MALFUNCTION_OTHER
        malfunction_other = max(agents_blocking_steps) if len(
            agents_blocking_steps) > 0 else malfunction_other

        # If there are other agents at switch consider them before entering the switch
        if switch_cells is not None:
            track_length += len(switch_cells)
            for cell_position in switch_cells:
                for agent in self.env.agents:
                    if agent.status == RailAgentStatus.ACTIVE:
                        if agent.position == cell_position:
                            if agent.handle in self.agents_path_at_switch.keys():
                                if self.agents_path_at_switch[agent.handle][0] == self.get_track(agent_position):
                                    num_agents_opp_dir += 1
                                elif self.agents_path_at_switch[agent.handle][0] == track_ID:
                                    num_agents_same_dir += 1
                            else:
                                num_agents_opp_dir += 1
                                
        if self.get_track(agent_position) == track_ID and agent_orientation == orientation:
            num_agents_same_dir = 0
            num_agents_opp_dir = 0
            malfunction_other = 0
            slowest_speed = 0


        return (
            track_length,
            target_distance,
            num_agents_same_dir,
            num_agents_opp_dir,
            target_on_track,
            another_target,
            malfunction_other,
            slowest_speed,
            agent_not_started,
            node_degree,
            agent_speed,
            num_intersections
        )

    def render_environment(self):
        env_renderer = RenderTool(self.env)
        env_renderer.render_env(show=True, frames=True,
                                show_observations=False)

    def plot_track_map(self):
        # Display matrix
        ax = sns.heatmap(self.track_map, annot=True, fmt="d")
        figure = ax.get_figure()
        figure.savefig('svm_conf.png', dpi=200)

    def choose_railenv_actions(self, handle, track):
        '''
        Parameters
        ----------
        target_track: tuple (track_id, path_index) where the first value is the track section ID and the 
                  second indicating which path to reach through the switch

        Compute the action/s required to reach the next track represented by track_ID
        '''
        target_track, action, _, _, _ = track
        track_ID, index = target_track
        if action == 1 or action == 0: 
            agents = self.env.agents
            agent = agents[handle]
            agent_orientation = agent.direction
            if agent.status == RailAgentStatus.READY_TO_DEPART:
                agent_position = agent.initial_position
            elif agent.status == RailAgentStatus.ACTIVE:
                agent_position = agent.position

        
            valid_move_action = get_valid_move_actions_(agent_orientation, agent_position, self.env.rail)
            switch_position, switch_orientation = get_new_position_for_action(agent_position, agent_orientation, valid_move_action.popitem()[0].action, self.env.rail)
            _, waypoint_path_dict = self._get_reachable_paths(switch_position, switch_orientation)
            self.agents_path_at_switch.update({handle: (track_ID, waypoint_path_dict[track_ID][index])})
            waypoint_path = [Waypoint(agent_position, agent_orientation)] + waypoint_path_dict[track_ID][index]

            return self.convert_waypoints_to_railenvactions(waypoint_path)
        
        elif action == 0:
            return [RailEnvActions.STOP_MOVING]
        
        else: 
            raise Exception("Error: Track_ID can't be negative")

    

    def find_closest_track_cell(self, agent_position, agent_orientation, target_track_id):
        # returns the closest cell to the agent position belonging to the target track section.
        # Used to compute the shortest path to the track section, during the computation of actions to reach a 
        # certain track section on a switch region
        track_cells = list(zip(*np.where(self.track_map == target_track_id)))
        if len(track_cells)>0:
            track_cell = track_cells[0]
        else: # intersection cell
            for key in self.intersections_dict.keys():
                intersection = self.intersections_dict[key] # list of IntersectionBranch
                if intersection[0].track_id == target_track_id or intersection[1].track_id == target_track_id:
                    track_cell = key # position of cell belonging to target_track_id

        waypoint_path = list(get_k_shortest_paths(
            self.env, agent_position, agent_orientation, track_cell)[0])
        for waypoint in waypoint_path:  # ugly way to get verify shortest path :(
            if self.track_map[waypoint.position[0], waypoint.position[1]] == target_track_id:
                return waypoint.position  # closest track cell to agent's position
            elif self.track_map[waypoint.position[0], waypoint.position[1]] == -1:
                if self.resolve_intersection_cell(waypoint.position, waypoint.direction) == target_track_id:
                    return waypoint.position
                else:
                    if waypoint != waypoint_path[0]:
                        raise Exception("Error: unexpected track section crossed.")
            elif self.track_map[waypoint.position[0], waypoint.position[1]] != -2:
                # first waypoint belongs to origin track section
                if waypoint != waypoint_path[0]:
                    raise Exception("Error: unexpected track section crossed.")
            
        raise Exception("Wrong shortest path to track section")
    

    def convert_waypoints_to_railenvactions(self, waypoints_list):
        railenv_actions = []
        for i in range(len(waypoints_list)-1):
            waypoint = waypoints_list[i]
            waypoint_next = waypoints_list[i+1]
            if i+1 < len(waypoints_list)-1:
                # if fails, it means that some cells do not belong to switch sections
                if not self.track_map[waypoint_next.position[0], waypoint_next.position[1]] in [-2]:
                    logging.debug("Waypoint path: {}".format(waypoints_list))
                    logging.debug("Waypoint is not in switch. Waypoint: {}, track_id: {}".format(waypoint_next.position, self.track_map[waypoint_next.position[0], waypoint_next.position[1]]))
                    raise Exception("Error: waypoint is not in switch section")
            action = get_action_for_move(
                        waypoint.position,
                        waypoint.direction,
                        waypoint_next.position,
                        waypoint_next.direction,
                        self.env.rail)
            railenv_actions.append(action)
            new_position_from_action = get_new_position_for_action(waypoint.position,
                                                                    waypoint.direction,
                                                                    action,
                                                                    self.env.rail)[0]
            assert new_position_from_action == waypoint_next.position
            if i+1 == len(waypoints_list)-1:
                assert self.track_map[new_position_from_action[0],
                    new_position_from_action[1]] != -2
        # always at least 2 actions to take (e.g. when we have one cell switch one action to enter cell and one to exit)
        assert len(railenv_actions) >= 2
        assert len(railenv_actions) == len(waypoints_list)-1
        return railenv_actions

    def is_agent_entering_switch(self, handle):
        agents = self.env.agents
        agent = agents[handle]
        agent_orientation = agent.direction
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_position = agent.position

        valid_move_action = get_valid_move_actions_(
            agent_orientation, agent_position, self.env.rail)
        assert len(valid_move_action) == 1
        assert self.get_track(agent_position) != -2  # can't be at switch
        new_position, _ = get_new_position_for_action(
            agent_position, agent_orientation, valid_move_action.popitem()[0].action, self.env.rail)
        # if next position is switch, then return True
        return self.track_map[new_position[0], new_position[1]] == -2
        '''
        if self.track_map[new_position[0], new_position[1]] == -2:
            if agent.speed_data["speed"] == 1:
                return True
            else:
                # agent is doing last step in the cell before reaching new cell
                if np.isclose(agent.speed_data["position_fraction"]+agent.speed_data["speed"]%1, 0.0, atol=1.e-3):
                    return True
        return False
        '''
              

    def is_agent_2_steps_from_switch(self, handle):
        agents = self.env.agents
        agent = agents[handle]
        agent_orientation = agent.direction
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_position = agent.position

        # next cell
        valid_move_action = get_valid_move_actions_(
            agent_orientation, agent_position, self.env.rail)
        assert len(valid_move_action) == 1
        new_position, new_orientation = get_new_position_for_action(
            agent_position, agent_orientation, valid_move_action.popitem()[0].action, self.env.rail)
        # if next position is switch, then return True
        if self.track_map[new_position[0], new_position[1]] == -2:
            return False
        # next next cell
        valid_move_action = get_valid_move_actions_(
            new_orientation, new_position, self.env.rail)
        assert len(valid_move_action) == 1
        new_position, _ = get_new_position_for_action(
            new_position, new_orientation, valid_move_action.popitem()[0].action, self.env.rail)
        return self.track_map[new_position[0], new_position[1]] == -2

    def is_agent_about_to_finish(self, handle):
        agents = self.env.agents
        agent = agents[handle]
        agent_orientation = agent.direction
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_position = agent.position

        valid_move_action = get_valid_move_actions_(
            agent_orientation, agent_position, self.env.rail)
        assert len(valid_move_action) == 1
        new_position, _ = get_new_position_for_action(
            agent_position, agent_orientation, valid_move_action.popitem()[0].action, self.env.rail)
        # if next position is switch, then return True
        return new_position == agent.target

    def is_agent_exiting_switch(self, handle, next_action):
        # Next cell is not a switch cell anymore
        agents = self.env.agents
        agent = agents[handle]
        agent_orientation = agent.direction
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_position = agent.position

        # be sure we are still at switch
        assert self.get_track(agent_position) == -2
        new_position, _ = get_new_position_for_action(
            agent_position, agent_orientation, next_action, self.env.rail)
        # if next position is switch, then return True
        return self.get_track(new_position) != -2

    def agent_could_move(self, handle, action, old_speed_data):
        agents = self.env.agents
        agent = agents[handle]
        agent_orientation = agent.direction
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_position = agent.position
        elif agent.status == RailAgentStatus.DONE_REMOVED:
            return True

        if agent_position != agent.old_position:
            return True
        elif agent.malfunction_data["malfunction"]==0: 
            if agent.speed_data["speed"] < 1:
                if agent.speed_data["position_fraction"] < 1 and (agent.speed_data["position_fraction"] != old_speed_data["position_fraction"]) and action != RailEnvActions.STOP_MOVING:
                # if speed is fractionary then check if agent could move 
                    return True
        return False

    def is_agent_in_deadlock(self, handle, deadlock_list=None):
        # We check that this agent can only move in one direction but it's blocked by another agent, who
        # itself can't move forward because it's blocked
        #print("IS_AGENT_IN_DEADLOCK: agent {}".format(handle))
        if deadlock_list is None:
            deadlock_list = []
            #print("Deadlock_list was None, now is {}".format(deadlock_list))
        #else:
            #print("Deadlock_list is {}".format(deadlock_list))
        agents = self.env.agents
        agent = agents[handle]
        agent_orientation = agent.direction
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            return False
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_position = agent.position
        elif agent.status == RailAgentStatus.DONE_REMOVED:
            return False
        elif agent.status == RailAgentStatus.DONE:
            return False
        
        

        # logging.debug("DEADLOCK: Agent {}, position {}, old_position {}".format(handle, agent_position, agent.old_position))
        
        valid_move_actions = get_valid_move_actions_(agent_orientation, agent_position, self.env.rail)
        #logging.debug("Len of valid_move_actions: {}".format(len(valid_move_actions)))
        if len(valid_move_actions) >= 1:  # agent only has 1 possible move
            for next_move_action in valid_move_actions:
                next_position = next_move_action[1]

                # Check if the next cell is free
                other_agent = None
                for a in self.env.agents:
                    if a != agent and a.position == next_position:
                        other_agent = a
                
                if other_agent is None:
                    return False
                    #logging.debug("Other agent is {}".format(other_agent.handle))
                    #assert not other_agent is None
                else:
                    if not other_agent.handle in deadlock_list: # Circular deadlock
                        if other_agent.malfunction_data["malfunction"] == 0:
                            new_deadlock_list = deadlock_list.copy()
                            new_deadlock_list.append(handle)
                            #print("Other agent is: {}, new deadlock list is: {}".format(other_agent.handle, new_deadlock_list))
                            is_other_agent_deadlock = self.is_agent_in_deadlock(other_agent.handle, new_deadlock_list)
                            # We have to check all the possible moves before declaring deadlock
                            # But if one agent is not in deadlock, than all the other behind are not as well
                            if not is_other_agent_deadlock: 
                                return False
                        else:
                            return False
        return True

            



    def preprocess_agent_obs(self, obs, handle):
        '''
        Processes agent's observation into a batch of graphs, where each graph is a possible path
        from the next switch
        '''
        GraphInfo = namedtuple("GraphInfo", "agent_handle, path")
        batch_list = []
        counter = 0

        obs_part = obs["partitioned"]
        for path in obs_part.keys():
            data = Data(x=obs_part[path]["node_features"],
                        edge_index=obs_part[path]["graph_edges"])
            data.index = counter
            data.new_to_old_map = obs_part[path]["new_to_old_map"]
            data.graph_info = GraphInfo(handle, path)
            batch_list.append(data)
            counter += 1

        return Batch.from_data_list(batch_list)
