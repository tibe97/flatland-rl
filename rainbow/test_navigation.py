from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from dueling_double_dqn import Agent
from graph_for_observation import GraphObservation, EpisodeController
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.agent_utils import RailAgentStatus
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from collections import deque, defaultdict
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_
from datetime import date, datetime
import numpy as np
import torch
import sys
import os
import argparse
import pprint
import math
import random
# make sure the root path is in system path
from pathlib import Path

import logging


# These 2 lines must go before the import from src/
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Test DQN
def main(args):

    random.seed(1)
    np.random.seed(1)


    agent_weights_path = "/home/runnphoenix/work/flatland-rl/rainbow/test_results/checkpoint_1_agents_on_25_25/epoch_200_11_01_2022__19_39_"
    ######## TEST SET SELECTION - PARAMETERS ########
    
    test_multi_agent_setup = 2             # 1 for Medium size test, 2 for Big size test
    test_n_agents = 5                      # Number of agents to test (3 - 5 - 7 for Medium, 5 - 7 - 10 for Big)
    test_malfunctions_enabled = False      # Malfunctions enabled?
    test_agents_one_speed = True           # Test agents with the same speed (1) or with 4 different speeds?

    #################################################

    # Medium size
    if test_multi_agent_setup == 1:
        x_dim = 16*3
        y_dim = 9*3
        max_num_cities = 5
        max_rails_between_cities = 2
        max_rails_in_city = 3

    # Big size
    if test_multi_agent_setup == 2:
        x_dim = 16*4
        y_dim = 9*4
        max_num_cities = 9
        max_rails_between_cities = 5
        max_rails_in_city = 5
        
    args.width = x_dim
    args.height = y_dim


    stochastic_data = {'malfunction_rate': 80,  # Rate of malfunction occurence of single agent
                       'min_duration': 15,  # Minimal duration of malfunction
                       'max_duration': 50  # Max duration of malfunction
                       }


        # Different agent types (trains) with different speeds.
    if test_agents_one_speed:
        speed_ration_map = {1.: 1.,  # Fast passenger train
                            1. / 2.: 0.0,  # Fast freight train
                            1. / 3.: 0.0,  # Slow commuter train
                            1. / 4.: 0.0}  # Slow freight train
    else:
        speed_ration_map = {1.: 0.25,  # Fast passenger train
                            1. / 2.: 0.25,  # Fast freight train
                            1. / 3.: 0.25,  # Slow commuter train
                            1. / 4.: 0.25}  # Slow freight train

    observation_depth = 2
    observation_builder = GraphObservation(2)
    if test_malfunctions_enabled:
        env = RailEnv(width=x_dim,
                      height=y_dim,
                      rail_generator=sparse_rail_generator(max_num_cities=max_num_cities,
                                                           # Number of cities in map (where train stations are)
                                                           seed=14,  # Random seed
                                                           grid_mode=False,
                                                           max_rails_between_cities=max_rails_between_cities,
                                                               max_rails_in_city=max_rails_in_city),
                    schedule_generator=sparse_schedule_generator(speed_ration_map),
                    malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                    number_of_agents=test_n_agents,
                    obs_builder_object=observation_builder)
    else:
        env = RailEnv(width=x_dim,
                      height=y_dim,
                      rail_generator=sparse_rail_generator(max_num_cities=max_num_cities,
                                                           # Number of cities in map (where train stations are)
                                                           seed=14,  # Random seed
                                                           grid_mode=False,
                                                           max_rails_between_cities=max_rails_between_cities,
                                                               max_rails_in_city=max_rails_in_city),
                    schedule_generator=sparse_schedule_generator(speed_ration_map),
                    number_of_agents=test_n_agents,
                    obs_builder_object=observation_builder)
    
    env.reset()

    state_size = 12+2
    rl_agent = Agent(
        args=args,
        state_size=state_size,
        obs_builder=observation_builder)

    rl_agent.load(agent_weights_path)


    if 'n_trials' not in locals():
        n_trials = 10
    
    # max_steps computation
    speed_weighted_mean = 0

    for key in speed_ration_map.keys():
        speed_weighted_mean += key * speed_ration_map[key]
    
    #max_steps = int(3 * (env.height + env.width))
    max_steps = int((1/speed_weighted_mean) * 3 * (env.height + env.width))



    """
    # metrics['steps'].append(T)
    metrics['episodes'].append(ep)
    T_rewards = []  # List of episodes rewards
    T_Qs = []  # List
    T_num_done_agents = []  # List of number of done agents for each episode
    T_all_done = []  # If all agents completed in each episode
    T_agents_deadlock = []
    network_action_dict = dict()
    """
    railenv_action_dict = dict()

    
    #rewards_dict = dict()
    #rewards_dict.done_reward = 0
    #rewards_dict.deadlock_reward = 0
    rewards_dict = {'done_reward':0, 'deadlock_rewad':0}

    print("--------------- TESTING STARTED ------------------")
    # Test performance over several episodes
    if True:
        for ep in range(n_trials):
            # reward_sum contains the cumulative reward obtained as sum during
            # the steps
            logging.debug("Episode {} of {}".format(ep, n_trials))
            ep_controller = EpisodeController(env, rl_agent, max_steps)


            obs, info = env.reset()
            ep_controller.reset()
            
            # first action
            for a in range(env.get_num_agents()):
                agent = env.agents[a]
                ep_controller.agent_obs[a] = obs[a].copy()

                #agent_action = ep_controller.compute_agent_action(a, info, eps)
                agent_action = RailEnvActions.STOP_MOVING  #TODO
                railenv_action_dict.update({a: agent_action})
                ep_controller.agent_old_speed_data.update({a: agent.speed_data})
                env.obs_builder.agent_requires_obs.update({a: True})

            # env step returns next observations, rewards
            next_obs, all_rewards, done, info = env.step(railenv_action_dict)
            
            eps = 0
            
            # MULTI AGENT
            # initialize actions for all agents in this episode
            num_agents = env.get_num_agents()
            actions = torch.randint(0, 2, size=(num_agents,))

            # Main loop
            for step in range(max_steps):
                logging.debug(
                    "------------------------------------------------------------")
                logging.debug(
                    "------------------------------------------------------------")
                logging.debug(
                    '\r{} Agents on ({},{}).\n Ep: {}\t Step/MaxSteps: {} / {}'.format(
                        env.get_num_agents(), x_dim, y_dim,
                        ep,
                        step + 1,
                        max_steps, end=" "))
                        
                        
                # MULTI AGENT
                states = [env.obs_builder.preprocess_agent_obs(ep_controller.agent_obs[i], i) for i in range(num_agents)]
                
                # step together
                def infer_acts(states, actions, num_iter=3):
                    N = actions.shape[0]
                    mean_fields = torch.zeros(N,2).to(device)
                    actions_ = actions.clone()
                    q_values = torch.zeros(N).to(device)
                    
                    # calculating distance matrix of all agents
                    if N > 4:
                        positions = [(0,0) for _ in range(num_agents)]
                        distance_matrix = [[0] * num_agents for _ in range(num_agents)]
                        for i in range(num_agents):
                            agent = env.agents[i]
                            positions[i] = agent.position if agent.position != None else agent.initial_position
                        #print(positions)
                        for i in range(num_agents):
                            for j in range(i+1, num_agents):
                                distance_matrix[i][j] = abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])
                                distance_matrix[j][i] = distance_matrix[i][j]
                        #print(distance_matrix)
                
                    #for i in range(num_iter):
                    if N <= 4:
                        onehot_actions = torch.nn.functional.one_hot(actions_, num_classes=2)
                        for j in range(N):
                            mean_fields[j] = torch.mean(onehot_actions.float(), dim=0) #Category actions to vectors first
                    else:
                        for j in range(N):
                            # select all other agents
                            #pre_actions = torch.index_select(actions_, 0, torch.LongTensor(range(j)))
                            #aft_actions = torch.index_select(actions_, 0, torch.LongTensor(range(j+1,N)))
                            #other_actions = torch.cat((pre_actions, aft_actions))
                            
                            # select 3 nearest neighbors
                            neighbors = np.argpartition(distance_matrix[j], 4)[:4]
                            #print("neighbors: {}".format(neighbors))
                            neighbor_actions = actions_[neighbors]
                            #print("neigh_actions:{}".format(neighbor_actions))
                            neighbor_actions = torch.nn.functional.one_hot(neighbor_actions, num_classes=2) 
                            mean_fields[j] = torch.mean(neighbor_actions.float(), dim=0) #Category actions to vectors first
                    for j in range(N):
                        # concatenate state and mf
                        state = states[j].to(device).clone()
                        new_x = torch.cat([state.x, mean_fields[j].repeat(state.x.shape[0],1)], dim=1)
                        state.x = new_x
                        # calculate q and action
                        q_action = rl_agent.act(state)
                        q_values[j] = q_action[j][3]
                        actions_[j] = q_action[j][1]
                        
                    return actions_, mean_fields, q_values
                 
                actions, mean_fields, q_values = infer_acts(states, actions)
                
                

                # for each agent
                for a in range(env.get_num_agents()):
                    agent = env.agents[a]
                    # compute action for agent a and update dict
                    agent_next_action = ep_controller.compute_agent_action(a, info, eps, mean_fields[a])
                    railenv_action_dict.update({a: agent_next_action})
                # Environment step
                next_obs, all_rewards, done, info = env.step(railenv_action_dict)
                
                # MULTI AGENT
                next_states = [None for i in range(env.get_num_agents())]
                for i, obs in enumerate(next_obs):
                    if len(next_obs[obs]) == 0:
                        next_states[i] = states[i]
                    else:
                        next_states[i] = env.obs_builder.preprocess_agent_obs(next_obs[i], i)
                _, _, next_q_values = infer_acts(next_states, actions)
                #print("#AGENTS: {}, values: {} for next Q".format(num_agents, next_q_values))

                # Update replay buffer and train agent
                for a in range(env.get_num_agents()):
                    ep_controller.save_experience_and_train(a, railenv_action_dict[a], all_rewards[a], next_obs[a], done[a], step, args, ep, mean_fields[a], next_q_values[a], train=False)
                if ep_controller.is_episode_done():
                    break

            # end of episode

            ep_controller.print_episode_stats(ep, args, eps, step)

            wandb_log_dict = ep_controller.retrieve_wandb_log(eps)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Flatland')

    # Flatland env parameters
    parser.add_argument('--width', type=int, default=25,
                        help='Environment width')
    parser.add_argument('--height', type=int, default=25,
                        help='Environment height')
    parser.add_argument('--num-agents', type=int, default=1,
                        help='Number of agents in the environment')
    parser.add_argument('--max-num-cities', type=int, default=2,
                        help='Maximum number of cities where agents can start or end')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed used to generate grid environment randomly')
    parser.add_argument('--grid-mode', type=bool, default=False,
                        help='Type of city distribution, if False cities are randomly placed')
    parser.add_argument('--max-rails-between-cities', type=int, default=2,
                        help='Max number of tracks allowed between cities, these count as entry points to a city')
    parser.add_argument('--max-rails-in-city', type=int, default=4,
                        help='Max number of parallel tracks within a city allowed')
    parser.add_argument('--malfunction-rate', type=int, default=0,
                        help='Rate of malfunction occurrence of single agent')
    parser.add_argument('--min-duration', type=int,
                        default=20, help='Min duration of malfunction')
    parser.add_argument('--max-duration', type=int,
                        default=50, help='Max duration of malfunction')
    parser.add_argument('--observation-builder', type=str, default='GraphObsForRailEnv',
                        help='Class to use to build observation for agent')
    parser.add_argument('--predictor', type=str, default='ShortestPathPredictorForRailEnv',
                        help='Class used to predict agent paths and help observation building')
    parser.add_argument('--view-semiwidth', type=int, default=7,
                        help='Semiwidth of field view for agent in local obs')
    parser.add_argument('--view-height', type=int, default=30,
                        help='Height of the field view for agent in local obs')
    parser.add_argument('--offset', type=int, default=25,
                        help='Offset of agent in local obs')
    parser.add_argument('--observation-depth', type=int, default=3,
                        help='Depth of observation graph of each agent')

    # Training parameters
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Number of episodes on which to train the agents')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Epoch from which resume training (useful for stats)')
    parser.add_argument('--max-steps', type=int, default=300,
                        help='Maximum number of steps for each episode')
    parser.add_argument('--eps', type=float, default=1,
                        help='epsilon value for e-greedy')
    parser.add_argument('--debug-print', type=bool, default=False,
                        help='requires debug printing')
    parser.add_argument('--load-memory', type=bool, default=False,
                        help='if load saved memory')
    parser.add_argument('--evaluation-episodes', type=int, default=3,
                        metavar='N', help='Number of evaluation episodes to average over')
    parser.add_argument('--render', action='store_true',
                        default=False, help='Display screen (testing only)')
    parser.add_argument('--evaluation-interval', type=int, default=10, metavar='EPISODES', help='Number of episodes between evaluations')
    parser.add_argument('--save-model-interval', type=int, default=50,
                        help='Save models every tot episodes')
    parser.add_argument('--start-lr-decay', type=int, default=150,
                        help='Save models every tot episodes')
    parser.add_argument('--eps-decay', type=float, default=0.999,
                        help='epsilon decay value')
    parser.add_argument('--learning-rate', type=float, default=0.02,
                        help='LR for DQN agent')
    parser.add_argument('--learning-rate-decay', type=float, default=0.5,
                        help='LR decay for DQN agent')
    parser.add_argument('--learning-rate-decay-policy', type=float, default=0.5,
                        help='LR decay for policy network')

    # WANDB Logging
    parser.add_argument('--run-title', type=str, default="first_run",
                        help='title for tensorboard run')
    parser.add_argument('--wandb-project-name', type=str, default="wandb_default",
                        help='title for wandb run')
    

    # Model arguments
    parser.add_argument('--model-path', type=str, default='test_results/', help="result directory")
    parser.add_argument('--model-name', type=str, default='',
                        help='Name to use to save the model .pth')
    parser.add_argument('--gat-layers', type=int, default=3,
                        help='Number of GAT layers for the model')
    parser.add_argument('--flow', type=str, default="source_to_target",
                        help='Message passing flow for graph neural networks')
    parser.add_argument('--resume-weights', type=bool, default=True,
                        help='True if load previous weights')
    parser.add_argument('--batch-norm', type=bool, default=False,
                        help='True if load previous weights')
    parser.add_argument('--dropout-rate', type=float, default=0.8,
                        help='Dropout rate for the model layers')
    parser.add_argument('--attention-heads', type=int, default=4,
                        help='Attention heads of GAT layer') 
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')    
    parser.add_argument('--use-stop-action', type=bool, default=False,
                        help='Whether to use STOP action')               

                        
    # Rewards
    parser.add_argument('--done-reward', type=int, default=0,
                        help='Reward given to agent when it reaches target')
    parser.add_argument('--deadlock-reward', type=int, default=-1000,
                        help='Reward given to agent when it reaches deadlock')
    parser.add_argument('--reward-scaling', type=float, default=0.1,
                        help='Reward scaling factor')

    

    args = parser.parse_args()

    main(args)

