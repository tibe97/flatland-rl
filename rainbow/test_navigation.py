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


# Test DQN
def main(args):

    random.seed(1)
    np.random.seed(1)


    agent_weights_path = ""
    ######## TEST SET SELECTION - PARAMETERS ########
    
    test_multi_agent_setup = 1             # 1 for Medium size test, 2 for Big size test
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

    state_size = 12
    agent = Agent(
        args=args,
        state_size=state_size,
        obs_builder=observation_builder)

    agent.load(agent_weights_path)


    if 'n_trials' not in locals():
        n_trials = 15000
    
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

    
    rewards_dict = dict()
    rewards_dict.done_reward = 0
    rewards_dict.deadlock_reward = 0

    print("--------------- TESTING STARTED ------------------")
    # Test performance over several episodes
    if True:
        for ep in range(n_trials):
            # reward_sum contains the cumulative reward obtained as sum during
            # the steps
            logging.debug("Episode {} of {}".format(ep, n_trials))
            ep_controller = EpisodeController(env, agent, max_steps)
            ep_controller.reset()

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

                # for each agent
                for a in range(env.get_num_agents()):
                    agent = env.agents[a]
                    # compute action for agent a and update dict
                    agent_next_action = ep_controller.compute_agent_action(
                        a, info, eps)
                    railenv_action_dict.update({a: agent_next_action})

                # Environment step
                next_obs, all_rewards, done, info = env.step(railenv_action_dict)

                # Update replay buffer and train agent
                for a in range(env.get_num_agents()):
                    ep_controller.save_experience_and_train(
                        a, railenv_action_dict[a], all_rewards[a], next_obs[a], done[a], step, rewards_dict, ep, train=False)

                if ep_controller.is_episode_done():
                    break

            # end of episode
            eps = 0

            ep_controller.print_episode_stats(ep, args, eps, step)

            wandb_log_dict = ep_controller.retrieve_wandb_log()




