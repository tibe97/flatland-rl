from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from print_info import print_info
from dueling_double_dqn import Agent
from predictions import ShortestPathPredictorForRailEnv
from graph_for_observation import GraphObservation
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.agent_utils import RailAgentStatus
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
# make sure the root path is in system path
from pathlib import Path

import logging


# These 2 lines must go before the import from src/
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))


def main(args):
    # print debug info or not
    if args.debug_print:
        print("debug print active")
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.NOTSET)
        logging.getLogger().disabled = True

    start_print = "About to train {} agents on ({},{}) env.\nParameters:\nmax_num_cities: {}\nmax_rails_between_cities: {}\nmax_rails_in_city: {}\nmalfunction_rate: {}\nmax_duration: {}\nmin_duration: {}\nnum_episodes: {}\nmax_steps: {}\n".format(
        args.num_agents,
        args.width,
        args.height,
        args.max_num_cities,
        args.max_rails_between_cities,
        args.max_rails_in_city,
        args.malfunction_rate,
        args.max_duration,
        args.min_duration,
        args.num_episodes,
        args.max_steps 
    )
    print(start_print)
    with open(args.model_path + 'training_stats.txt', 'a') as f:
        print(start_print, file=f, end=" ")

    rail_generator = sparse_rail_generator(max_num_cities=args.max_num_cities,
                                           seed=args.seed,
                                           grid_mode=args.grid_mode,
                                           max_rails_between_cities=args.max_rails_between_cities,
                                           max_rails_in_city=args.max_rails_in_city,
                                           )
    # Maps speeds to % of appearance in the env
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    schedule_generator = sparse_schedule_generator(speed_ration_map)

    stochastic_data = MalfunctionParameters(
        malfunction_rate=args.malfunction_rate,  # Rate of malfunction occurrence
        min_duration=args.min_duration,  # Minimal duration of malfunction
        max_duration=args.max_duration)  # Max duration of malfunction

    if args.observation_builder == 'GraphObsForRailEnv':
        prediction_depth = args.prediction_depth
        bfs_depth = args.bfs_depth
        observation_builder = GraphObservation()
        state_size = 12
        network_action_size = 4  # {follow path, stop}
        railenv_action_size = 4  # The RailEnv possible actions
        dqn_agent = Agent(
            state_size=state_size, action_size=network_action_size, obs_builder=observation_builder)

    # Construct the environment with the given observation, generators, predictors, and stochastic data
    env = RailEnv(width=args.width,
                  height=args.height,
                  rail_generator=rail_generator,
                  schedule_generator=schedule_generator,
                  number_of_agents=args.num_agents,
                  obs_builder_object=observation_builder,
                  malfunction_generator_and_process_data=malfunction_from_params(
                      stochastic_data),
                  remove_agents_at_target=True)
    env.reset()

    # max_steps = env.compute_max_episode_steps(args.width, args.height, args.num_agents/args.max_num_cities)
    max_steps = args.max_steps  # TODO DEBUG
    eps = args.eps
    eps_end = 0.005
    eps_decay = 0.999
    # Need to have two since env works with RailEnv actions but agent works with network actions
    railenv_action_dict = dict()
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    scores = []
    dones_list = []

    # Load previous weight is available
    if args.resume_weights:
        dqn_agent.load(args.model_path + "results/" + args.model_name)

    for ep in range(1, args.num_episodes + 1):
        logging.debug("Episode {} of {}".format(ep, args.num_episodes))
        obs, info = env.reset()
        # env.obs_builder.render_environment()
        agent_obs = [None] * env.get_num_agents()
        agent_obs_buffer = [None] * env.get_num_agents()  # updated twice
        agent_action_buffer = defaultdict(list)
        # some switches are long so we have to accumulate reward or penalty
        acc_rewards = defaultdict(lambda: 0)
        # how many timesteps remaining to complete current action
        agents_speed_timesteps = [0] * env.get_num_agents()
        agent_at_switch = [False] * env.get_num_agents()
        agent_path_obs_buffer = [None] * env.get_num_agents()
        # Used to check if agent with fractionary speed could move at current cell
        agent_old_speed_data = {}
        agent_done_removed = [False] * env.get_num_agents()
        update_values = [False] * env.get_num_agents()
        agents_in_deadlock = [False] * env.get_num_agents()
        #################
        # if ep < 1000: continue
        #################

        num_agents_at_switch = 0
        for a in range(env.get_num_agents()):
            agent = env.agents[a]
            if env.obs_builder.get_track(agent.initial_position) == -2:
                num_agents_at_switch += 1
            agent_obs[a] = obs[a].copy()
            agent_obs_buffer[a] = agent_obs[a].copy()
            action = RailEnvActions.STOP_MOVING  # TODO
            # All'inizio faccio partire a random TODO Prova
            railenv_action_dict.update({a: action})
            agent_old_speed_data.update({a: agent.speed_data})
            env.obs_builder.agent_requires_obs.update({a: True})

        next_obs, all_rewards, done, info = env.step(railenv_action_dict)

        logging.debug("{} agents at switch".format(num_agents_at_switch))

        score = 0
        env_done = 0

        # Main loop
        for step in range(max_steps):
            logging.debug(
                "------------------------------------------------------------")
            logging.debug(
                "------------------------------------------------------------")
            logging.debug(
                '\r{} Agents on ({},{}).\n Ep: {}\t Step/MaxSteps: {} / {}'.format(
                    env.get_num_agents(), args.width, args.height,
                    ep,
                    step+1,
                    max_steps, end=" "))

            # Logging
            # print_info(env)
            num_active_agents = 0
            num_agents_not_started = 0
            num_agents_done = 0
            agents_with_malfunctions = 0

            for a in range(env.get_num_agents()):
                agent = env.agents[a]
                if env.agents[a].status == RailAgentStatus.ACTIVE:
                    num_active_agents += 1
                    if agent.malfunction_data["malfunction"] > 0:
                        agents_with_malfunctions += 1
                    agent_position = agent.position
                elif agent.status == RailAgentStatus.READY_TO_DEPART:
                    num_agents_not_started += 1
                    agent_position = agent.initial_position
                elif agent.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                    num_agents_done += 1

                logging.debug("Agent {} at position {}, fraction {}".format(
                    a, agent.position, agent.speed_data["position_fraction"]))

                if info['action_required'][a] and agent_at_switch[a] and agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.READY_TO_DEPART] and agent.malfunction_data["malfunction"] == 0:
                    # if we finish previous action (action may take more than 1 timestep)
                    if agents_speed_timesteps[a] == 0:
                        # if we arrive before a switch we compute the next path to reach and the actions required
                        # (this is due to the switch being eventually composed of more cells)
                        # We are about to enter the switch
                        if len(agent_action_buffer[a]) == 0:
                            # Check that dict is not empty
                            assert agent_obs[a]["partitioned"]
                            obs_batch = env.obs_builder.preprocess_agent_obs(
                                agent_obs[a], a)
                            # Choose path to take at the current switch
                            path_values = dqn_agent.act(obs_batch, eps=eps)
                            railenv_action = env.obs_builder.choose_railenv_actions(
                                a, path_values[a][0])
                            agent_action_buffer[a] = railenv_action
                            # as state to save we take the path chosen by agent
                            agent_path_obs_buffer[a] = agent_obs[a]["partitioned"][path_values[a][0]]
                            logging.debug("Agent {} choses path {} at position {}. Num actions to take: {}".format(
                                a, path_values[a][0], agent.position, len(agent_action_buffer[a])))
                            logging.debug(
                                "Agent {} actions: {}".format(a, railenv_action))
                        next_action = agent_action_buffer[a].pop(0)
                        logging.debug("Agent {} at: {}. Action is: {}. Speed: {}. Fraction {}. Remaining actions: {}. SpeedTimesteps: {}".format(
                            a, agent.position, next_action, agent.speed_data["speed"], agent.speed_data["position_fraction"], len(agent_action_buffer[a]), agents_speed_timesteps[a]))
                        # if agent has to stop, do it for 1 timestep
                        if (next_action == RailEnvActions.STOP_MOVING):
                            agents_speed_timesteps[a] = 1
                            env.obs_builder.agent_requires_obs.update(
                                {a: True})
                        else:
                            # speed is a fractionary value between 0 and 1
                            agents_speed_timesteps[a] = int(
                                round(1 / info["speed"][a]))

                elif agent.status != RailAgentStatus.DONE_REMOVED:  # When going straight
                    next_action = 0
                    if not agent.moving and agent.malfunction_data["malfunction"] == 0:
                        valid_move_actions = get_valid_move_actions_(
                            agent.direction, agent_position, env.rail)
                        # agent could be at switch, so more actions possible
                        assert len(valid_move_actions) >= 1
                        next_action = valid_move_actions.popitem()[0][2]

                # Update action dicts
                railenv_action_dict.update({a: next_action})

            # Environment step
            next_obs, all_rewards, done, info = env.step(railenv_action_dict)

            logging.debug("Active agents: {}. Not started yet: {}. With malfunctions: {}".format(
                num_active_agents, num_agents_not_started, agents_with_malfunctions))
            logging.debug("-------- STEP DONE -------------")

            # Update replay buffer and train agent
            for a in range(env.get_num_agents()):
                agent = env.agents[a]
                logging.debug("Agent {} at position {}, fraction {}, speed Timesteps {}".format(
                    a, agent.position, agent.speed_data["position_fraction"], agents_speed_timesteps[a]))
                # if agent didn't move do nothing: agent couldn't perform action because another agent
                # occupied next cell or agent's action was STOP
                # print("Agent {}, old_fraction {}, new fraction {}".format(a, agent_old_speed_data[a]["position_fraction"], agent.speed_data["position_fraction"]))

                score += all_rewards[a] / env.get_num_agents()  # Update score

                # if agent didn't move don't do anything
                if env.obs_builder.agent_could_move(a, railenv_action_dict[a], agent_old_speed_data[a]):

                    # update replay memory
                    if ((update_values[a] and agent.speed_data["position_fraction"] == 0) or agent.status == RailAgentStatus.DONE_REMOVED) and not agent_done_removed[a]:
                        logging.debug("Update=True: agent {}".format(a))
                        # next state is the complete state, with all the possible path choices
                        if len(next_obs[a]) > 0 and len(agent_path_obs_buffer[a]) > 0:
                            dqn_agent.step(
                                agent_path_obs_buffer[a], acc_rewards[a], next_obs[a], done[a], agents_in_deadlock[a])
                            acc_rewards[a] = 0
                            update_values[a] = False
                            # agent_obs_buffer[a] = agent_obs[a].copy()
                            if agent.status == RailAgentStatus.DONE_REMOVED or agent.status == RailAgentStatus.DONE:
                                logging.debug(
                                    "Agent {} DONE! It has been removed and experience saved!".format(a))
                                agent_done_removed[a] = True

                    if len(next_obs[a]) > 0:
                        agent_obs[a] = next_obs[a].copy()

                    if not agent_at_switch[a]:
                        # agent_obs_buffer[a] = agent_obs[a].copy()
                        acc_rewards[a] += all_rewards[a]
                        # accumulate in case there is a deadlock (highly penalized as reward is negatively accum. till end of episode)
                    else:
                        agents_speed_timesteps[a] -= 1
                        # accumulate when passing a switch
                        acc_rewards[a] += all_rewards[a]

                    # At next step we compute observations only in these cases:
                    # 1. Agent is entering switch (obs for last cell of current path)
                    # 2. Agent is exiting a switch (obs for new cell of new path)
                    # 3. Agent is about to finish
                    env.obs_builder.agent_requires_obs.update({a: False})

                    if agent.status == RailAgentStatus.ACTIVE:
                        # Compute when agent is about to enter a switch and when it's about to leave a switch
                        # PURPOSE: to compute observations only when needed, i.e. before a switch and after, also before and
                        # after making an action that leads to the target of an agent
                        if not agent_at_switch[a]:
                            agent_pos = agent.position
                            assert env.obs_builder.get_track(agent_pos) != -2
                            if env.obs_builder.is_agent_entering_switch(a) and agent.speed_data["position_fraction"] == 0:
                                logging.debug(
                                    "Agent {} arrived at 1 cell before switch".format(a))
                                agent_at_switch[a] = True
                                acc_rewards[a] = 0
                                agents_speed_timesteps[a] = 0
                                # env.obs_builder.agent_requires_obs.update({a: False})
                            elif env.obs_builder.is_agent_2_steps_from_switch(a):
                                env.obs_builder.agent_requires_obs.update(
                                    {a: True})
                            if env.obs_builder.is_agent_about_to_finish(a):
                                env.obs_builder.agent_requires_obs.update(
                                    {a: True})
                        else:  # Agent at SWITCH. In the step before reaching target path we want to make sure to compute the obs
                            # in order to update the replay memory. We need to be careful if the agent can't reach new path because of another agent blocking the cell
                            # when agent speed is 1 we reach the target in 1 step
                            if len(agent_action_buffer[a]) == 1 and agent.speed_data["speed"] == 1:
                                agent_next_action = agent_action_buffer[a][0]
                                assert env.obs_builder.is_agent_exiting_switch(
                                    a, agent_next_action)
                                update_values[a] = True
                                env.obs_builder.agent_requires_obs.update(
                                    {a: True})
                            # if speed is less than 1, we need more steps to reach target. So only compute obs if doing last step
                            elif len(agent_action_buffer[a]) == 0:
                                if env.obs_builder.get_track(agent.position) == -2 and agent.speed_data["speed"] < 1 and np.isclose(agent.speed_data["speed"] + agent.speed_data["position_fraction"], 1, rtol=1e-03):
                                    # same check as "if" condition
                                    assert agents_speed_timesteps[a] > 0
                                    update_values[a] = True
                                    env.obs_builder.agent_requires_obs.update(
                                        {a: True})
                                else:
                                    if env.obs_builder.get_track(agent.position) != -2:
                                        if env.obs_builder.is_agent_entering_switch(a):
                                            assert len(next_obs[a]) > 0
                                            logging.debug(
                                                "Agent {} just exited switch and ALREADY entering another one".format(a))
                                            agent_obs_buffer[a] = next_obs[a].copy(
                                            )
                                        else:
                                            logging.debug(
                                                "Agent {} is not at switch anymore".format(a))
                                            agent_at_switch[a] = False
                                            agents_speed_timesteps[a] = 0
                                            agent_obs_buffer[a] = next_obs[a].copy(
                                            )
                                        if env.obs_builder.is_agent_about_to_finish(a):
                                            env.obs_builder.agent_requires_obs.update(
                                                {a: True})

                else:  # agent did not move. Check if it stopped on purpose
                    acc_rewards[a] += all_rewards[a]
                    if railenv_action_dict[a] == RailEnvActions.STOP_MOVING:
                        agents_speed_timesteps[a] -= 1
                        env.obs_builder.agent_requires_obs.update({a: False})
                        # When agent stops, store the experience
                        if len(next_obs[a]) > 0 and agent_path_obs_buffer[a] is not None:
                            # TODO: for now avoid deadlocks at switches, hard to handle
                            if len(next_obs[a]["partitioned"]) > 0:
                                dqn_agent.step(
                                    agent_path_obs_buffer[a], acc_rewards[a], next_obs[a], done[a], agents_in_deadlock[a])
                    else:
                        logging.debug("Agent {} cannot move at position {}, fraction {}".format(
                            a, agent.position, agent.speed_data["position_fraction"]))
                        if agents_in_deadlock[a] or env.obs_builder.is_agent_in_deadlock(a):
                            agents_in_deadlock[a] = True
                            env.obs_builder.agent_requires_obs.update(
                                {a: False})
                            if step == max_steps-2:
                                env.obs_builder.agent_requires_obs.update(
                                    {a: True})
                            if step == max_steps-1:  # At last timestep we save the experience of agents in deadlock
                                logging.debug("Agent {} in DEADLOCK saved as experience with reward of {}".format(
                                    a, acc_rewards[a]))
                                if len(next_obs[a]) > 0 and agent_path_obs_buffer[a] is not None:
                                    # TODO: for now avoid deadlocks at switches, hard to handle
                                    if len(next_obs[a]["partitioned"]) > 0:
                                        dqn_agent.step(
                                            agent_path_obs_buffer[a], acc_rewards[a], next_obs[a], done[a], agents_in_deadlock[a])
                            logging.debug("Agent {} is in DEADLOCK, accum. reward: {}, required_obs: {}".format(
                                a, acc_rewards[a], env.obs_builder.agent_requires_obs[a]))

                agent_old_speed_data.update({a: agent.speed_data.copy()})

            if done['__all__']:
                env_done = 1
                break

        # At the end of the episode TODO This part could be done in another script
        eps = max(eps_end, eps_decay * eps)  # Decrease epsilon
        # Metrics
        done_window.append(env_done)
        num_agents_done = 0  # Num of agents that reached their target
        num_agents_in_deadlock = 0
        num_agents_in_deadlock_at_switch = 0
        for a in range(env.get_num_agents()):
            if done[a]:
                num_agents_done += 1
            elif env.obs_builder.is_agent_in_deadlock(a):
                num_agents_in_deadlock += 1
                if env.obs_builder.get_track(env.agents[a].position) == -2:
                    num_agents_in_deadlock_at_switch += 1

        scores_window.append(score / max_steps)  # Save most recent score
        scores.append(np.mean(scores_window))
        dones_list.append((np.mean(done_window)))

        # Print training results info
        episode_stats = '\rEp: {}\t {} Agents on ({},{}).\t Ep score {:.3f}\tAvg Score: {:.3f}\t Env Dones so far: {:.2f}%\t Done Agents in ep: {:.2f}%\t In deadlock {:.2f}%(at switch {})\t Not started {}\t Eps: {:.2f}\n'.format(
            ep,
            env.get_num_agents(), args.width, args.height,
            score,
            np.mean(scores_window),
            100 * np.mean(done_window),
            100 * (num_agents_done/args.num_agents),
            100 * (num_agents_in_deadlock/args.num_agents),
            (num_agents_in_deadlock_at_switch),
            num_agents_not_started,
            eps)
        print(episode_stats, end=" ")
        with open(args.model_path + 'training_stats.txt', 'a') as f:
            print(episode_stats, file=f, end=" ")

        if ep % 20 == 0:
            dqn_agent.save(args.model_path + args.model_name)  # Save models
            eps_stats = '\rTraining {} Agents.\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \n'.format(
                env.get_num_agents(),
                ep,
                np.mean(scores_window),
                100 * (num_agents_done/args.num_agents),
                eps)
            print(eps_stats)
            with open(args.model_path + 'training_stats.txt', 'a') as f:
                print(eps_stats, file=f)

        if ep % 100 == 0:  # backup weights
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y__%H_%M_")
            if not os.path.exists(args.model_path + "checkpoint"):
                os.makedirs(args.model_path + "checkpoint")
            dqn_agent.save(args.model_path + "checkpoint/" +
                           dt_string + args.model_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Flatland')
    # Flatland parameters
    parser.add_argument('--width', type=int, default=50,
                        help='Environment width')
    parser.add_argument('--height', type=int, default=50,
                        help='Environment height')
    parser.add_argument('--num-agents', type=int, default=10,
                        help='Number of agents in the environment')
    parser.add_argument('--max-num-cities', type=int, default=6,
                        help='Maximum number of cities where agents can start or end')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed used to generate grid environment randomly')
    parser.add_argument('--grid-mode', type=bool, default=False,
                        help='Type of city distribution, if False cities are randomly placed')
    parser.add_argument('--max-rails-between-cities', type=int, default=4,
                        help='Max number of tracks allowed between cities, these count as entry points to a city')
    parser.add_argument('--max-rails-in-city', type=int, default=6,
                        help='Max number of parallel tracks within a city allowed')
    parser.add_argument('--malfunction-rate', type=int, default=0,
                        help='Rate of malfunction occurrence of single agent')
    parser.add_argument('--min-duration', type=int,
                        default=5, help='Min duration of malfunction')
    parser.add_argument('--max-duration', type=int,
                        default=10, help='Max duration of malfunction')
    parser.add_argument('--observation-builder', type=str, default='GraphObsForRailEnv',
                        help='Class to use to build observation for agent')
    parser.add_argument('--predictor', type=str, default='ShortestPathPredictorForRailEnv',
                        help='Class used to predict agent paths and help observation building')
    parser.add_argument('--bfs-depth', type=int, default=4,
                        help='BFS depth of the graph observation')
    parser.add_argument('--prediction-depth', type=int, default=108,
                        help='Prediction depth for shortest path strategy, i.e. length of a path')
    parser.add_argument('--view-semiwidth', type=int, default=7,
                        help='Semiwidth of field view for agent in local obs')
    parser.add_argument('--view-height', type=int, default=30,
                        help='Height of the field view for agent in local obs')
    parser.add_argument('--offset', type=int, default=25,
                        help='Offset of agent in local obs')
    # Training parameters
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Number of episodes on which to train the agents')
    parser.add_argument('--max-steps', type=int, default=200,
                        help='Maximum number of steps for each episode')
    parser.add_argument('--eps', type=float, default=0.3,
                        help='epsilon value for e-greedy')
    parser.add_argument('--model-path', type=str, default='', help="")
    parser.add_argument('--model-name', type=str, default='dd_dqn',
                        help='Name to use to save the model .pth')
    parser.add_argument('--resume-weights', type=bool, default=True,
                        help='True if load previous weights')
    parser.add_argument('--debug-print', type=bool, default=True,
                        help='requires debug printing')
    # DDQN hyperparameters

    args = parser.parse_args()
    # Check arguments
    if args.offset > args.height:
        raise ValueError(
            "Agent offset can't be greater than view height in local obs")
    if args.offset < 0:
        raise ValueError("Agent offset must be a positive integer")

    main(args)
