from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from dueling_double_dqn import Agent
from graph_for_observation import GraphObservation
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
# make sure the root path is in system path
from pathlib import Path

import logging


# These 2 lines must go before the import from src/
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))


# Test DQN
def test(args, ep, dqn_agent, metrics, results_dir, evaluate=False):

    # Init env and set in evaluation mode
    # Maps speeds to % of appearance in the env
    '''
    speed_ration_map = {1.: 0.25,    # Fast passenger train
                        1. / 2.: 0.25,    # Fast freight train
                        1. / 3.: 0.25,    # Slow commuter train
                        1. / 4.: 0.25}    # Slow freight train
    '''
    all_actions = []
    speed_ration_map = {1.0: 1.0}
    schedule_generator = sparse_schedule_generator(speed_ration_map)

    observation_builder = GraphObservation(args.observation_depth)
    max_num_cities_adaptive = (args.num_agents//10)+2
    max_steps = int(4 * 2 * (args.width + args.height + args.num_agents / max_num_cities_adaptive))

    env = RailEnv(
        width=args.width,
        height=args.height,
        rail_generator=sparse_rail_generator(
            max_num_cities=max_num_cities_adaptive,
            seed=ep,  # Use episode as seed when evaluation is performed during training
            grid_mode=args.grid_mode,
            max_rails_between_cities=args.max_rails_between_cities,
            max_rails_in_city=args.max_rails_in_city,
        ),
        schedule_generator=schedule_generator,
        number_of_agents=args.num_agents,
        obs_builder_object=observation_builder,
        malfunction_generator_and_process_data= malfunction_from_params(MalfunctionParameters(
            malfunction_rate=args.malfunction_rate,
            min_duration=args.min_duration,
            max_duration=args.max_duration
        )),
    )

    if args.render:
        env_renderer = RenderTool(
            env,
            #gl="PILSVG",
            #agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
            show_debug=False,
            screen_height=1080,
            screen_width=1920)

    
    # metrics['steps'].append(T)
    metrics['episodes'].append(ep)
    T_rewards = []  # List of episodes rewards
    T_Qs = []  # List
    T_num_done_agents = []  # List of number of done agents for each episode
    T_all_done = []  # If all agents completed in each episode
    T_agents_deadlock = []
    network_action_dict = dict()
    railenv_action_dict = dict()
    
    print("--------------- TESTING STARTED ------------------")
    # Test performance over several episodes
    for ep in range(args.evaluation_episodes):
        # reward_sum contains the cumulative reward obtained as sum during the steps
        reward_sum, all_done = 0, False
        num_done_agents = 0
        if args.render:
            env_renderer.reset()

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
        agents_in_deadlock = [False] * env.get_num_agents()

        num_agents_at_switch = 0
        for a in range(env.get_num_agents()):
            agent = env.agents[a]
            if env.obs_builder.get_track(agent.initial_position) == -2:
                num_agents_at_switch += 1
            agent_obs[a] = obs[a].copy()
            agent_obs_buffer[a] = agent_obs[a].copy()
            action = RailEnvActions.STOP_MOVING  # TODO
            railenv_action_dict.update({a: action})
            agent_old_speed_data.update({a: agent.speed_data})
            env.obs_builder.agent_requires_obs.update({a: True})

        next_obs, all_rewards, done, info = env.step(railenv_action_dict)

        reward_sum += sum([all_rewards[a] for a in range(env.get_num_agents())])


        score = 0

        # Main loop
        for step in range(max_steps):
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
                            path_values = dqn_agent.act(obs_batch, eps=0)
                            railenv_action = env.obs_builder.choose_railenv_actions(
                                a, path_values[a])
                            agent_action_buffer[a] = railenv_action
                            # as state to save we take the path chosen by agent
                            agent_path_obs_buffer[a] = agent_obs[a]["partitioned"][path_values[a][0]]

                        next_action = agent_action_buffer[a].pop(0)
                        all_actions.append(int(next_action))

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
                    if agent.status == RailAgentStatus.READY_TO_DEPART or (not agent.moving and agent.malfunction_data["malfunction"] == 0):
                        valid_move_actions = get_valid_move_actions_(
                            agent.direction, agent_position, env.rail)
                        # agent could be at switch, so more actions possible
                        assert len(valid_move_actions) >= 1
                        next_action = valid_move_actions.popitem()[0].action

                # Update action dicts
                railenv_action_dict.update({a: next_action})

            # Environment step
            next_obs, all_rewards, done, info = env.step(railenv_action_dict)

            reward_sum += sum([all_rewards[a]
                              for a in range(env.get_num_agents())])

            # Update replay buffer and train agent
            for a in range(env.get_num_agents()):
                agent = env.agents[a]
                if not agent_done_removed[a]:
                    if agent.status == RailAgentStatus.DONE_REMOVED or agent.status == RailAgentStatus.DONE:
                        agent_done_removed[a] = True
                # if agent didn't move do nothing: agent couldn't perform action because another agent
                # occupied next cell or agent's action was STOP
                # print("Agent {}, old_fraction {}, new fraction {}".format(a, agent_old_speed_data[a]["position_fraction"], agent.speed_data["position_fraction"]))
                score += all_rewards[a] / env.get_num_agents()  # Update score

                # if agent didn't move don't do anything
                if env.obs_builder.agent_could_move(a, railenv_action_dict[a], agent_old_speed_data[a]):

                    if len(next_obs[a]) > 0:
                        agent_obs[a] = next_obs[a].copy()

                    if agent_at_switch[a]:   
                        agents_speed_timesteps[a] -= 1
                        # accumulate when passing a switch
                        

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
                            if env.obs_builder.is_agent_entering_switch(a) and agent.speed_data["position_fraction"] == 0:
                                agent_at_switch[a] = True
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
                                env.obs_builder.agent_requires_obs.update(
                                    {a: True})
                            # if speed is less than 1, we need more steps to reach target. So only compute obs if doing last step
                            elif len(agent_action_buffer[a]) == 0:
                                if env.obs_builder.get_track(agent.position) == -2 and agent.speed_data["speed"] < 1 and np.isclose(agent.speed_data["speed"] + agent.speed_data["position_fraction"], 1, rtol=1e-03):
                                    # same check as "if" condition
                                    env.obs_builder.agent_requires_obs.update(
                                        {a: True})
                                else:
                                    if env.obs_builder.get_track(agent.position) != -2:
                                        if env.obs_builder.is_agent_entering_switch(a):
                                            agent_obs_buffer[a] = next_obs[a].copy(
                                            )
                                        else:
                                            agent_at_switch[a] = False
                                            agents_speed_timesteps[a] = 0
                                            agent_obs_buffer[a] = next_obs[a].copy(
                                            )
                                        

                else:  # agent did not move. Check if it stopped on purpose
                    if railenv_action_dict[a] == RailEnvActions.STOP_MOVING:
                        agents_speed_timesteps[a] -= 1
                        
                    else:
                        if agents_in_deadlock[a] or env.obs_builder.is_agent_in_deadlock(a):
                            agents_in_deadlock[a] = True

                agent_old_speed_data.update({a: agent.speed_data.copy()})

            
            if args.render:
                env_renderer.render_env(
                    show=True, show_observations=False, show_predictions=False)
            

            if agent_done_removed.count(True) == env.get_num_agents():
                break

        T_rewards.append(reward_sum)

        # At the end of the episode TODO This part could be done in another script
        # Metrics
        num_agents_done = 0  # Num of agents that reached their target
        num_agents_in_deadlock = 0
        num_agents_in_deadlock_at_switch = 0
        for a in range(env.get_num_agents()):
            if agent_done_removed[a]:
                num_agents_done += 1
            if env.obs_builder.is_agent_in_deadlock(a):
                num_agents_in_deadlock += 1
                if env.obs_builder.get_track(env.agents[a].position) == -2:
                    num_agents_in_deadlock_at_switch += 1

        # In proportion to total
        T_num_done_agents.append(num_agents_done / env.get_num_agents())
        T_all_done.append(all_done)
        T_agents_deadlock.append(num_agents_in_deadlock / env.get_num_agents())
        # Print training results info
        episode_stats = '\rEp: {}\t {} Agents on ({},{}).\t Ep score {:.3f}\t Done Agents in ep: {:.2f}%\t In deadlock {:.2f}%(at switch {})\t Not started {}\tEP ended at step: {}/{}\n'.format(
            ep,
            env.get_num_agents(), args.width, args.height,
            score,
            100 * (num_agents_done/args.num_agents),
            100 * (num_agents_in_deadlock/args.num_agents),
            (num_agents_in_deadlock_at_switch),
            num_agents_not_started,
            step+1,
            max_steps)
        print(episode_stats)

    # Average number of agents that reached their target
    avg_done_agents = sum(T_num_done_agents) / len(T_num_done_agents)
    avg_reward = sum(T_rewards) / len(T_rewards)
    avg_norm_reward = avg_reward / (max_steps / env.get_num_agents())
    avg_deadlock_agents = sum(T_agents_deadlock) / len(T_agents_deadlock)

    # avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
    if not evaluate:
        # Save model parameters if improved
        if avg_reward > metrics['best_avg_reward']:
            metrics['best_avg_reward'] = avg_reward
            dqn_agent.save(args.model_path + "best_model_{}_agents_on_{}_{}".format(args.num_agents, args.width, args.height))

        # Append to results and save metrics
        metrics['rewards'].append(T_rewards)
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        # Plot HTML
        """
        _plot_line(metrics['episodes'], metrics['rewards'],
                   'Reward', path=results_dir)  # Plot rewards in episodes
        """
    # Return average number of done agents (in proportion) and average reward
    return avg_done_agents, avg_reward, avg_norm_reward, avg_deadlock_agents, np.array(all_actions, dtype=int)


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

    ys = torch.tensor(ys_population, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(
        1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(
        color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(
        color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty',
                         fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(
        color=transparent), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(
        color=max_colour, dash='dash'), name='Min')

  
