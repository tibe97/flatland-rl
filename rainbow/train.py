from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from dueling_double_dqn import Agent
from graph_for_observation import GraphObservation
from test import test
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
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import logging
import wandb

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
    
    # initialize tensorboard 
    tb = SummaryWriter(args.model_path + 'runs/{}_{}_agents_on_{}_{}_start_epoch_{}'.format(args.tb_title, args.num_agents, args.width, args.height, args.start_epoch))
    tb_path = "agents_{}_on_{}_{}_start_{}_LR_{}".format(args.num_agents, args.width, args.height, args.start_epoch, args.learning_rate)

    wandb.init(project="Flatland-V11/{}_agents_on_({}, {})".format(args.num_agents, args.width, args.height), config=args)


    # ADAPTIVE parameters according to official configurations of tests 
    max_num_cities_adaptive = (args.num_agents//10)+2
    max_steps = int(4 * 2 * (args.width + args.height + args.num_agents / max_num_cities_adaptive))

    start_print = "About to train {} agents on ({},{}) env.\nParameters:\nmax_num_cities: {}\nmax_rails_between_cities: {}\nmax_rails_in_city: {}\nmalfunction_rate: {}\nmax_duration: {}\nmin_duration: {}\nnum_episodes: {}\nstarting from episode: {}\nmax_steps: {}\neps_initial: {}\neps_decay_rate: {}\nlearning_rate: {}\nlearning_rate_decay: {}\ndone_reward: {}\ndeadlock_reward: {}\n".format(
        args.num_agents,
        args.width,
        args.height,
        max_num_cities_adaptive,
        args.max_rails_between_cities,
        args.max_rails_in_city,
        args.malfunction_rate,
        args.max_duration,
        args.min_duration,
        args.num_episodes,
        args.start_epoch,
        max_steps,
        args.eps,
        args.eps_decay,
        args.learning_rate,
        args.learning_rate_decay,
        args.done_reward,
        args.deadlock_reward
    )
    print(start_print)
    with open(args.model_path + 'training_stats.txt', 'a') as f:
        print(start_print, file=f, end=" ")

    metrics = {'episodes': [], # originally 'steps'
			   'rewards': [],
			   'best_avg_done_agents': -float('inf'),
			   'best_avg_reward': -float('inf')}

    
    rail_generator = sparse_rail_generator(max_num_cities=max_num_cities_adaptive,
                                           seed=args.seed,
                                           grid_mode=args.grid_mode,
                                           max_rails_between_cities=args.max_rails_between_cities,
                                           max_rails_in_city=args.max_rails_in_city,
                                           )
    """ 
    Ignored at flatland 2020 edition. Only speed=1 is used

    # Maps speeds to % of appearance in the env
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train
    """
    speed_ration_map = {1.0: 1.0}

    schedule_generator = sparse_schedule_generator(speed_ration_map)

    stochastic_data = MalfunctionParameters(
        malfunction_rate=args.malfunction_rate,  # Rate of malfunction occurrence
        min_duration=args.min_duration,  # Minimal duration of malfunction
        max_duration=args.max_duration)  # Max duration of malfunction

    if args.observation_builder == 'GraphObsForRailEnv':
        observation_builder = GraphObservation() # custom observation
        state_size = 12
        dqn_agent = Agent(args=args, state_size=state_size, obs_builder=observation_builder, summary_writer=tb)

    wandb.watch((dqn_agent.qnetwork_action, dqn_agent.qnetwork_value_local), log='all')
    # LR scheduler to reduce learning rate over epochs
    lr_scheduler = StepLR(dqn_agent.optimizer_value, step_size=25, gamma=args.learning_rate_decay)
    #lr_scheduler = ReduceLROnPlateau(dqn_agent.optimizer_value, mode='min', factor=args.learning_rate_decay, patience=0, verbose=True, eps=1e-25)
    lr_scheduler_policy = StepLR(dqn_agent.optimizer_action, step_size=25, gamma=args.learning_rate_decay_policy)
    
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
    eps = args.eps
    eps_end = 0.005
    eps_decay = args.eps_decay
    railenv_action_dict = dict()
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    scores = []
   
    # Load previous weight is available
    if args.resume_weights:
        dqn_agent.load(args.model_path + args.model_name)
    
    # load previous saved experience memory if available
    if args.load_memory:
        dqn_agent.memory.load_memory(args.model_path + "replay_buffer")

    # Plot initial weights to see difference after 100 epochs
    '''
    tb.add_histogram("value.conv1.weight", dqn_agent.qnetwork_value_local.conv1.mlp[0].weight, args.start_epoch)
    tb.add_histogram("value.conv1.bias", dqn_agent.qnetwork_value_local.conv1.mlp[0].bias, args.start_epoch)
    tb.add_histogram("value.conv2.weight", dqn_agent.qnetwork_value_local.conv2.mlp[0].weight, args.start_epoch)
    tb.add_histogram("value.conv2.bias", dqn_agent.qnetwork_value_local.conv2.mlp[0].bias, args.start_epoch)
    tb.add_histogram("value.conv3.weight", dqn_agent.qnetwork_value_local.conv3.mlp[0].weight, args.start_epoch)
    tb.add_histogram("value.conv3.bias", dqn_agent.qnetwork_value_local.conv3.mlp[0].bias, args.start_epoch)
    tb.add_histogram("value.linear1.weight", dqn_agent.qnetwork_value_local.linear1.weight, args.start_epoch)
    tb.add_histogram("value.linear1.bias", dqn_agent.qnetwork_value_local.linear1.bias, args.start_epoch)
    tb.add_histogram("value.out.weight", dqn_agent.qnetwork_value_local.out.weight, args.start_epoch)
    tb.add_histogram("value.out.bias", dqn_agent.qnetwork_value_local.out.bias, args.start_epoch)

    tb.add_histogram("action.conv1.weight", dqn_agent.qnetwork_action.conv1.mlp[0].weight, args.start_epoch)
    tb.add_histogram("action.conv1.bias", dqn_agent.qnetwork_action.conv1.mlp[0].bias, args.start_epoch)
    tb.add_histogram("action.conv2.weight", dqn_agent.qnetwork_action.conv2.mlp[0].weight, args.start_epoch)
    tb.add_histogram("action.conv2.bias", dqn_agent.qnetwork_action.conv2.mlp[0].bias, args.start_epoch)
    tb.add_histogram("action.conv3.weight", dqn_agent.qnetwork_action.conv3.mlp[0].weight, args.start_epoch)
    tb.add_histogram("action.conv3.bias", dqn_agent.qnetwork_action.conv3.mlp[0].bias, args.start_epoch)
    tb.add_histogram("action.linear1.weight", dqn_agent.qnetwork_action.linear1.weight, args.start_epoch)
    tb.add_histogram("action.linear1.bias", dqn_agent.qnetwork_action.linear1.bias, args.start_epoch)
    tb.add_histogram("action.out.weight", dqn_agent.qnetwork_action.out.weight, args.start_epoch)
    tb.add_histogram("action.out.bias", dqn_agent.qnetwork_action.out.bias, args.start_epoch)
    tb.close()
    '''

    
    for ep in range(1+args.start_epoch, args.num_episodes + args.start_epoch + 1):
        
        logging.debug("Episode {} of {}".format(ep, args.num_episodes))
        obs, info = env.reset()
        path_values_buffer = [] # to compute mean path value for debugging
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
        epoch_loss = []
        rewards_buffer = [list()] * env.get_num_agents()
        log_probs_buffer = [list()] * env.get_num_agents()
        agent_ending_timestep = [max_steps] * env.get_num_agents()
        num_agents_at_switch = 0 # number of agents at switches
 

        for a in range(env.get_num_agents()):
            agent = env.agents[a]
            if env.obs_builder.get_track(agent.initial_position) == -2:
                num_agents_at_switch += 1
            agent_obs[a] = obs[a].copy()
            agent_obs_buffer[a] = agent_obs[a].copy()
            # First action of all agents is STOP_MOVING, so they all enter the env at the same time
            # TODO: decide the order of agents entering the env?
            action = RailEnvActions.STOP_MOVING  # TODO
            railenv_action_dict.update({a: action})
            agent_old_speed_data.update({a: agent.speed_data})
            env.obs_builder.agent_requires_obs.update({a: True})

        # env step returns next observations, rewards
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

            num_active_agents = 0
            num_agents_not_started = 0
            num_agents_done = 0
            agents_with_malfunctions = 0

            # for each agent
            for a in range(env.get_num_agents()):
                agent = env.agents[a]

                # Compute some stats to show
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

                if agent.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                    logging.debug("Agent {} is done".format(a))
                else:
                    logging.debug("Agent {} at position {}, fraction {}, status {}".format(
                        a, agent.position, agent.speed_data["position_fraction"], agent.status))

                ''' 
                    If agent is arriving at switch we need to compute the next path to reach in advance.
                    The agent computes a sequence of actions because the switch could be composed of more cells.
                    Each action (e.g. MOVE_FORWARD) could take more than 1 timestep if speed is not 1.
                    Each action has a fixed TIME_STEPS value for each agent, computed as 1/agent_speed.
                    At each environment step we subtract 1 from the remaining timestep for a certain action of an agent.
                    When we finish all the actions we have reached the new node, here the only allowed action is MOVE_FORWARD
                    until we reach the next switch.
                '''
                # If the agent arrives at a switch
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
                            log_probs_buffer[a].append(path_values[a][2])
                            railenv_action = env.obs_builder.choose_railenv_actions(a, path_values[a])
                            agent_action_buffer[a] = railenv_action
                            # as state to save we take the path (with its children) chosen by agent
                            agent_path_obs_buffer[a] = agent_obs[a]["partitioned"][path_values[a][0]]
                            logging.debug("Agent {} choses path {} with value {} at position {}. Num actions to take: {}".format(
                                a, path_values[a][0][0], path_values[a][3], agent.position, len(agent_action_buffer[a])))
                            path_values_buffer.append(path_values[a][1]) # for debug 
                            tb.add_histogram("Path values", path_values[a][3].item(), (ep//100)*100 + 100)
                            logging.debug(
                                "Agent {} actions: {}".format(a, railenv_action))
                        next_action = agent_action_buffer[a].pop(0)
                        tb.add_histogram("Actions train ({} agents)".format(args.num_agents), int(next_action), (ep//100)*100 + 100)
                        logging.debug("Agent {} at: {}. Action is: {}. Speed: {}. Fraction {}. Remaining actions: {}. SpeedTimesteps: {}".format(
                            a, agent.position, next_action, agent.speed_data["speed"], agent.speed_data["position_fraction"], len(agent_action_buffer[a]), agents_speed_timesteps[a]))
                        # if agent has to stop, do it for 1 timestep
                        if (next_action == RailEnvActions.STOP_MOVING):
                            agents_speed_timesteps[a] = 1
                            env.obs_builder.agent_requires_obs.update(
                                {a: True})
                        else:
                            # speed is a fractionary value between 0 and 1
                            agents_speed_timesteps[a] = int(round(1 / info["speed"][a]))
                # if agent is not at switch just go straight
                elif agent.status != RailAgentStatus.DONE_REMOVED:  
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

            logging.debug("Active agents: {}. Not started yet: {}. With malfunctions: {}".format(
                num_active_agents, num_agents_not_started, agents_with_malfunctions))
            logging.debug("-------- STEP DONE -------------")

            # Update replay buffer and train agent
            for a in range(env.get_num_agents()):
                agent = env.agents[a]
                if not agent_done_removed[a]:
                    logging.debug("Agent {} at position {}, fraction {}, speed Timesteps {}, reward {}".format(
                        a, agent.position, agent.speed_data["position_fraction"], agents_speed_timesteps[a], acc_rewards[a]))

                score += all_rewards[a] / env.get_num_agents()  # Update score

                # if agent didn't move do nothing: agent couldn't perform action because another agent
                # occupied next cell or agent's action was STOP
                if env.obs_builder.agent_could_move(a, railenv_action_dict[a], agent_old_speed_data[a]):
                    # update replay memory
                    acc_rewards[a] += all_rewards[a]
                    if ((update_values[a] and agent.speed_data["position_fraction"] == 0) or agent.status == RailAgentStatus.DONE_REMOVED) and not agent_done_removed[a]:
                        logging.debug("Update=True: agent {}".format(a))
                        # next state is the complete state, with all the possible path choices
                        if len(next_obs[a]) > 0 and agent_path_obs_buffer[a] is not None:
                            # if agent reaches target
                            if agent.status == RailAgentStatus.DONE_REMOVED or agent.status == RailAgentStatus.DONE:
                                agent_done_removed[a] = True
                                acc_rewards[a] = args.done_reward
                                agent_ending_timestep[a] = step
                                #acc_rewards[a] += args.done_reward # Explicit individual reward of reaching target
                                logging.debug("Agent {} DONE! It has been removed and experience saved with reward of {}!".format(a, acc_rewards[a]))
                            else: 
                                logging.debug("Agent reward is {}".format(acc_rewards[a]))
                            # step saves experience tuple and can perform learning (every T time steps)
                            step_loss = dqn_agent.step(agent_path_obs_buffer[a], acc_rewards[a], next_obs[a], agent_done_removed[a], agents_in_deadlock[a], ep=ep)
                            
                            # save stats
                            if step_loss is not None:
                                epoch_loss.append(step_loss)
                            if agent_done_removed[a]:
                                rewards_buffer[a].append(0)
                            else:
                                rewards_buffer[a].append(acc_rewards[a])
                            acc_rewards[a] = 0
                            update_values[a] = False
                            
                            
                    if len(next_obs[a]) > 0:
                        # prepare agent obs for next timestep
                        agent_obs[a] = next_obs[a].copy()

                    if agent_at_switch[a]:
                        # we descrease timestep if agent is performing actions at switch
                        agents_speed_timesteps[a] -= 1
                        
                    """
                        We want to optimize computation of observations only when it's needed, i.e. before 
                        making a decision.
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
                        if not agent_at_switch[a]:
                            agent_pos = agent.position
                            assert env.obs_builder.get_track(agent_pos) != -2
                            if env.obs_builder.is_agent_entering_switch(a) and agent.speed_data["position_fraction"] == 0:
                                logging.debug("Agent {} arrived at 1 cell before switch".format(a))
                                agent_at_switch[a] = True
                                agents_speed_timesteps[a] = 0
                                # env.obs_builder.agent_requires_obs.update({a: False})
                            elif env.obs_builder.is_agent_2_steps_from_switch(a):
                                env.obs_builder.agent_requires_obs.update({a: True})
                                update_values[a] = True
                            if env.obs_builder.is_agent_about_to_finish(a):
                                env.obs_builder.agent_requires_obs.update({a: True})
                        else:  # Agent at SWITCH. In the step before reaching target path we want to make sure to compute the obs
                            # in order to update the replay memory. We need to be careful if the agent can't reach new path because of another agent blocking the cell.
                            # when agent speed is 1 we reach the target in 1 step
                            if len(agent_action_buffer[a]) == 1 and agent.speed_data["speed"] == 1:
                                # agent_next_action = agent_action_buffer[a][0]
                                # assert env.obs_builder.is_agent_exiting_switch(a, agent_next_action)
                                #update_values[a] = True
                                env.obs_builder.agent_requires_obs.update({a: True})

                            # if speed is less than 1, we need more steps to reach target. So only compute obs if doing last step
                            elif len(agent_action_buffer[a]) == 0:
                                if env.obs_builder.get_track(agent.position) == -2 and agent.speed_data["speed"] < 1 and np.isclose(agent.speed_data["speed"] + agent.speed_data["position_fraction"], 1, rtol=1e-03):
                                    # same check as "if" condition
                                    assert agents_speed_timesteps[a] > 0
                                    #update_values[a] = True
                                    env.obs_builder.agent_requires_obs.update({a: True})
                                else:
                                    if env.obs_builder.get_track(agent.position) != -2:
                                        if env.obs_builder.is_agent_entering_switch(a):
                                            assert len(next_obs[a]) > 0
                                            logging.debug("Agent {} just exited switch and ALREADY entering another one".format(a))
                                            agent_obs_buffer[a] = next_obs[a].copy()
                                            update_values[a] = True
                                        else:
                                            logging.debug("Agent {} is not at switch anymore".format(a))
                                            agent_at_switch[a] = False
                                            agents_speed_timesteps[a] = 0
                                            agent_obs_buffer[a] = next_obs[a].copy()
                                        if env.obs_builder.is_agent_about_to_finish(a):
                                            env.obs_builder.agent_requires_obs.update(
                                                {a: True})

                else:  # agent did not move. Check if it stopped on purpose
                    # acc_rewards[a] += all_rewards[a]
                    if railenv_action_dict[a] == RailEnvActions.STOP_MOVING:
                        agents_speed_timesteps[a] -= 1
                        env.obs_builder.agent_requires_obs.update({a: True})
                      
                    else:
                        logging.debug("Agent {} cannot move at position {}, fraction {}".format(
                            a, agent.position, agent.speed_data["position_fraction"]))
                        if env.obs_builder.is_agent_in_deadlock(a) and not agents_in_deadlock[a]:
                            env.obs_builder.agent_requires_obs.update({a: True})
                            logging.debug("Agent {} in DEADLOCK saved as experience with reward of {}".format(
                                a, acc_rewards[a]))
                            if len(next_obs[a]) > 0 and agent_path_obs_buffer[a] is not None:
                                agent_obs_buffer[a] = next_obs[a]
                                acc_rewards[a] = args.deadlock_reward
                                agents_in_deadlock[a] = True
                                step_loss = dqn_agent.step(agent_path_obs_buffer[a], acc_rewards[a], agent_obs_buffer[a], done[a], agents_in_deadlock[a], ep=ep)
                                if step_loss is not None:
                                    epoch_loss.append(step_loss)
                                env.obs_builder.agent_requires_obs.update({a: False})
                            logging.debug("Agent {} is in DEADLOCK, accum. reward: {}, required_obs: {}".format(a, acc_rewards[a], env.obs_builder.agent_requires_obs[a]))

                agent_old_speed_data.update({a: agent.speed_data.copy()})
            
            
            if agent_done_removed.count(True) == env.get_num_agents(): 
                break  

        # Learn action STOP/GO only at the end of episode
        dqn_agent.learn_actions(log_probs_buffer, agent_ending_timestep, agent_done_removed, max_steps, ep)

        # end of episode
        eps = max(eps_end, eps_decay * eps)  # Decrease epsilon
        # Metrics
        num_agents_done = 0  # Num of agents that reached their target
        num_agents_in_deadlock = 0
        num_agents_not_started = 0
        num_agents_in_deadlock_at_switch = 0
        for a in range(env.get_num_agents()):
            if env.agents[a].status in [RailAgentStatus.DONE_REMOVED, RailAgentStatus.DONE]:
                num_agents_done += 1
            elif env.agents[a].status == RailAgentStatus.READY_TO_DEPART:
                num_agents_not_started += 1
            elif env.obs_builder.is_agent_in_deadlock(a):
                num_agents_in_deadlock += 1
                if env.obs_builder.get_track(env.agents[a].position) == -2:
                    num_agents_in_deadlock_at_switch += 1
        if num_agents_done == env.get_num_agents():
            env_done = 1

        scores_window.append(score / max_steps)  # Save most recent score
        scores.append(np.mean(scores_window))
        done_window.append(env_done)
        if len(epoch_loss) > 0:
            epoch_mean_loss = (sum(epoch_loss)/(len(epoch_loss)))
        else:
            epoch_mean_loss = None
        


        # Print training results info
        episode_stats = '\rEp: {}\t {} Agents on ({},{}).\t Ep score {:.3f}\tAvg Score: {:.3f}\t Env Dones so far: {:.2f}%\t Done Agents in ep: {:.2f}%\t In deadlock {:.2f}%(at switch {})\n\t\t Not started {}\t Eps: {:.2f}\tEP ended at step: {}/{}\tMean state_value: {}\t Epoch avg_loss: {}\n'.format(
            ep,
            env.get_num_agents(), args.width, args.height,
            score,
            np.mean(scores_window),
            100 * np.mean(done_window),
            100 * (num_agents_done/args.num_agents),
            100 * (num_agents_in_deadlock/args.num_agents),
            (num_agents_in_deadlock_at_switch),
            num_agents_not_started,
            eps,
            step+1,
            max_steps,
            (sum(path_values_buffer)/len(path_values_buffer)),
            epoch_mean_loss)
        if epoch_mean_loss is not None:
            tb.add_scalar("Loss", epoch_mean_loss, ep)
            tb.close()
            wandb.log({"mean_loss": epoch_mean_loss})

        print(episode_stats, end=" ")
        '''
        with open(args.model_path + 'training_stats.txt', 'a') as f:
            print(episode_stats, file=f, end=" ")
        '''
        if (ep % args.evaluation_interval) == 0:  # Evaluate only at the end of the episodes

            dqn_agent.eval()  # Set DQN (online network) to evaluation mode
            avg_done_agents, avg_reward, avg_norm_reward, avg_deadlock_agents, test_actions = test(
                args, ep, dqn_agent, metrics, args.model_path)  # Test
            
            testing_stats = '\nEpoch ' + str(ep) + ', testing agents on ' + str(args.evaluation_episodes) + ': Avg. done agents: ' + str(avg_done_agents*100) + '% | Avg. reward: ' + str(avg_reward) + ' | Avg. normalized reward: ' + str(avg_norm_reward) + ' | Avg. agents in deadlock: ' + str(avg_deadlock_agents*100) + '%' + '| LR: ' + str(dqn_agent.optimizer_value.param_groups[0]['lr']) + "\n"
            print(testing_stats)
            with open(args.model_path + 'training_stats.txt', 'a') as f:
                print(testing_stats, file=f)
            dqn_agent.train()  # Set DQN (online network) back to training mode
            # Save replay buffer
            dqn_agent.memory.save_memory(args.model_path + "replay_buffer")

            wandb.log({"avg_reward": avg_reward, "done_agents": avg_done_agents, "deadlock_agents": avg_deadlock_agents})

            '''
            tb.add_scalar("Avg reward ({} agents)".format(args.num_agents), avg_reward, ep)
            tb.add_scalar("Done Agents ({} agents)".format(args.num_agents), avg_done_agents, ep)
            tb.add_scalar("Deadlock Agents ({} agents)".format(args.num_agents), avg_deadlock_agents, ep)

            tb.add_histogram("value.conv1.weight", dqn_agent.qnetwork_value_local.conv1.mlp[0].weight, ep)
            tb.add_histogram("value.conv1.bias", dqn_agent.qnetwork_value_local.conv1.mlp[0].bias, ep)
            tb.add_histogram("value.conv2.weight", dqn_agent.qnetwork_value_local.conv2.mlp[0].weight, ep)
            tb.add_histogram("value.conv2.bias", dqn_agent.qnetwork_value_local.conv2.mlp[0].bias, ep)
            tb.add_histogram("value.conv3.weight", dqn_agent.qnetwork_value_local.conv3.mlp[0].weight, ep)
            tb.add_histogram("value.conv3.bias", dqn_agent.qnetwork_value_local.conv3.mlp[0].bias, ep)
            tb.add_histogram("value.linear1.weight", dqn_agent.qnetwork_value_local.linear1.weight, ep)
            tb.add_histogram("value.linear1.bias", dqn_agent.qnetwork_value_local.linear1.bias, ep)
            tb.add_histogram("value.out.weight", dqn_agent.qnetwork_value_local.out.weight, ep)
            tb.add_histogram("value.out.bias", dqn_agent.qnetwork_value_local.out.bias, ep)

            tb.add_histogram("action.conv1.weight", dqn_agent.qnetwork_action.conv1.mlp[0].weight, ep)
            tb.add_histogram("action.conv1.bias", dqn_agent.qnetwork_action.conv1.mlp[0].bias, ep)
            tb.add_histogram("action.conv2.weight", dqn_agent.qnetwork_action.conv2.mlp[0].weight, ep)
            tb.add_histogram("action.conv2.bias", dqn_agent.qnetwork_action.conv2.mlp[0].bias, ep)
            tb.add_histogram("action.conv3.weight", dqn_agent.qnetwork_action.conv3.mlp[0].weight, ep)
            tb.add_histogram("action.conv3.bias", dqn_agent.qnetwork_action.conv3.mlp[0].bias, ep)
            tb.add_histogram("action.linear1.weight", dqn_agent.qnetwork_action.linear1.weight, ep)
            tb.add_histogram("action.linear1.bias", dqn_agent.qnetwork_action.linear1.bias, ep)
            tb.add_histogram("action.out.weight", dqn_agent.qnetwork_action.out.weight, ep)
            tb.add_histogram("action.out.bias", dqn_agent.qnetwork_action.out.bias, ep)

            tb.add_histogram("Actions test ({} agents)".format(args.num_agents), test_actions, ep)
            tb.close()
            '''
            
            '''
            # reduce LR based on reward
            if ep >= args.start_lr_decay:
                lr_scheduler.step(-avg_reward)
            '''
            
            # Plot Learning rate in tensorboard
            tb.add_scalar("Learning rate value", dqn_agent.optimizer_value.param_groups[0]['lr'], ep)
            tb.add_scalar("Learning rate action", dqn_agent.optimizer_action.param_groups[0]['lr'], ep)

            # Adaptive Epsilon Greedy 
            #eps = 1 - avg_done_agents
        '''
        if ep % args.save_model_interval == 0:
            dqn_agent.save(args.model_path + args.model_name)  # Save models
        '''   

        if ep % (args.save_model_interval) == 0:  #backup weights
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y__%H_%M_")
            checkpoint_path = args.model_path + "checkpoint_" + str(args.num_agents) + "_agents_on_" + str(args.width) + "_" + str(args.height)
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            dqn_agent.save(checkpoint_path + "/" + "epoch_" + str(ep) + "_" + dt_string + args.model_name)

        lr_scheduler.step()
        lr_scheduler_policy.step()
        '''
        if epoch_mean_loss is not None:
            #lr_scheduler.step()
            lr_scheduler.step(epoch_mean_loss)
        '''
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Flatland')
    # Flatland parameters
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
    # Training parameters
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Number of episodes on which to train the agents')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Epoch from which resume training (useful for stats)')
    parser.add_argument('--max-steps', type=int, default=300,
                        help='Maximum number of steps for each episode')
    parser.add_argument('--eps', type=float, default=1,
                        help='epsilon value for e-greedy')
    parser.add_argument('--tb-title', type=str, default="no_title",
                        help='title for tensorboard run')
    parser.add_argument('--eps-decay', type=float, default=0.999,
                        help='epsilon decay value')
    parser.add_argument('--learning-rate', type=float, default=0.005,
                        help='LR for DQN agent')
    parser.add_argument('--learning-rate-decay', type=float, default=0.5,
                        help='LR decay for DQN agent')
    parser.add_argument('--learning-rate-decay-policy', type=float, default=0.5,
                        help='LR decay for policy network')
    parser.add_argument('--model-path', type=str, default='', help="results/")
    parser.add_argument('--model-name', type=str, default='weights/best_model_8_agents_on_25_25',
                        help='Name to use to save the model .pth')
    parser.add_argument('--resume-weights', type=bool, default=True,
                        help='True if load previous weights')
    parser.add_argument('--debug-print', type=bool, default=False,
                        help='requires debug printing')
    parser.add_argument('--load-memory', type=bool, default=True,
                        help='if load saved memory')
    parser.add_argument('--evaluation-episodes', type=int, default=15,
                        metavar='N', help='Number of evaluation episodes to average over')
    parser.add_argument('--render', action='store_true',
                        default=False, help='Display screen (testing only)')
    parser.add_argument('--evaluation-interval', type=int, default=50, metavar='EPISODES', help='Number of episodes between evaluations')
    parser.add_argument('--save-model-interval', type=int, default=50,
                        help='Save models every tot episodes')
    parser.add_argument('--start-lr-decay', type=int, default=150,
                        help='Save models every tot episodes')
    parser.add_argument('--done-reward', type=int, default=0,
                        help='Reward given to agent when it reaches target')
    parser.add_argument('--deadlock-reward', type=int, default=-1000,
                        help='Reward given to agent when it reaches deadlock')
    parser.add_argument('--reward-scaling', type=float, default=0.1,
                        help='Reward scaling factor')

    

    args = parser.parse_args()
    # Check arguments
    if args.offset > args.height:
        raise ValueError(
            "Agent offset can't be greater than view height in local obs")
    if args.offset < 0:
        raise ValueError("Agent offset must be a positive integer")

    main(args)
