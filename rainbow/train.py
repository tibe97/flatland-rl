from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from dueling_double_dqn import Agent
from graph_for_observation import GraphObservation, EpisodeController
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
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, CyclicLR
from torch.utils.tensorboard import SummaryWriter
import logging
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    
    # initialize Weight and Biases for logging results
    
    # turn wandb off when only testing the code correctness
    wandb.init(project="Flatland-{}".format(args.wandb_project_name), name= "{}_{}_agents_on_({}, {})_{}".format(args.run_title, args.num_agents, args.width, args.height, datetime.now().strftime("%d/%m/%Y %H:%M:%S")), config=args)
    #wandb.init(mode='disabled')

    # ADAPTIVE parameters according to official configurations of tests 
    max_num_cities_adaptive = (args.num_agents//10)+2
    max_steps = int(4 * 2 * (args.width + args.height + args.num_agents / max_num_cities_adaptive))

    start_print = "About to train {} agents on ({},{}) env.\nParameters:\nmax_num_cities: {}\nmax_rails_between_cities: {}\nmax_rails_in_city: {}\nmalfunction_rate: {}\nmax_duration: {}\nmin_duration: {}\nnum_episodes: {}\nstarting from episode: {}\nmax_steps: {}\neps_initial: {}\neps_decay_rate: {}\nlearning_rate: {}\nlearning_rate_decay: {}\ndone_reward: {}\ndeadlock_reward: {}\nbatch_size: {}".format(
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
        args.deadlock_reward,
        args.batch_size
    )
    print(start_print)
    os.makedirs(args.model_path, exist_ok=True) 
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
        observation_builder = GraphObservation(args.observation_depth) # custom observation
        state_size = 12
        rl_agent = Agent(args=args, state_size=state_size, obs_builder=observation_builder)

    wandb.watch(rl_agent.qnetwork_value_local, log_freq=1, log='all')
    #wandb.watch(rl_agent.qnetwork_action, log_freq=1, log='all')

    params = list(rl_agent.qnetwork_value_local.named_parameters())
    for p in params:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    # LR scheduler to reduce learning rate over epochs
    #lr_scheduler = ReduceLROnPlateau(rl_agent.optimizer_value, mode='min', factor=args.learning_rate_decay, patience=0, verbose=True, eps=1e-25)
    #lr_scheduler = StepLR(rl_agent.optimizer_value, step_size=25, gamma=args.learning_rate_decay)
    #lr_scheduler_policy = StepLR(rl_agent.optimizer_action, step_size=25, gamma=args.learning_rate_decay_policy)
    
    #lr_scheduler = CosineAnnealingLR(rl_agent.optimizer_value, T_max = 200)
    #lr_scheduler_policy = CosineAnnealingLR(rl_agent.optimizer_action, T_max=200)
    
    lr_scheduler = CyclicLR(rl_agent.optimizer_value, base_lr=0.00, max_lr=args.learning_rate, step_size_up=20, cycle_momentum=False, mode="triangular2")
    lr_scheduler_policy = CyclicLR(rl_agent.optimizer_action, base_lr=0.00, max_lr=args.learning_rate, step_size_up=20, cycle_momentum=False, mode="triangular2")
    
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

    # create multiple envs with different number of agents
    envs = [RailEnv(width=args.width,
                    height=args.height,
                    rail_generator=sparse_rail_generator(max_num_cities=(num_agents//10)+2,
                                           seed=args.seed,
                                           grid_mode=args.grid_mode,
                                           max_rails_between_cities=args.max_rails_between_cities,
                                           max_rails_in_city=args.max_rails_in_city,
                                           ),
                    schedule_generator=schedule_generator,
                    number_of_agents=num_agents,
                    obs_builder_object=observation_builder,
                    malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                    remove_agents_at_target=True) 
            for num_agents in [1,2,3,4,5,6,7,8]]
    env.reset()
    

    # max_steps = env.compute_max_episode_steps(args.width, args.height, args.num_agents/args.max_num_cities)
    eps = args.eps
    eps_end = 0.005
    eps_decay = args.eps_decay
    railenv_action_dict = dict()
 
   
    # Load previous weight is available
    if args.resume_weights:
        rl_agent.load(args.model_path + args.model_name)
    
    # load previous saved experience memory if available
    if args.load_memory:
        rl_agent.memory.load_memory(args.model_path + "replay_buffer")
    
    #ep_controller = EpisodeController(env, rl_agent, max_steps)

    for ep in range(1+args.start_epoch, args.num_episodes + args.start_epoch + 1):
        env = envs[ep%8]
        
        logging.debug("Episode {} of {}".format(ep, args.num_episodes))
        ep_controller = EpisodeController(env, rl_agent, max_steps)
        ep_controller.reset()
       
        obs, info = env.reset()
        ep_controller.reset()
        max_num_cities_adaptive = (env.get_num_agents()//10)+2
        max_steps = int(4 * 2 * (args.width + args.height + env.get_num_agents() / max_num_cities_adaptive))
        
        # first action
        for a in range(env.get_num_agents()):
            agent = env.agents[a]
            ep_controller.agent_obs[a] = obs[a].copy()

            #agent_action = ep_controller.compute_agent_action(a, info, eps)
            agent_action = RailEnvActions.STOP_MOVING  # TODO
            railenv_action_dict.update({a: agent_action})
            ep_controller.agent_old_speed_data.update({a: agent.speed_data})
            env.obs_builder.agent_requires_obs.update({a: True})
           

        # env step returns next observations, rewards
        next_obs, all_rewards, done, info = env.step(railenv_action_dict)


        # MULTI AGENT
        # initialize actions for all agents in this episode
        num_agents = env.get_num_agents()
        actions = torch.randint(0, 1, size=(num_agents,)) # 0, start from stop

      
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

            # MULTI AGENT
            states = [env.obs_builder.preprocess_agent_obs(ep_controller.agent_obs[i], i) for i in range(num_agents)]
            
            # step together
            def infer_acts(states, actions, num_iter=3):
                N = actions.shape[0]
                actions_ = actions.clone()               
                joint_actions = torch.zeros(N, 2**7).to(device)
                q_values = torch.zeros(N).to(device)

                # calculating distance matrix of all agents
                positions = [(0,0) for _ in range(num_agents)]
                distance_matrix = [[0] * num_agents for _ in range(num_agents)]
                for i in range(num_agents):
                    agent = env.agents[i]
                    positions[i] = agent.position if agent.position != None else agent.initial_position
                #print(positions)
                for i in range(num_agents):
                    for j in range(i+1, num_agents):
                        distance_matrix[i][j] = abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])
                        #distance_matrix[i][j] = math.sqrt( (positions[i][0] - positions[j][0]) ** 2 + (positions[i][1] - positions[j][1]) ** 2)
                        distance_matrix[j][i] = distance_matrix[i][j]
                #print(distance_matrix)
                
                for i in range(num_iter):
                    # using joint action of all other agents' actions
                    for j in range(N):
                        neighbors = np.argsort(distance_matrix[j])[1:] # all neighbor except for agent itself
                        neighbor_actions = actions_[neighbors]
                        neighbor_actions = torch.nn.functional.one_hot(neighbor_actions, num_classes=2)
                        complement = torch.tensor([1, 0])
                        
                        if N == 1:
                            joint_action = complement
                            for k in range(6):
                                joint_action = torch.outer(complement, joint_action).flatten()
                        else:
                            joint_action = neighbor_actions[0]   
                            # joint action of exsiting neighbors
                            for k in range(N-2): 
                                joint_action = torch.outer(neighbor_actions[k+1], joint_action).flatten()
                            # joint action of completments
                            for k in range(8-N): 
                                joint_action = torch.outer(complement, joint_action).flatten()
                            
                        joint_actions[j] = joint_action

                    for j in range(N):
                        state = states[j].to(device)
                        q_action = ep_controller.rl_agent.act(state, joint_actions[j], eps=eps)
                        q_values[j] = q_action[j][3]
                        actions_[j] = q_action[j][1]
                        
                return actions_, joint_actions, q_values
                 
            actions, joint_actions, q_values = infer_acts(states, actions)
            #print("#AGENTS: {}, ACTIONS: {}, MF: {} for current Q".format(num_agents, actions, joint_actions))
            
            # For each agent
            for a in range(env.get_num_agents()):
                agent = env.agents[a]
                agent_next_action = ep_controller.compute_agent_action(a, info, eps, joint_actions[a]) # here the action is 5-dim
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
                ep_controller.save_experience_and_train(a, railenv_action_dict[a], all_rewards[a], next_obs[a], done[a], step, args, ep, joint_actions[a], next_q_values[a])
                
            if ep_controller.is_episode_done():
                break  

        eps = max(eps_end, eps_decay * eps)  # Decrease epsilon

        # Learn action STOP/GO only at the end of episode
        # For now let's just use value network
        rl_agent.learn_actions(ep_controller.log_probs_buffer, ep_controller.agent_ending_timestep, ep_controller.agent_done_removed, max_steps, ep)

        
        # end of episode
        ep_controller.print_episode_stats(ep, args, eps, step)

        wandb_log_dict = ep_controller.retrieve_wandb_log(eps)
        
        if (ep % args.evaluation_interval) == 0:  # Evaluate only at the end of the episodes

            rl_agent.eval()  # Set DQN (online network) to evaluation mode
            avg_done_agents, avg_reward, avg_norm_reward, avg_deadlock_agents, test_actions = test(args, ep, rl_agent, metrics, args.model_path)  # Test
            
            testing_stats = '\nEpoch ' + str(ep) + ', testing agents on ' + str(args.evaluation_episodes) + ': Avg. done agents: ' + str(avg_done_agents*100) + '% | Avg. reward: ' + str(avg_reward) + ' | Avg. normalized reward: ' + str(avg_norm_reward) + ' | Avg. agents in deadlock: ' + str(avg_deadlock_agents*100) + '%' + '| LR: ' + str(rl_agent.optimizer_value.param_groups[0]['lr'])
            print(testing_stats)
            with open(args.model_path + 'training_stats.txt', 'a') as f:
                print(testing_stats, file=f)
            rl_agent.train()  # Set DQN (online network) back to training mode
            # Save replay buffer
            rl_agent.memory.save_memory(args.model_path + "replay_buffer")

            wandb_log_dict.update({"avg_reward": avg_reward, "done_agents": avg_done_agents, "deadlock_agents": avg_deadlock_agents})
            
           
        wandb.log(wandb_log_dict, step=ep)  
       
        if ep % (args.save_model_interval) == 0:  #backup weights
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y__%H_%M_")
            checkpoint_path = args.model_path + "checkpoint_" + str(args.num_agents) + "_agents_on_" + str(args.width) + "_" + str(args.height)
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            rl_agent.save(checkpoint_path + "/" + "epoch_" + str(ep) + "_" + dt_string + args.model_name)

        lr_scheduler.step()
        lr_scheduler_policy.step()
        
        
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
    parser.add_argument('--eps-decay', type=float, default=0.992,
                        help='epsilon decay value')
    parser.add_argument('--learning-rate', type=float, default=0.03,
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
    # Check arguments
    if args.offset > args.height:
        raise ValueError(
            "Agent offset can't be greater than view height in local obs")
    if args.offset < 0:
        raise ValueError("Agent offset must be a positive integer")

    main(args)
