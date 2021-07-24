import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.evaluators.client import FlatlandRemoteClient
from flatland.evaluators.client import TimeoutException

from reinforcement_learning.dddqn_policy import DDDQNPolicy
from reinforcement_learning.deadlockavoidance_with_decision_agent import DeadLockAvoidanceWithDecisionAgent
from reinforcement_learning.multi_decision_agent import MultiDecisionAgent
from reinforcement_learning.ppo_agent import PPOPolicy
from utils.agent_action_config import get_action_size, map_actions, set_action_size_reduced, set_action_size_full
from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent
from utils.deadlock_check import check_if_all_blocked
from utils.fast_tree_obs import FastTreeObs
from utils.observation_utils import normalize_observation

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

####################################################
# EVALUATION PARAMETERS
set_action_size_full()

# Print per-step logs
VERBOSE = True
USE_FAST_TREEOBS = True

if False:
    # -------------------------------------------------------------------------------------------------------
    # RL solution
    # -------------------------------------------------------------------------------------------------------
    # 116591 adrian_egli
    # graded	71.305	0.633	RL	Successfully Graded ! More details about this submission can be found at:
    # http://gitlab.aicrowd.com/adrian_egli/neurips2020-flatland-starter-kit/issues/51
    # Fri, 22 Jan 2021 23:37:56
    set_action_size_reduced()
    load_policy = "DDDQN"
    checkpoint = "./checkpoints/210122120236-3000.pth"  # 17.011131341978228
    EPSILON = 0.0

if False:
    # -------------------------------------------------------------------------------------------------------
    # RL solution
    # -------------------------------------------------------------------------------------------------------
    # 116658 adrian_egli
    # graded	73.821	0.655	RL	Successfully Graded ! More details about this submission can be found at:
    # http://gitlab.aicrowd.com/adrian_egli/neurips2020-flatland-starter-kit/issues/52
    # Sat, 23 Jan 2021 07:41:35
    set_action_size_reduced()
    load_policy = "PPO"
    checkpoint = "./checkpoints/210122235754-5000.pth"  # 16.00113400887389
    EPSILON = 0.0

if True:
    # -------------------------------------------------------------------------------------------------------
    # RL solution
    # -------------------------------------------------------------------------------------------------------
    # 116659 adrian_egli
    # graded	80.579	0.715	RL	Successfully Graded ! More details about this submission can be found at:
    # http://gitlab.aicrowd.com/adrian_egli/neurips2020-flatland-starter-kit/issues/53
    # Sat, 23 Jan 2021 07:45:49
    set_action_size_reduced()
    load_policy = "DDDQN"
    checkpoint = "./checkpoints/210122165109-5000.pth"  # 17.993750197899438
    EPSILON = 0.0

if False:
    # -------------------------------------------------------------------------------------------------------
    # !! This is not a RL solution !!!!
    # -------------------------------------------------------------------------------------------------------
    # 116727 adrian_egli
    # graded	106.786	0.768	RL	Successfully Graded ! More details about this submission can be found at:
    # http://gitlab.aicrowd.com/adrian_egli/neurips2020-flatland-starter-kit/issues/54
    # Sat, 23 Jan 2021 14:31:50
    set_action_size_reduced()
    load_policy = "DeadLockAvoidance"
    checkpoint = None
    EPSILON = 0.0

# Use last action cache
USE_ACTION_CACHE = False

# Observation parameters (must match training parameters!)
observation_tree_depth = 2
observation_radius = 10
observation_max_path_depth = 30

####################################################

remote_client = FlatlandRemoteClient()

# Observation builder
predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
if USE_FAST_TREEOBS:
    def check_is_observation_valid(observation):
        return True

    def get_normalized_observation(
            observation,
            tree_depth: int,
            observation_radius=0):
        return observation

    tree_observation = FastTreeObs(max_depth=observation_tree_depth)
    state_size = tree_observation.observation_dim
else:
    def check_is_observation_valid(observation):
        return observation

    def get_normalized_observation(
            observation,
            tree_depth: int,
            observation_radius=0):
        return normalize_observation(
            observation, tree_depth, observation_radius)

    tree_observation = TreeObsForRailEnv(
        max_depth=observation_tree_depth,
        predictor=predictor)
    # Calculate the state size given the depth of the tree observation and the
    # number of features
    n_features_per_node = tree_observation.observation_dim
    n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
    state_size = n_features_per_node * n_nodes

#####################################################################
# Main evaluation loop
#####################################################################
evaluation_number = 0

while True:
    evaluation_number += 1

    # We use a dummy observation and call TreeObsForRailEnv ourselves when needed.
    # This way we decide if we want to calculate the observations or not instead
    # of having them calculated every time we perform an env step.
    time_start = time.time()
    observation, info = remote_client.env_create(
        obs_builder_object=DummyObservationBuilder()
    )
    env_creation_time = time.time() - time_start

    if not observation:
        # If the remote_client returns False on a `env_create` call,
        # then it basically means that your agent has already been
        # evaluated on all the required evaluation environments,
        # and hence it's safe to break out of the main evaluation loop.
        break

    print("Env Path : ", remote_client.current_env_path)
    print("Env Creation Time : ", env_creation_time)

    local_env = remote_client.env
    nb_agents = len(local_env.agents)
    max_nb_steps = local_env._max_episode_steps

    tree_observation.set_env(local_env)
    tree_observation.reset()

    # Creates the policy. No GPU on evaluation server.
    if load_policy == "DDDQN":
        policy = DDDQNPolicy(state_size, get_action_size(), Namespace(
            **{'use_gpu': False}), evaluation_mode=True)
    elif load_policy == "PPO":
        policy = PPOPolicy(state_size, get_action_size())
    elif load_policy == "DeadLockAvoidance":
        policy = DeadLockAvoidanceAgent(
            local_env, get_action_size(), enable_eps=False)
    elif load_policy == "DeadLockAvoidanceWithDecision":
        # inter_policy = PPOPolicy(state_size, get_action_size(), use_replay_buffer=False, in_parameters=train_params)
        inter_policy = DDDQNPolicy(state_size, get_action_size(), Namespace(
            **{'use_gpu': False}), evaluation_mode=True)
        policy = DeadLockAvoidanceWithDecisionAgent(
            local_env, state_size, get_action_size(), inter_policy)
    elif load_policy == "MultiDecision":
        policy = MultiDecisionAgent(
            state_size, get_action_size(), Namespace(**{'use_gpu': False}))
    else:
        policy = PPOPolicy(state_size,
                           get_action_size(),
                           use_replay_buffer=False,
                           in_parameters=Namespace(**{'use_gpu': False}))

    policy.load(checkpoint)

    policy.reset(local_env)
    observation = tree_observation.get_many(list(range(nb_agents)))

    print(
        "Evaluation {}: {} agents in {}x{}".format(
            evaluation_number,
            nb_agents,
            local_env.width,
            local_env.height))

    # Now we enter into another infinite loop where we
    # compute the actions for all the individual steps in this episode
    # until the episode is `done`
    steps = 0

    # Bookkeeping
    time_taken_by_controller = []
    time_taken_per_step = []

    # Action cache: keep track of last observation to avoid running the same inferrence multiple times.
    # This only makes sense for deterministic policies.
    agent_last_obs = {}
    agent_last_action = {}
    nb_hit = 0

    policy.start_episode(train=False)
    while True:
        try:
            ###################################################################
            # Evaluation of a single episode
            ###################################################################
            steps += 1
            obs_time, agent_time, step_time = 0.0, 0.0, 0.0
            no_ops_mode = False

            if not check_if_all_blocked(env=local_env):
                time_start = time.time()
                action_dict = {}
                policy.start_step(train=False)
                for agent_handle in range(nb_agents):
                    if info['action_required'][agent_handle]:
                        if agent_handle in agent_last_obs and np.all(
                                agent_last_obs[agent_handle] == observation[agent_handle]):
                            # cache hit
                            action = agent_last_action[agent_handle]
                            nb_hit += 1
                        else:
                            normalized_observation = get_normalized_observation(
                                observation[agent_handle],
                                observation_tree_depth,
                                observation_radius=observation_radius)

                            action = policy.act(
                                agent_handle, normalized_observation, eps=EPSILON)

                    action_dict[agent_handle] = action

                    if USE_ACTION_CACHE:
                        agent_last_obs[agent_handle] = observation[agent_handle]
                        agent_last_action[agent_handle] = action

                policy.end_step(train=False)
                agent_time = time.time() - time_start
                time_taken_by_controller.append(agent_time)

                time_start = time.time()
                _, all_rewards, done, info = remote_client.env_step(
                    map_actions(action_dict))
                step_time = time.time() - time_start
                time_taken_per_step.append(step_time)

                time_start = time.time()
                observation = tree_observation.get_many(list(range(nb_agents)))
                obs_time = time.time() - time_start

            else:
                # Fully deadlocked: perform no-ops
                no_ops_mode = True

                time_start = time.time()
                _, all_rewards, done, info = remote_client.env_step({})
                step_time = time.time() - time_start
                time_taken_per_step.append(step_time)

            nb_agents_done = 0
            for i_agent, agent in enumerate(local_env.agents):
                # manage the boolean flag to check if all agents are indeed
                # done (or done_removed)
                if (agent.status in [RailAgentStatus.DONE,
                                     RailAgentStatus.DONE_REMOVED]):
                    nb_agents_done += 1

            if VERBOSE or done['__all__']:
                print(
                    "Step {}/{}\tAgents done: {}\t Obs time {:.3f}s\t Inference time {:.5f}s\t Step time {:.3f}s\t Cache hits {}\t No-ops? {}".format(
                        str(steps).zfill(4),
                        max_nb_steps,
                        nb_agents_done,
                        obs_time,
                        agent_time,
                        step_time,
                        nb_hit,
                        no_ops_mode
                    ), end="\r")

            if done['__all__']:
                # When done['__all__'] == True, then the evaluation of this
                # particular Env instantiation is complete, and we can break out
                # of this loop, and move onto the next Env evaluation
                print()
                break

        except TimeoutException as err:
            # A timeout occurs, won't get any reward for this episode :-(
            # Skip to next episode as further actions in this one will be ignored.
            # The whole evaluation will be stopped if there are 10 consecutive
            # timeouts.
            print("Timeout! Will skip this episode and go to the next.", err)
            break

    policy.end_episode(train=False)

    np_time_taken_by_controller = np.array(time_taken_by_controller)
    np_time_taken_per_step = np.array(time_taken_per_step)
    print(
        "Mean/Std of Time taken by Controller : ",
        np_time_taken_by_controller.mean(),
        np_time_taken_by_controller.std())
    print(
        "Mean/Std of Time per Step : ",
        np_time_taken_per_step.mean(),
        np_time_taken_per_step.std())
    print("=" * 100)

print("Evaluation of all environments complete!")
########################################################################
# Submit your Results
#
# Please do not forget to include this call, as this triggers the
# final computation of the score statistics, video generation, etc
# and is necessary to have your submission marked as successfully evaluated
########################################################################
print(remote_client.submit())
