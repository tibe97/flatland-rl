from graph_for_observation import GraphObservation
from predictions import ShortestPathPredictorForRailEnv
from agent import RainbowAgent
from dueling_double_dqn import Agent
from model import DQN
import VRSPConv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator
from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.core.env_observation_builder import DummyObservationBuilder
from torch_geometric.data import Data
from torch_geometric.data import DataLoader, Batch
from collections import namedtuple, defaultdict

observation_builder = GraphObservation()
dummy_obs = DummyObservationBuilder()
'''
rail_generator = sparse_rail_generator(max_num_cities=6,
										   seed=1,
										   grid_mode=False,
										   max_rails_between_cities=4,
										   max_rails_in_city=6,
										   )
'''

rail_generator = complex_rail_generator(nr_start_goal=10, nr_extra=1, \
                min_dist=8, max_dist=99999, seed=1)
	
env = RailEnv(width=30,
              height=30,
              rail_generator=rail_generator,
              number_of_agents=8,
              obs_builder_object=observation_builder)


env.reset()

env_renderer = RenderTool(env)
env_renderer.render_env(show=True, frames=True, show_observations=False)

obs, all_rewards, done, info = env.step({0: 2})
GraphInfo = namedtuple("GraphInfo", "agent_handle, path")
batch_list = []
counter = 0
for a in obs.keys():
    obs_part = obs[a]["partitioned"]
    for path in obs_part:
        data = Data(x=obs_part[path]["node_features"], edge_index=obs_part[path]["graph_edges"])
        data.graph_info = GraphInfo(a, path)
        data.index = counter
        data.new_to_old_map = obs_part[path]["new_to_old_map"]
        batch_list.append(data)
        counter += 1

batch = Batch.from_data_list(batch_list)


agent = Agent(15, 4, {})

out = agent.act(batch)

env.obs_builder.plot_track_map()
"""
out_mapped = defaultdict(lambda: defaultdict(list))
for i, res in enumerate(out):
    batch_index = batch.batch[i]
    handle, path = batch_list[batch_index].graph_info
    out_mapped[handle][path].append(res)
    # out_mapped.append([res, handle, path])
"""   

print("ciao")




