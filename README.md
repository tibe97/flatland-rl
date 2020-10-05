# flatland_rl

### Observation
The rail environment can be seen as composed of track sections, where each track section is a portion of the track delimited by 2 switches (forks). A switch is a cell where the train can choose between 2 different direcitions, so it's a choice point. Other cells where there is only one possible direction are part of at least one track section. There are cells belonging to 2 track sections, these are the intersections.
The state is the subjective point of view of an agent. Each agent has its own state-representation.
The state is composed of the neighboring tracks features, up to a certain depth/level. In this way we perform GNN of track (nodes in this case) features. For every agent each track feature could be different, because they are seen from its point of view. 

### Graph Neural Network (GNN) approach
The problem of finding a suitable observation is the variable size of the state. In fact by choosing a fixed-size state representation we could limit ourselves when changing the size of the rail environment. By using a fixed-size observation we use are determining a fixed architecture of the NN, which we would tend to implement as big as possible in order to capture the most of information available. But this leads to an increase in complexity.
By using Graph Neural Networks we leverage the natural undelying graph structure of a railway. More importantly however, we don't limit ourselves to a fixed size state, since with a GNN we can have whatever number of nodes at a certain layer, because in the end all the values will be aggregated.
Assign to each track section (node of the graph) a value. When navigating, each agent runs GNN independently from other agents (each agent has its own graph representation). Each agent access information about other agents position and speed.
Each node represents the value of reaching it. So when the agent arrives at a switch, it has to decide which direction to go, and it does it by choosing the direction leading to the track section with the highest value. It also captures in some way the value of the possible through it. (Similarly to Q-value in RL) ---> Can we use RL methods to assign a value to each node?
This method requires to compute the value for each adjacent node to the switch, which we will compare to make the decision of the direction to go.

### GNN propagation
The input layer is just the node representations (part of the state).
We then compute the intermediary hidden states with NN and aggregations.
We can start with 3 layers.
The output value is computed not only for the track where the agent is (to determine if the best action is to STOP at the current track or not), but also for the tracks reachable from the next fork (choice points). For example another train could be passing in the chosen track, so we don't want to go against it and cause a deadlock
We do this because we adopt a value-based approach, such that we choose the track with the highest value. 


### Reinforcement learning
PROBLEMS with policy-based methos: neighbors are order invariant, so each node can't recall from which direction a certain information comes from. We can't really determine the value of choosing a certain direction in this way.
So, policy-based approach should be NOT FEASIBIBLE combined with Graph Neural Network approach.
For this reason we use a Value-based approach where we only compute the values of the reachable track section from a certain switch and then select the path leading to the highest value.

### Track section features
This feature is used to represent railway tracks. In this approach, tracks become the nodes of the graph, while the switches are implicitly represented by the nature of the neighbourhood of a certain track (possible directions).
In this way we simplify an important aspect:
• Adjacent tracks which are not directly reachable with an action are not considered, because we only add an edge between 2 tracks that are directly linked and communicating.

A possible problem emerging from this type of representation would be that intersecting paths are not directly represented.
We can thus integrate this type of information in the track feature representation.

### GNN Convolution layer
This is the explaination of the VRSPConvolution.py file.
The GNN convolution layer consists of a target-to-source convolutions, where the target track section (N2) features and the parent track section (N2, from which we reach the child track section) are concatenated into a 1D vector and then fed into a 2-layer Neural Network. The output of the first conv layer is part of the hidden representation of parent track section N2. In fact we compute the output for all the pairs (N2, N_child) and then combine then in a max, mean and min pooling layer. In this way we obtain a final hidden representation at layer 1 for node N2. This is done for all nodes at layer 1 (layer 0 is represented by input features, i.e. track features).
We repeat the conv layer other 2 times for a total of 3 conv layers, so the depth of out GNN can be considered to be 3. This means that from the current node or track section where we are, we can reach the information of nodes 3 hops away from us.
The root node is the only node about which we care the value, because it represents all the path reachable (at depth 3) from it.
So, from the current switch, we consider as root node of different trees each reachable track section, in addition to the section we are at.
