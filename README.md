# temporal-dynamical-networks

The data folder contains subdatafolders for each of the 39 studied temporal network datasets.
The pre-processed temporal data can be found in the following files:
- *_data_graph.pkl stores the temporal edge-grouped data. For each edge index e, data_graph[e] is the list of timestamps associated with that edge.
- *_data_node_edge_translation.pkl stores the mapping between node pairs and edge indices. The key (node_i, node_j) maps to the edge index e, which identifies the corresponding entry in *_data_graph.pkl.

The files generated from the activation probability (edge- and node-level) analyses can also be found in the corresponding dataset subfolders (Window width = column 1, corresponding activation probability = column 2):
- *_edge_Activation_probability_overlap*_MAXwindows1000.txt (for edge-level, mean activation probability)
- *_edge_Activation_probability_std_overlap*_MAXwindows1000.txt (for edge-level, std activation probability)
- *_node_Activation_probability_overlap*_MAXwindows1000.txt (for node-level, mean activation probability)
- *_node_Activation_probability_std_overlap*_MAXwindows1000.txt (for node-level, std activation probability)

For the bibsonomy dataset, we also provide the datafiles generated during other analyses:
- bibsonomy_preprocessing_effect_on_* for the effect of the cutoff w^* on various measures
- bibsonomy_ccdf_degree_reduced_observation_*offull.pkl for the effect of partial observation windows on the degree distribution
- bibsonomy_ccdf_iet_reduced_observation_*offull.pkl for the effect of partial observation windows on the iet distribution
- bibsonomy_data_graph_projection*.pkl for the temporal edge-grouped data after projection on one layer
- bibsonomy_data_node_edge_translation_projectionA.pkl for the mapping between node pairs and edge indices after projection on one layer
- bibsonomy_degree_degree_change_overlap*_MAXwindows1000_eta*.txt for the raw Delta k_n vs k_n points
- bibsonomy_mean_pop_degree_degree_change_overlap*_MAXwindows1000_eta*.txt for the cyclic trajectory in <Delta k_n> vs <k_n>
- bibsonomy_slopes_overlap*_MAXwindows1000.txt for the slopes vs overlap
