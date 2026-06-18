import pickle
import os
os.environ["PATH"] = "/Library/TeX/texbin:/opt/homebrew/bin:" + os.environ["PATH"]

from functions_library import *
import numpy as np

def get_activation_prob(Activity):

    activation_prob = []
    for index_e in range(Activity.shape[0]):
        vector_active = list(Activity[index_e, :]) # tells us the edge time evolution of its activity level (0 if not active or 1 if active (more than one event))
        vector_activation = list(np.diff(vector_active))
        num_activation = vector_activation.count(1) # counts how many times the edge activates in a single time step: 0->1

        activation_prob.append(num_activation/len(vector_activation))
    mean_Pplus = np.mean(activation_prob)
    std_Pplus = np.std(activation_prob)

    return mean_Pplus, std_Pplus

def get_time_evolution_of_degree_and_activation(graph_info_list, translation_dict, tmin, tmax, eta, nu, MAX_windows=1000):
    # make a function that computes the egde, node, triad, network activation probability for a given dataset, eta and nu. The function also computes the evolution of degree
    times = get_window_times(tmin, tmax, eta, nu, MAX_windows)

    num_edges = len(graph_info_list)
    Activity_edge = np.zeros((num_edges, len(times)))
    triads = triads_from_edge_indices(graph_info_list, translation_dict)
    if len(triads): Activity_triad = np.zeros((len(triads), len(times)))    
    Activity_network = np.zeros(len(times))
    nodes = get_nodes_from_list_edge_index(list(translation_dict.values()), translation_dict)
    K_dyn = np.zeros((len(nodes), len(times)))
    Activity_node = np.zeros((len(nodes), len(times)))

    if times:
        for time_iter, t in enumerate(times):
            if time_iter % 100 == 0: print(len(times) - time_iter)

            list_now = filter_events_in_interval(graph_info_list, t-eta, t)
            active_edges_now = [i for i, lst in enumerate(list_now) if lst]

            # create the matrix with rows=edges and columns=times, such that elements are 0 or 1 if the edge is active or inactive
            for index_e in active_edges_now:
                Activity_edge[index_e, time_iter] = 1

            if len(triads): 
                triads_now = triads_from_edge_indices(active_edges_now, translation_dict)
                if len(triads_now) > 0:
                    for iter_triad, i in enumerate(triads_now):
                        Activity_triad[iter_triad, time_iter] = 1

            # create the matrix with rows=nodes and columns=times, such that elements are the degree of the node
            nodes_now, neighbours_dict_now = get_nodes_from_list_edge_index(active_edges_now, translation_dict, k_dict = True) # nodes that are active in the present time window
            for iter_node, i in enumerate(nodes_now):
                K_dyn[i, time_iter] = neighbours_dict_now[iter_node]
                Activity_node[iter_node, time_iter] = 1

            if len(active_edges_now) >= 1:
                Activity_network[time_iter] = 1

    mean_edge_Pplus, std_edge_Pplus = get_activation_prob(Activity_edge)
    mean_node_Pplus, std_node_Pplus = get_activation_prob(Activity_node)
    if len(triads): 
        mean_triad_Pplus, std_triad_Pplus = get_activation_prob(Activity_triad)
    else:
        mean_triad_Pplus = None
        std_triad_Pplus = None

    vector_activation = list(np.diff(Activity_network))
    num_activation = vector_activation.count(1) # counts how many times the triad activates in a single time step: 0->1
    mean_network_Pplus = num_activation/len(vector_activation)

    return mean_edge_Pplus, std_edge_Pplus, mean_node_Pplus, std_node_Pplus, mean_triad_Pplus, std_triad_Pplus, mean_network_Pplus, K_dyn

def triads_from_edge_indices(edge_list, translation_dict):
    """
    Given a list of edge indices and a translation dict ((u, v) -> edge_index),
    return sets of open and closed triads (as sorted node triples).
    """
    edge_set = set()
    node_set = set()

    for (u, v), idx in translation_dict.items():
        if idx < len(edge_list):
            u, v = sorted((u, v))
            edge_set.add((u, v))
            node_set.update([u, v])

    triads = []
    for triad in combinations(node_set, 3):
        u, v, w = sorted(triad)

        e1 = (u, v)
        e2 = (u, w)
        e3 = (v, w)

        count = sum(e in edge_set for e in (e1, e2, e3))

        if count == 3:
            triads.append((translation_dict[e1], translation_dict[e2], translation_dict[e3]))

    return triads

def save_two_col_data(filename_activation, x, y):
    data = np.column_stack((x, y))
    with gzip.open(filename_activation, "wt") as f:
        np.savetxt(f, data, fmt="%.6f")

def read_two_col_data(filename_activation):
    with gzip.open(filename_activation, "rt") as f:
        data_loaded = np.loadtxt(f)

    x = data_loaded[:, 0]
    y = data_loaded[:, 1]
    return x, y

all_dfs = ['bibsonomy', 'edit_wiktionary/mg', 'escorts', 'movielens_100k', 'citeulike', 'sp_infectious', 'edit_wiktionary/fr', 'wiki_link_dyn', 'edit_wiktionary/en', 'lastfm/song', 'wikiconflict', 'facebook_wall', 'contact', 'email_company', 'lkml_reply', 'sp_primary_school', 'sp_colocation/SFHH', 'sp_colocation/InVS15', 'sp_colocation/InVS13', 'sp_colocation/LH10', 'lastfm/band', 'edit_wikibooks/de', 'edit_wikibooks/fr', 'sp_high_school/proximity', 'sp_colocation/Thiers13', 'wiki_talk/fr', 'wiki_talk/de', 'sp_hypertext/contacts', 'sp_high_school_new/2011', 'wiki_talk/en', 'sp_office', 'edit_wikibooks/en', 'sp_high_school_new/2012', 'reality_mining', 'lkml_thread', 'sp_colocation/LyonSchool', 'sp_hospital', 'edit_wikinews/en', 'edit_wikinews/it']

iter_dataset = int(sys.argv[1])
overlap = float(sys.argv[2])

dataset_name = all_dfs[iter_dataset]
MAX_windows = 1000

# loading the mean IET and std IET of each dataset
with open(f'./mean_std_IET.pkl', 'rb') as f:
    dataset_names_file2, mean_iet, std_iet = pickle.load(f)

# graph info
new_name = dataset_name.replace("/", "_")
graph_list_filename = f"./data/{new_name}/{new_name}_data_graph.pkl"
translation_dict_null_filename = f"./data/{new_name}/{new_name}_data_node_edge_translation.pkl"
with open(graph_list_filename, 'rb') as f:
    graph_info_list = pickle.load(f)

tmin = min(min(row) for row in graph_info_list)
tmax = max(max(row) for row in graph_info_list)
with open(translation_dict_null_filename, 'rb') as f:
    translation_dict = pickle.load(f)

etas = get_log_spaced_array(overlap, mean_iet[iter_dataset], tmin, tmax, min_val=0.0001, resol=20, min_windows=3)

# # where to save activation prob info
activation_filename_mean_edge = f"./data/{new_name}/{new_name}_edge_Activation_probability_overlap{overlap}_MAXwindows{MAX_windows}.txt"
activation_filename_std_edge = f"./data/{new_name}/{new_name}_edge_Activation_probability_std_overlap{overlap}_MAXwindows{MAX_windows}.txt"
activation_filename_mean_node = f"./data/{new_name}/{new_name}_node_Activation_probability_overlap{overlap}_MAXwindows{MAX_windows}.txt"
activation_filename_std_node = f"./data/{new_name}/{new_name}_node_Activation_probability_std_overlap{overlap}_MAXwindows{MAX_windows}.txt"
activation_filename_mean_triad = f"./data/{new_name}/{new_name}_triad_Activation_probability_overlap{overlap}_MAXwindows{MAX_windows}.txt"
activation_filename_std_triad = f"./data/{new_name}/{new_name}_triad_Activation_probability_std_overlap{overlap}_MAXwindows{MAX_windows}.txt"
activation_filename_mean_network = f"./data/{new_name}/{new_name}_network_Activation_probability_overlap{overlap}_MAXwindows{MAX_windows}.txt"

# computing activation probability and saving it.
for eta in etas:
    degree_degree_change_filename = f"./data/{new_name}/{new_name}_degree_degree_change_overlap{overlap}_MAXwindows{MAX_windows}_eta{eta:.2f}.txt.gz"
    mean_pop_degree_degree_change_filename = f"./data/{new_name}/{new_name}_mean_pop_degree_degree_change_overlap{overlap}_MAXwindows{MAX_windows}_eta{eta:.2f}.txt.gz"

    print('eta', eta)
    nu = eta*(1-overlap) # define the time jump using the window width eta and the overlap between consecutive windows

    mean_edge_Pplus, std_edge_Pplus, mean_node_Pplus, std_node_Pplus, mean_triad_Pplus, std_triad_Pplus, mean_network_Pplus, K_dyn = get_time_evolution_of_degree_and_activation(graph_info_list, translation_dict, tmin, tmax, eta, nu)

    with open(activation_filename_mean_edge, "a") as file:
        file.write(f"{eta}, {mean_edge_Pplus}\n")
    with open(activation_filename_std_edge, "a") as file:
        file.write(f"{eta}, {std_edge_Pplus}\n")

    with open(activation_filename_mean_node, "a") as file:
        file.write(f"{eta}, {mean_node_Pplus}\n")
    with open(activation_filename_std_node, "a") as file:
        file.write(f"{eta}, {std_node_Pplus}\n")

    if mean_triad_Pplus is not None:
        with open(activation_filename_mean_triad, "a") as file:
            file.write(f"{eta}, {mean_triad_Pplus}\n")
        with open(activation_filename_std_triad, "a") as file:
            file.write(f"{eta}, {std_triad_Pplus}\n")

    with open(activation_filename_mean_network, "a") as file:
        file.write(f"{eta}, {mean_network_Pplus}\n")

    degree, degree_change, mean_pop_degree_evol, mean_pop_degree_change_evol = get_deltak_k(K_dyn)

    save_two_col_data(degree_degree_change_filename, degree, degree_change)
    save_two_col_data(mean_pop_degree_degree_change_filename, mean_pop_degree_evol[:-1], mean_pop_degree_change_evol)
