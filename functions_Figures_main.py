import os
os.environ["PATH"] = "/Library/TeX/texbin:/opt/homebrew/bin:" + os.environ["PATH"]

import pickle
import numpy as np

import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm, gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize

import scipy.stats as stats
from scipy.interpolate import PchipInterpolator

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import fitz

import model_misc_Gerardo as mm
from functions_library import *

def combine_two_colormaps(cmap1, cmap2, split=0.66, N=256):
    """Return a combined colormap with cmap1 below split, cmap2 above split."""
    N1 = int(N * split)
    N2 = N - N1

    colors1 = cmap1(np.linspace(0, 1, N1))
    colors2 = cmap2(np.linspace(0, 1, N2))

    combined = np.vstack([colors1, colors2])
    return mcolors.LinearSegmentedColormap.from_list("combined", combined)

def read_Pplus_eta_data(filename):
    eta, Pplus = np.loadtxt(filename, delimiter=",", unpack=True)

    pair = sorted(zip(eta, Pplus))
    sorted_etas, sorted_pplus = zip(*pair)
    return sorted_etas, sorted_pplus

def sort_data(x, y):
    pair = sorted(zip(x, y))
    sorted_etas, sorted_pplus = zip(*pair)

    return sorted_etas, sorted_pplus

def get_color(value, norm, cmap_blue_orange):
    return cmap_blue_orange(norm(value))

def l1_cdf_distance(sample, target_cdf, ngrid=1000):
    x = np.linspace(np.min(sample), np.max(sample), ngrid)
    s_sorted = np.sort(sample)
    Fn = np.searchsorted(s_sorted, x, side='right') / len(sample)
    Fx = target_cdf(x)
    l1 = np.trapz(np.abs(Fn - Fx), x)         # integral of vertical differences
    l1_per_x = l1 / (x[-1] - x[0])            # mean vertical difference (unitless)
    return l1, l1_per_x

def get_plot_pplus(iter_data, ax, overlap, colors, all_dfs, lw, mean_iet, MAX_windows):
    alphas = {0.0: 0.2, 0.2: 0.4, 0.4:0.6, 0.6:0.8, 0.9:1}
    dataset_name = all_dfs[iter_data]
    color = colors[iter_data]
    new_name = dataset_name.replace("/", "_")
    alpha_overlap = alphas[overlap]

    activation_filename_mean_edge = f"./data/{new_name}/{new_name}_edge_Activation_probability_overlap{overlap}_MAXwindows{MAX_windows}.txt"
    eta, Pplus = read_Pplus_eta_data(activation_filename_mean_edge) #np.loadtxt(activation_filename_mean_edge, delimiter=",", unpack=True)

    pair = sorted(zip(eta, Pplus))
    sorted_etas, sorted_pplus = zip(*pair)
    r = mean_iet[iter_data]
    ax.plot(sorted_etas/r, sorted_pplus, color = color, lw = lw, alpha = alpha_overlap,label = f'{overlap}')
    
    ax.set_xscale('log')
    
def legend_format(x):
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(x)))
    mant = x / 10**exp
    return rf"10^{exp}" #rf"{int(mant)}\cdot 10^{exp}"

def P(eta_considered, x_cont, ccdf_cont, mu, std, mean):
    term1 = (mu - E0(eta_considered, x_cont, ccdf_cont))*(std**2 + mean**2 - mean) 
    term2 = mean*mu
    return term1 - term2
    
def find_eta_zero(Q, eta_min, eta_max, tol=1e-6, max_iter=100):

    Q_min = Q(eta_min)
    Q_max = Q(eta_max)
    
    if Q_min > 0 or Q_max < 0:
        raise ValueError("Function does not have a root in the interval or is not monotonic increasing")
    
    for i in range(max_iter):
        eta_mid = (eta_min + eta_max) / 2
        Q_mid = Q(eta_mid)

        if abs(Q_mid) < tol or (eta_max - eta_min) < 1:
            return eta_mid
                
        if Q_mid > 0:
            eta_max = eta_mid
        else:  
            eta_min = eta_mid
    
    return (eta_min + eta_max) / 2

def exp_func(x, A, B):
    return A * np.exp(B * x)

def make_fig_theor_pplus(prms, ax, color, my_label, alpha = 1, edgecolor = None, my_legend = 'on'):
    ovl_frac = prms['ovl_frac']
    iet_dist, iet_name = prms['iet_dist'], prms['iet_name']
    win_arr = np.logspace(-1, 3, 100) #array of window values 50
    iet_arr = mm.renewal_process(prms)
    edge_zero_arr = np.zeros(len(win_arr)) #init array of 0-edge fractions
    prob_trans_arr = np.zeros((len(win_arr), 2, 2)) #init array of trans probs
    for win_pos, win in enumerate(win_arr): #loop through window values
        prms['win'] = win #update parameter dict
        prms['ovl'] = win*ovl_frac #window overlap
        iet_cnt, edg_dist, cnt_prev, cnt_next = mm.edge_types_full(iet_arr, prms)
        edge_zero_arr[win_pos] = edg_dist[0] #store 0-edge fraction
        prob_trans_arr[win_pos, :, :] = mm.edge_trans_prob(iet_arr, prms)
    win_opt, act_prob_max = mm.optimal_window(prms, type=iet_name)
    
    #numerical plot
    xplot = win_arr / iet_arr.mean()
    yplot = prob_trans_arr[:, 0, 1]
    yplot /= yplot.max() # normalise
    ax.semilogx(xplot, yplot, 'o', color = color, markeredgecolor = edgecolor, alpha = alpha, zorder=1) #prob to gain/lose a link (equal to prob_trans_arr[:, 1, 0] due to time reversal)
    #theoretical plots
    xplot = win_arr / iet_arr.mean()
    yplot = np.zeros(len(xplot))
    for win_pos, win in enumerate(win_arr): #loop through window values
        yplot[win_pos] = mm.act_prob_theo(win, ovl_frac, iet_dist, type=iet_name)
    yplot /= act_prob_max #normalize
    ax.semilogx(xplot, yplot, '-', color = color, zorder=2, alpha = alpha)
    ax.axvline(win_opt/iet_dist.mean(), ls='--', lw = 3, color = color, zorder=0, alpha = alpha) #optimal aggregation window
    if my_legend == 'on':
        ax.legend(loc='upper right')

    # return win_opt/iet_dist.mean()

def make_fig_theor_sn(prms, ax, color, my_label, alpha = 1, edgecolor = None, my_legend = 'on'):
    ovl_frac = prms['ovl_frac']
    iet_dist, iet_name = prms['iet_dist'], prms['iet_name']
    deg_dist, deg_name = prms['deg_dist'], prms['deg_name']
    win_arr = np.logspace(-1, 3, 100) #array of window values 50
    iet_arr = mm.renewal_process(prms)
    lcc_win_arr, lcc_arr = np.zeros(len(win_arr)), np.zeros(len(win_arr))
    lcc_win_theo, lcc_theo = np.zeros(len(win_arr)), np.zeros(len(win_arr))
    for win_pos, win in enumerate(win_arr): #loop through window values
        prms['win'] = win #update parameter dict
        prms['ovl'] = win*ovl_frac #window overlap
        lcc_win_arr[win_pos], lcc_arr[win_pos] = mm.network_lcc(iet_arr, prms)
        lcc_win_theo[win_pos], lcc_theo[win_pos] = mm.network_lcc_theo(deg_name, deg_dist, mm.edge_zero_theo(win, iet_dist, type=iet_name))
    win_opt_lcc = mm.optimal_window_lcc(prms, type=None)
    xplot = win_arr / iet_arr.mean()
    yplot = lcc_win_arr / lcc_arr
    ax.semilogx(xplot, yplot, 'o', color = color, alpha = alpha, markeredgecolor = edgecolor, zorder=1) #E_0
    xplot = win_arr / iet_arr.mean()
    yplot = lcc_win_theo / lcc_theo
    ax.semilogx(xplot, yplot, '-', color = color, zorder=2, alpha = alpha)
    ax.axvline(win_opt_lcc/iet_dist.mean(), ls=':', lw = 3, color = color, alpha = alpha, zorder=0) #optimal aggregation window

def find_Psi_eta(eta_target, cdf_x_values, ccdf_vals):
    idx = np.abs(cdf_x_values - eta_target).argmin()

    y_target = ccdf_vals[idx]
    return y_target

def E0(eta_target, cdf_x_values, ccdf_vals):
    sorted_idx = np.argsort(cdf_x_values)
    x_sorted = cdf_x_values[sorted_idx]
    y_sorted = ccdf_vals[sorted_idx]

    mask = x_sorted >= eta_target
    y_filtered = y_sorted[mask]

    return np.sum(y_filtered)

def Q(eta, x_cont, ccdf_cont, mu, overlap):
    term1 = find_Psi_eta(eta, x_cont, ccdf_cont) * (mu - E0(eta * (1 - overlap), x_cont, ccdf_cont))
    term2 = (1 - overlap) * E0(eta, x_cont, ccdf_cont) * find_Psi_eta(eta * (1 - overlap), x_cont, ccdf_cont)
    return term1 - term2

def get_chosen_eta(which, new_name, overlap, iter_eta, MAX_windows = 1000):
    if which == 'node':
        activation_filename_mean_edge = f"./data/{new_name}/{new_name}_node_Activation_probability_overlap{overlap}_MAXwindows{MAX_windows}.txt"
        xx, yy = read_Pplus_eta_data(activation_filename_mean_edge)

        index_eta = np.argmax(yy)
        eta = xx[index_eta]
    if which == 'edge':

        activation_filename_mean_edge = f"./data/{new_name}/{new_name}_edge_Activation_probability_overlap{overlap}_MAXwindows{MAX_windows}.txt"
        xx, yy = read_Pplus_eta_data(activation_filename_mean_edge) #np.loadtxt(activation_filename_mean_edge, delimiter=",", unpack=True)

        index_eta = np.argmax(yy)
        eta = xx[index_eta]

    if which == 'choose':
        folder_path = f"./data/{new_name}/"
        pattern = f"{new_name}_degree_degree_change_overlap{float(overlap)}_MAXwindows1000_eta*.txt.gz"                
        search_path = os.path.join(folder_path, pattern)
        file_list = glob.glob(search_path)
        etas = []
        for file in file_list:
                base = os.path.basename(file)
                match = re.search(r"eta([0-9.]+)\.txt.gz$", base)
                if match:
                        eta = float(match.group(1))
                        etas.append(eta)

        sorted_etas = sorted(etas)
        eta = sorted_etas[iter_eta]
        
    return eta

def read_two_col_data(filename_activation):
    with gzip.open(filename_activation, "rt") as f:
        data_loaded = np.loadtxt(f)

    x = data_loaded[:, 0]
    y = data_loaded[:, 1]
    return x, y
    
def plot2(ax, degree_change, color):
    ax.hist(degree_change, bins=10, alpha=0.3, color=color, density=True, orientation='horizontal')
    # ax.axhline(y=0, color='k', lw = 1)
    ax.set_xscale('log')
    ax.set_yticklabels([])
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

def plot3(ax, degree, color):
    ax.hist(degree, bins=10, alpha=0.3, color=color, density=True)
    ax.set_yscale('log')
    ax.set_xticklabels([])

def one_decimal(x, pos):
    s = f"{x:.1f}"
    return s.replace('-', '−')

def get_quantile_binned(x_values, y_values, num_bins=10):
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)

    mask = x_values > 0
    x_values = x_values[mask]
    y_values = y_values[mask]

    order = np.argsort(x_values)
    x_sorted = x_values[order]
    y_sorted = y_values[order]

    x_bins = np.array_split(x_sorted, num_bins)
    y_bins = np.array_split(y_sorted, num_bins)

    x_binned = np.array([np.mean(xb) for xb in x_bins if len(xb) > 0])
    y_binned = np.array([np.mean(yb) for yb in y_bins if len(yb) > 0])

    return x_binned, y_binned

def plot1(ax, degree, degree_change, color, dataset_name):
    ax.scatter(degree, degree_change, color = color, alpha=0.2)
    bin_degree, bin_degree_change = get_quantile_binned(degree, degree_change, num_bins = 10)
    if bin_degree[0] >= np.mean(degree)/2:
        bin_degree, bin_degree_change = get_quantile_binned(degree, degree_change, num_bins = np.max(degree)) #50)
    ax.plot(bin_degree, bin_degree_change, color = color, lw = 3, label = f'{dataset_name}')

def get_unique_edges_of_node_from_translation_dict(node, translation_dict):
    node = int(node)
    return list({
        edge_id
        for (s, t), edge_id in translation_dict.items()
        if s == node or t == node
    })

def build_node_to_edges(translation_dict):
    node_to_edges = defaultdict(list)
    for (s, t), edge_id in translation_dict.items():
        node_to_edges[s].append(edge_id)
        node_to_edges[t].append(edge_id)
    return node_to_edges

def get_nodes_from_list_edge_index(list_edge_index, edges_to_index, k_dict=False):
    # Precompute reverse mapping (O(n))
    index_to_edge = {v: k for k, v in edges_to_index.items()}
    
    nodes_involved = []
    
    if k_dict:
        neighbours_dict = defaultdict(list)
    
    for edge_index in list_edge_index:
        edge = index_to_edge.get(edge_index)
        if edge is None:
            continue  # or raise error if preferred
        
        u, v = edge
        nodes_involved.extend((u, v))
        
        if k_dict:
            neighbours_dict[u].append(edge_index)
            neighbours_dict[v].append(edge_index)
    
    nodes = np.unique(nodes_involved)
    
    if not k_dict:
        return nodes
    
    degree_list = [len(neighbours_dict[node]) for node in nodes]
    return nodes, degree_list

def include_layout(x_label, y_label, iter_data, ax, cols, rows, legend = 'on'): # not complete
    if legend == 'on':
        if iter_data == 5:
            ax.legend()

    row = iter_data // cols
    col = iter_data % cols
    if row == rows - 1:
        ax.set_xlabel(x_label)
    elif row == rows - 2:
        if col == 3 or col == 4 or col == 5:
            ax.set_xlabel(x_label)
    else:
        ax.set_xlabel('')
    if col == 0:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel('')
    ax.tick_params(axis='both', which='both', labelsize=16)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

def cdf(data):
    """
    Computes the empirical CDF.
    Returns:
        x_points : unique sorted values
        cdf_vals : P(X < x)
    """

    if isinstance(data, dict):
        arr = np.asarray(list(data.values()))
    else:
        arr = np.asarray(data)

    arr = np.sort(arr)
    n = arr.size

    x_points, first_idx = np.unique(arr, return_index=True)

    cdf_vals = first_idx / n

    return x_points, cdf_vals

def get_gamma_max_dyn_scale(mu, overlap):
    burstiness_values = np.linspace(0.05, 0.9, 100)
    x = np.geomspace(0.0000001, 1000, 1000)

    ccdf_s = []
    x_values = []
    Bs_gamma = []
    for B in burstiness_values:
        sigma = mu * (1 + B) / (1 - B)      
        shape = mu**2 / sigma**2           
        scale = sigma**2 / mu              
        pdf = stats.gamma.pdf(x, a=shape, scale=scale)

        ccdf = stats.gamma.sf(x, a=shape, scale=scale)
        ccdf_s.append(list(ccdf))
        x_values.append(list(x))

        n = len(x)
        if n >= 1:
            r = sigma/mu
            B_s = compute_burstiness(r, n)
        Bs_gamma.append(B_s)

    eta_stars_gamma = []

    eta_min = np.min(x)
    eta_max = np.max(x)
    for xs, ccdfs in zip(x_values, ccdf_s):

        ccdf_cont_spline = PchipInterpolator(xs, ccdfs)
        x_cont = np.linspace(xs[0], xs[-1], 1000)
        ccdf_cont = ccdf_cont_spline(x_cont)

        mu = E0(0, x_cont, ccdf_cont)

        eta_star = find_eta_zero(
            Q=lambda eta: Q(eta, x_cont, ccdf_cont, mu, overlap),
            eta_min=eta_min,
            eta_max=eta_max,
            tol=1e-6,
            max_iter = 100
        )

        eta_stars_gamma.append(eta_star/mu)
    return  Bs_gamma, eta_stars_gamma

def get_burstiness_node_edge(all_dfs, overlap, mean_iet):
    Bs_nodes = []
    Bs_edges = []
    eta_star_nodes = []
    eta_star_edges = []
    for iter_data in range(39):
        dataset_name = all_dfs[iter_data]
        new_name = dataset_name.replace("/", "_")
        graph_list_filename = f"./data/{new_name}/{new_name}_data_graph.pkl"
        translation_dict_null_filename = f"./data/{new_name}/{new_name}_data_node_edge_translation.pkl"
        with open(graph_list_filename, 'rb') as f:
            graph_info_list = pickle.load(f)
        with open(translation_dict_null_filename, 'rb') as f:
            translation_dict = pickle.load(f)
            
        nodes_now, _ = get_nodes_from_list_edge_index(list(translation_dict.values()), translation_dict, k_dict = True)
        node_to_edges = build_node_to_edges(translation_dict)
        eta_star_node = get_value_from_file(f'optimally_dynamic_scale_node_overlap{float(overlap)}_MAXwindows1000.txt', f'{dataset_name}')
        eta_star_edge = get_value_from_file(f'optimally_dynamic_scale_edge_overlap{float(overlap)}_MAXwindows1000.txt', f'{dataset_name}')

        node_iets = []
        for node in nodes_now:
            edges = node_to_edges[node]
            
            events = [
                event
                for e in edges
                for event in graph_info_list[e]
            ]
            
            if len(events) > 1:
                node_iets.extend(np.diff(np.sort(events)))

        mean_tau_node = np.mean(node_iets)
        mean_tau = mean_iet[iter_data]

        if eta_star_node is not None:
            eta_star_nodes.append(eta_star_node/mean_tau_node)
        else:
            eta_star_nodes.append(eta_star_node)

        Bs_node = get_system_burstiness(node_iets)
        Bs_nodes.append(Bs_node)

        filename = f'./data/{new_name}/{new_name}_edge_interevents.pkl'
        edge_iets = {}
        with open(filename, "rb") as file:
            edge_iets = pickle.load(file)
        system_interevents = flatten_dict_and_concatenate(edge_iets)

        mean_tau_edge = np.mean(system_interevents)
        if eta_star_edge is not None:
            eta_star_edges.append(eta_star_edge/mean_tau_edge)
        else:
            eta_star_edges.append(eta_star_edge)

        # system burstiness
        Bs_edge = get_system_burstiness(system_interevents)
        Bs_edges.append(Bs_edge)

    return Bs_edges, Bs_nodes

def get_openness(dataset_names, level, its_s_burstiness):
    l1_means = []
    for iter_data, dataset_name in enumerate(dataset_names):

        dataset_name = dataset_names[iter_data]
        new_name = dataset_name.replace("/", "_")

        name_list = f'./data/{new_name}/{new_name}_{level}_firstevents.pkl'
        with open(name_list, 'rb') as f:
            minimums = pickle.load(f)

        ccdf_filename = f"./data/{new_name}/{new_name}_ccdf_{level}_interevents.pkl"
        with open(ccdf_filename, 'rb') as f:
            ccdf = pickle.load(f)
        cdf_x_values = ccdf[:, 0] + min(minimums)
        ccdf_vals = ccdf[:, 1]
        ccdf_vals /= ccdf_vals.sum()
        cdf_values = np.cumsum(ccdf_vals)

        def target_cdf(x):
            return np.interp(x, cdf_x_values, cdf_values, left=0.0, right=1.0)

        l1_total, l1_mean = l1_cdf_distance(minimums, target_cdf)
        l1_means.append(l1_mean)

    datasets_open = []
    datasets_closed = []
    burstiness_open = []
    burstiness_closed = []
    for iter_data, dataset_name in enumerate(dataset_names):
        bs = its_s_burstiness[iter_data]
        if l1_means[iter_data] > np.median(l1_means):
            datasets_open.append(dataset_name)
            burstiness_open.append(bs)
        else:
            datasets_closed.append(dataset_name)
            burstiness_closed.append(bs)

    return l1_means, datasets_open, datasets_closed, burstiness_open, burstiness_closed

def get_eta_c_and_star(all_dfs, mean_iet, overlap, mean_ks, std_ks):
    eta_stars = []
    eta_cs = []

    for iter_data, dataset_name in enumerate(all_dfs):

        new_name = dataset_name.replace("/", "_")

        # get IETs
        filename = f"./data/{new_name}/{new_name}_ccdf_edge_interevents.pkl"
        with open(filename, "rb") as file:
            data = pickle.load(file)
        cdf_x_values = data[:, 0]
        ccdf_vals = data[:, 1]

        ccdf_cont_spline = PchipInterpolator(cdf_x_values, ccdf_vals)
        x_cont = np.linspace(cdf_x_values[0], cdf_x_values[-1], 1000)
        ccdf_cont = ccdf_cont_spline(x_cont)

        mu = E0(0, x_cont, ccdf_cont)

        eta_min = 0.00000001
        eta_max = 80*mean_iet[iter_data] 

        eta_star = find_eta_zero(Q=lambda eta: Q(eta, x_cont, ccdf_cont, mu, overlap), eta_min=eta_min, eta_max=eta_max, tol=1e-6, max_iter = 100)
        eta_stars.append(eta_star/mean_iet[iter_data])

        eta_c = find_eta_zero(Q=lambda eta: P(eta, x_cont, ccdf_cont, mu, std_ks[iter_data], mean_ks[iter_data]), eta_min=eta_min, eta_max=eta_max, tol=1e-6, max_iter = 100)
        eta_cs.append(eta_c/mean_iet[iter_data])
        
    eta_stars = np.array(eta_stars)
    eta_cs = np.array(eta_cs)

    return eta_stars, eta_cs