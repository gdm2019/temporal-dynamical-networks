import os
os.environ["PATH"] = "/Library/TeX/texbin:/opt/homebrew/bin:" + os.environ["PATH"]

import numpy as np
import graph_tool.all as gt

from functions_library import *
from functions_Figures_main import *

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


def get_system_burstiness(system_interevents):
    '''Takes the list of IETs and computes the burstiness index (adjusting for system size)'''
    n = len(system_interevents)
    if n >= 1:
        r = np.std(system_interevents)/np.mean(system_interevents)
        B_s = compute_burstiness(r, n)
    return B_s

def flatten_dict_and_concatenate(my_dict):
    '''Takes a dictionary and flattens and concatenates its values across keys'''
    return np.concatenate(list(my_dict.values()))

def flatten_dict(my_dict):
    '''Takes a dictionary and flattens its values across keys'''
    return list(my_dict.values())

def get_length_valueslist_dict(my_dict):
    '''Transforms a dictionary with lists of values associated to each key into the scalar len(list) for each key'''
    weight_dict = defaultdict(int)
    for e, my_list in my_dict.items():
        weight_dict[e] = len(my_list)
    return weight_dict

def get_log_binned(x_values, y_values, num_bins = 20):
    '''Provides the log binned x and y average data values'''
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    mask = x_values > 0
    x_values = x_values[mask]
    y_values = y_values[mask]

    bins = np.logspace(np.log10(x_values.min()), np.log10(x_values.max()), num_bins)

    indices = np.digitize(x_values, bins)

    x_binned = []
    y_binned = []

    for i in range(1, len(bins)):
        mask = indices == i  
        if np.any(mask):     
            x_binned.append(np.mean(x_values[mask]))
            y_binned.append(np.mean(y_values[mask]))

    x_binned = np.array(x_binned)
    y_binned = np.array(y_binned)
    return x_binned, y_binned

def collapse_dictoflists(my_dict):
    '''Transforms a dictionary with lists of values associated to each key into a single list'''
    output_list = []
    for my_list in my_dict.values():
        if len(my_list) > 0:
            output_list.extend(my_list)
    return output_list

def sort_data(x, y):
    '''sorts a pairs of data'''
    pair = sorted(zip(x, y))
    sorted_etas, sorted_pplus = zip(*pair)

    return sorted_etas, sorted_pplus

def read_two_col_data(filename_activation):
    '''Opens a file and extracts the two columns data'''
    with gzip.open(filename_activation, "rt") as f:
        data_loaded = np.loadtxt(f)

    x = data_loaded[:, 0]
    y = data_loaded[:, 1]
    return x, y

def read_Pplus_eta_data(filename):
    '''Opens a file, extracts the two columns data and sorts them'''
    eta, Pplus = np.loadtxt(filename, delimiter=",", unpack=True)

    pair = sorted(zip(eta, Pplus))
    sorted_etas, sorted_pplus = zip(*pair)
    return sorted_etas, sorted_pplus

def get_chosen_eta(which, new_name, overlap, iter_eta, MAX_windows = 1000): 
    '''Chooses a window width either at the node or edge level'''
    if which == 'node': # node level activation probability
        activation_filename_mean_edge = f"./data/{new_name}/{new_name}_node_Activation_probability_overlap{overlap}_MAXwindows{MAX_windows}.txt"
        xx, yy = read_Pplus_eta_data(activation_filename_mean_edge) 
        xx, yy = sort_data(xx, yy)
        xx = np.array(xx)
        yy = np.array(yy)

        index_eta = np.argmax(yy)
        eta = xx[index_eta]
    if which == 'edge':
        activation_filename_mean_edge = f"./data/{new_name}/{new_name}_edge_Activation_probability_overlap{overlap}_MAXwindows{MAX_windows}.txt"
        xx, yy = read_Pplus_eta_data(activation_filename_mean_edge)
        xx, yy = sort_data(xx, yy)
        xx = np.array(xx)
        yy = np.array(yy)

        index_eta = np.argmax(yy)
        eta = xx[index_eta]
    return eta

def plot_single(x_values, y_values, y_errors_data, my_color, label, alpha, ax, lw):
    '''plot of a curve with the standard deviation as a shaded area'''
    ax.plot(x_values, y_values, color = my_color, label = label, lw = lw)
    ax.fill_between(x_values, [max(0, y - e) for y, e in zip(y_values, y_errors_data)], [y + (2 * e if y - e < 0 else e) for y, e in zip(y_values, y_errors_data)], color = my_color, alpha=alpha)

def plot1(ax, degree, degree_change, color):
    '''scatter plot of raw datapoints and line plot of quantile binned averages'''
    ax.scatter(degree, degree_change, color = color, alpha=0.2, rasterized=True)
    bin_degree, bin_degree_change = get_quantile_binned(degree, degree_change, num_bins = 10)
    ax.plot(bin_degree, bin_degree_change, color = 'k', lw = 3)

def include_layout(x_label, y_label, iter_data, ax, cols, rows, legend = 'on'): 
    '''layout of maxi all datasets figures'''
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

def include_layout_special(x_label, y_label, iter_data, ax, cols, rows): 
    '''layout of maxi all datasets figures for a specific case'''
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
    