import numpy as np
import matplotlib.pylab as plt
import pickle
import graph_tool.all as gt
import pandas as pd
import statistics as st
import random
from scipy import stats
from scipy.special import gamma, factorial
from scipy.optimize import curve_fit

def create_dataset_file(dataset_name):
    '''Takes the Netzschleuder repository dataset name of the dataset as an input.
    It creates a Pandas dataframe with timestamps, senders, recipients and undirected edges of the events. 
    The dataframe is saved as a pkl file.'''
    df_before, dataset_name = get_g_df(dataset_name, 'undirected')

    cleaned_name = dataset_name.replace("/", "_")
    df_before.to_pickle(f'{cleaned_name}_pandasdf.pkl')

    threshold = 10
    df = df_threshold(df_before, threshold)
    df.to_pickle(f'{cleaned_name}_pandasdf_filtered{threshold}.pkl')

def get_g_df(dataset_name):
    '''Takes the Netzschleuder repository dataset name as an input
    Returns the Pandas dataframe with keys 'timestamp', 'sender', 'recipient', as well as 'edge'.'''
    g = gt.collection.ns[dataset_name]
    # g.reindex_edges()
    print('dataset_name:', dataset_name)

    isolated_nodes = [v for v in g.vertices() if v.out_degree() == 0 and v.in_degree() == 0]
    g.remove_vertex(isolated_nodes, fast=True)
    g.reindex_edges()

    sources = []
    recipients = []
    timestamps = []
    edge = []

    for e in g.edges():
        n = int(e.source())
        x = int(e.target())
        edge.append((min(n, x), max(n, x)))
        sources.append(n)
        recipients.append(x)
        timestamps.append(g.ep.time[e])
    timestamps = list(map(lambda x: int(float(x)), timestamps))
    d = {'timestamp': timestamps, 'edge': edge, 'sender': sources, 'recipient': recipients}

    df = pd.DataFrame(data=d)
    return df

def df_threshold(df, link_threshold):
    '''Takes a Pandas dataframe with keys timestamp and edge.
    Returns a filter dataset, where edges that have less than link_threshold occurrences in the original dataframe are taken out.'''
    
    edge_counts = df['edge'].value_counts()
    edges_to_keep = edge_counts[edge_counts >= link_threshold].index
    filtered_df = df[df['edge'].isin(edges_to_keep)]

    return filtered_df

def get_time(timestamp):
    '''Assumes that the input timestamp is expressed in seconds.
    Returns the timestamp in the appriate time unit.'''

    if timestamp<60:
        per = timestamp
        unit = f'{per:.1f} s'
    elif timestamp < 60*60:
        per = timestamp/60
        unit = f'{per:.1f} min'
    elif timestamp < 60*60*24:
        per = timestamp/(60*60)
        unit = f'{per:.1f} h'
    elif timestamp < 60*60*24*7:
        per = timestamp/(60*60*24)
        unit = f'{per:.1f} d'
    elif timestamp < 60*60*24*7*4:
        per = timestamp/(60*60*24*7)
        unit = f'{per:.1f} w'
    elif timestamp < 60*60*24*7*4*12:
        per = timestamp/(60*60*24*7*4)
        unit = f'{per:.1f} m'
    else:
        per = timestamp/(60*60*24*7*4*12)
        unit = f'{per:.1f} y'  
    return unit

def compute_unique_neighbors(df_before):
    '''Takes a pandas dataframe with keys timestamp, sender, recipient.
    For each unique node in the dataset, it returns the count of unique neighbour it has in the whole timeperiod covered by the data.'''
    
    df = df_before.drop_duplicates(subset=['edge'], keep='first')
    senders = df['sender'].unique()
    neighbours = {}
    iter = 0
    for v in senders:
        aux = df[df['sender'] == v]
        if v not in neighbours:
            neighbours[v] = []
        neighbours[v].append(list(aux['recipient'].unique()))
        for rec in list(aux['recipient'].unique()):
            if rec not in neighbours:
                neighbours[rec] = []       
            neighbours[rec].append([v])  
        iter += 1    

    unique_neighbours = {}
    for keys in neighbours.keys():
        unique_neighbours[keys] = len(np.unique(np.concatenate(neighbours[keys])))
    return unique_neighbours

def undirected_link_interevent_edge(df): 
    '''Takes the Pandas dataframe with keys edge and timestamp.
    Returns the list of inter-event times for each unique undirected edge in the dataset.'''

    my_dict = {}
    for _, row in df.iterrows():
        edge = row['edge']
        time = row['timestamp']
        if edge not in my_dict:
            my_dict[edge] = []
        my_dict[edge].append(time)

    interevent = {}
    for edge in my_dict.keys():
        a = sorted(np.unique(my_dict[edge]))
        interevent[edge] = [j - i for i, j in zip(a[:-1], a[1:])]

    return interevent

def get_quantities(df, scalar_true, degree_true, interevent_true):
    '''Computes tmin, tmax, period of the dataset, the number of nodes, for each of them its aggregateed degree, the total number of events, the edge interevent times and the system interevent times.'''

    stuff = []
    if scalar_true == True:
        times = df['timestamp'].astype(float).astype(int)
        tmax = np.max(times)
        tmin = np.min(times)
        num_nodes = len(pd.concat([df['sender'],df['recipient']]).unique())
        num_events = len(df)
        period = get_time(tmax-tmin)
        stuff.append([tmax, tmin, num_nodes, num_events, period])

    if degree_true == True:
        nodes = pd.concat([df['sender'],df['recipient']]).unique()
        edges = df['edge'].drop_duplicates()
        agg_degree = compute_unique_neighbors(df)
        stuff.append([nodes, edges, agg_degree])

    if interevent_true == True:
        interevent = undirected_link_interevent_edge(df)
        agg_interevent_tot = np.concatenate(list(interevent.values()))
        stuff.append([interevent, agg_interevent_tot])
    return stuff

def generate_log_bins(min_value, max_value, multiplier):
    '''INPUTS: minimum and maximum of the coordinate over which the binning is wished to be performed, as well as a multiplier for the increment of the size of the bins. For example, multiplier = 2.5.
    OUTPUT = array containing the log bins.'''
    bins = [min_value]
    cur_value = bins[0]
    while cur_value < max_value:
        if cur_value * multiplier < max_value: cur_value = cur_value * multiplier
        else: cur_value = max_value
        bins.append(cur_value)
    return np.array(bins)

def plot_ccdf(data, ax, label=None, col=None):
    """
    Plots on the axes object the complementary cumulative distribution function
    (1-CDF(x)) based on the raw data.
    """
    sorted_vals = np.sort(np.unique(data))
    ccdf = np.zeros(len(sorted_vals))
    n = float(len(data))
    for i, val in enumerate(sorted_vals):
        ccdf[i] = np.sum(data >= val)/n
    ax.plot(sorted_vals, ccdf, "-", linewidth=4, label=label, color= col)

def lin_func(x, a, b):
    y = []
    for coor in x:
        y.append(a * coor + b)
    return y

def get_spectrum(dataset_name, df, MAX_windows, widths_ref, time_shifts_ref):
    [tmax, tmin, _, _, _], [nodes, edges, agg_degree], [interevent, agg_interevent_tot] = get_quantities(df, True, True, True)

    mean_tau = np.mean(agg_interevent_tot)
    ref = int(mean_tau)
    widths = ref*np.array(widths_ref)
    widths = widths.astype(int)
    time_shifts = ref*np.array(time_shifts_ref)
    time_shifts = time_shifts.astype(int)

    colors = ['navy', 'dodgerblue', 'darkorange']

    fig0, ax0 = plt.subplots(figsize=(10, 6))
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig4, ax4 = plt.subplots(figsize=(10, 6)) #

    for iter_windows in range(len(widths)):

        width = widths[iter_windows]
        time_shift = time_shifts[iter_windows]
        chosen_color = colors[iter_windows]

        if width >= time_shift: 
            number_windows = int((tmax-tmin)/(time_shift) - (width-time_shift)/time_shift)
            print('number_windows:', number_windows)

            if number_windows >= 2: 
                activity = {}
                delta_activity = {}
                dynamic_degree = {}
                index = 0
                end_t = tmin
                while end_t <= tmax and index <= MAX_windows:
                    start_t = tmin + time_shift*index
                    end_t = start_t + width
                    # I should be able to transform activity, delta_activity as 2d arrays.
                    df_window = df[(df['timestamp'] >= start_t) & (df['timestamp'] < end_t)]
                    dynamic_degree[index] = compute_unique_neighbors(df_window)
                    for edge in edges:
                        if edge in list(df_window['edge'].drop_duplicates()):
                            activity[edge, end_t] = 1
                        else:
                            activity[edge, end_t] = 0
                        if index > 0:
                            delta_activity[edge, end_t] = activity[edge, end_t] -  activity[edge, end_t - time_shift]
                    index += 1

                vector = {}
                for dict in dynamic_degree.values():
                    for individual, its_degree in dict.items():
                        # print(len(its_degree))
                        # mean_dynamic_degree[individual] = np.mean(its_degree)
                        if individual not in vector: #
                            vector[individual] = [] #
                        vector[individual].append(its_degree) #

                mean_dynamic_degree = {} #
                mean_dynamic_degree_static = {} #
                for ind in vector.keys(): #
                    mean_dynamic_degree[ind] = np.mean(vector[ind]) #
                    if agg_degree[ind] not in mean_dynamic_degree_static: #
                        mean_dynamic_degree_static[agg_degree[ind]] = [] #
                    mean_dynamic_degree_static[agg_degree[ind]].append(np.mean(vector[ind])) #
                times = np.unique([time for _, time in activity.keys()])
                n_active = {}
                delta_n_active = {}
                delta_n_active_static = {}
                delta_n_active_mean_dynamic = {}

                for ind in nodes: 
                    lines = df[(df['sender'] == ind) | (df['recipient'] == ind)]
                    aux = list(lines['edge'].drop_duplicates())
                    iter = 0
                    for time_point in times:
                        n_active[ind, time_point] = np.sum([activity[e, time_point] == 1 for e in aux]) #np.sum(activity[aux, time_point] == 1)
                        if iter > 0:
                            delta_k = n_active[ind, time_point] - n_active[ind, time_point - time_shift]
                            if n_active[ind, time_point - time_shift] not in delta_n_active:
                                delta_n_active[n_active[ind, time_point - time_shift]] = []
                            delta_n_active[n_active[ind, time_point - time_shift]].append(delta_k)
                            if agg_degree[ind] not in delta_n_active_static:
                                delta_n_active_static[agg_degree[ind]] = []
                            delta_n_active_static[agg_degree[ind]].append(delta_k)
                            if mean_dynamic_degree[ind] not in delta_n_active_mean_dynamic:
                                delta_n_active_mean_dynamic[mean_dynamic_degree[ind]] = []
                            delta_n_active_mean_dynamic[mean_dynamic_degree[ind]].append(delta_k)
                        iter += 1

                all_deltaks = np.concatenate(list(delta_n_active.values()))
                x, y = np.unique(all_deltaks, return_counts = True)
                ax0.bar(x, y/sum(y), alpha = 0.2, color = chosen_color, edgecolor = chosen_color, label = f'eta = {width/int(mean_tau):.2f} <tau>, nu = {(width-time_shift)/width:.2f}')
                
                meandeltak = {}
                for once, k in enumerate(sorted(delta_n_active.keys())):
                    if once == 0: 
                        ax1.scatter([k] * len(delta_n_active[k]), delta_n_active[k], color = chosen_color, alpha = 0.1, s = 10)
                    else: ax1.scatter([k] * len(delta_n_active[k]), delta_n_active[k], color = chosen_color, alpha = 0.1, s = 10)
                    meandeltak[k] = np.mean(delta_n_active[k])
                ks = list(meandeltak.keys())
                if len(ks) > 1: 
                    (a_plus, b_plus), _ = curve_fit(lin_func, ks, list(meandeltak.values()))
                    mean_deltak_fit = lin_func(ks, a_plus, b_plus)
                    ax1.plot(ks, mean_deltak_fit, linewidth = 4, color=chosen_color, label = f'eta = {width/int(mean_tau):.2f} <tau>, nu = {(width-time_shift)/width:.2f}')
                ax1.axvline(x = np.mean(list(mean_dynamic_degree.values())), linestyle = '--', color=chosen_color, label = f'mean dyn. degree')
 
                meandeltak = {}
                for once, k in enumerate(sorted(delta_n_active_static.keys())):
                    if once == 0: 
                        ax2.scatter([k] * len(delta_n_active_static[k]), delta_n_active_static[k], color = chosen_color, alpha = 0.1, s = 10)
                    else: ax2.scatter([k] * len(delta_n_active_static[k]), delta_n_active_static[k], color = chosen_color, alpha = 0.1, s = 10)
                    meandeltak[k] = np.mean(delta_n_active_static[k])
                ks = list(meandeltak.keys())
                if len(ks) > 1: 
                    (a_plus, b_plus), _ = curve_fit(lin_func, ks, list(meandeltak.values()))
                    mean_deltak_fit = lin_func(ks, a_plus, b_plus)
                    ax2.plot(ks, mean_deltak_fit, linewidth = 4, color=chosen_color, label = f'eta = {width/int(mean_tau):.2f} <tau>, nu = {(width-time_shift)/width:.2f}')
                xlim = ax2.get_xlim()
                x_bisector2 = np.linspace(xlim[0], xlim[1], 100)
                y_bisector2 = -x_bisector2

                meandeltak = {}
                for once, k in enumerate(sorted(delta_n_active_mean_dynamic.keys())):
                    if once == 0: 
                        ax3.scatter([k] * len(delta_n_active_mean_dynamic[k]), delta_n_active_mean_dynamic[k], color = chosen_color, alpha = 0.1, s = 10)
                    else: ax3.scatter([k] * len(delta_n_active_mean_dynamic[k]), delta_n_active_mean_dynamic[k], color = chosen_color, alpha = 0.1, s = 10)
                    meandeltak[k] = np.mean(delta_n_active_mean_dynamic[k])
                ks = list(meandeltak.keys())
                if len(ks) > 1: 
                    (a_plus, b_plus), _ = curve_fit(lin_func, ks, list(meandeltak.values()))
                    mean_deltak_fit = lin_func(ks, a_plus, b_plus)
                    ax3.plot(ks, mean_deltak_fit, linewidth = 4, color=chosen_color, label = f'eta = {width/int(mean_tau):.2f} <tau>, nu = {(width-time_shift)/width:.2f}')

                for once, static_k in enumerate(list(mean_dynamic_degree_static.keys())): #
                    if once == 0:
                        ax4.scatter([static_k]*len(list(mean_dynamic_degree_static[static_k])), list(mean_dynamic_degree_static[static_k]), alpha = 0.4, color = chosen_color, label = f'eta = {width/int(mean_tau):.2f} <tau>, nu = {(width-time_shift)/width:.2f}') #
                    else:
                        ax4.scatter([static_k]*len(list(mean_dynamic_degree_static[static_k])), list(mean_dynamic_degree_static[static_k]), alpha = 0.4, color = chosen_color) #
                    xlim = ax4.get_xlim() #
                    x_bisector3 = np.linspace(xlim[0], xlim[1], 100) #
    ax0.set_yscale('log')
    ax0.axvline(x = 0, linewidth = 1, color='black')
    ax0.set_ylabel('Probability', fontsize = 25)
    ax0.legend(fontsize = 18)
    ax0.set_xlabel('degree change', fontsize = 25)
    ax0.set_title(f'{dataset_name}', fontsize = 22)

    ax1.axhline(y = 0, color='k')
    ax1.set_ylabel('Degree change', fontsize = 23)
    ax1.legend(fontsize = 18)
    ax1.set_xlabel('dynamic degree', fontsize = 25)
    ax1.plot(x_bisector2, y_bisector2, color='black', label='y = -x')
    ax1.set_title(f'{dataset_name}', fontsize = 22)

    ax2.axhline(y = 0, color='k')
    ax2.set_ylabel('Degree change', fontsize = 23)
    ax2.legend(fontsize = 18)
    ax2.set_xlabel('static degree k', fontsize = 25)
    ax2.plot(x_bisector2, y_bisector2, color='black', label='y = -x')
    ax2.set_title(f'{dataset_name}', fontsize = 22)

    ax3.axhline(y = 0, color='k')
    ax3.set_ylabel('Degree change', fontsize = 23)
    ax3.legend(fontsize = 18)
    ax3.set_xlabel('mean individual dynamic degree', fontsize = 25)
    ax3.plot(x_bisector2, y_bisector2, color='black', label='y = -x')
    ax3.set_title(f'{dataset_name}', fontsize = 22)

    ax4.plot(x_bisector3, x_bisector3, color='black', label='y = x') #
    ax4.axhline(y=0, color='k') #
    ax4.set_ylabel('mean individual dynamic degree', fontsize = 23) #
    ax4.legend(fontsize = 18) #
    ax4.set_xlabel('static degree', fontsize = 25) #
    ax4.set_title(f'{dataset_name}', fontsize = 22) #

# def get_spectrum(dataset_name, df, MAX_windows, widths_ref, time_shifts_ref):

#     '''INPUTS: dataset_name is the string of the Netzschleuder dataset name, df is the Pandas dataframe with keys timestamp, sender, recipiend and edge, MAX_windows is an upper cutoff in the number of temporal windows that it runs the code for, widths_ref is the vector of factors of the reference mean inter-event time for the width of the memory window, time_shifts_ref is the vector (same length as widths_ref) with the factors of the reference inter-event time.
#     OUTPUTS: Plots the frequency of degree change, the degree change as a function of dynamic degree, the degree change as a function of the sattic degree and the degree change as a function of the mean individual dynamic degree.'''
#     [tmax, tmin, _, _, _], [nodes, edges, agg_degree], [interevent, agg_interevent_tot] = get_quantities(df, True, True, True)

#     mean_tau = np.mean(agg_interevent_tot)
#     ref = int(mean_tau)
#     widths = ref*np.array(widths_ref)
#     widths = widths.astype(int)
#     time_shifts = ref*np.array(time_shifts_ref)
#     time_shifts = time_shifts.astype(int)

#     colors = ['navy', 'dodgerblue', 'darkorange']

#     fig0, ax0 = plt.subplots(figsize=(10, 6))
#     fig1, ax1 = plt.subplots(figsize=(10, 6))
#     fig2, ax2 = plt.subplots(figsize=(10, 6))
#     fig3, ax3 = plt.subplots(figsize=(10, 6))
#     fig4, ax4 = plt.subplots(figsize=(10, 6))

#     for iter_windows in range(len(widths)):

#         width = widths[iter_windows]
#         time_shift = time_shifts[iter_windows]
#         chosen_color = colors[iter_windows]

#         if width >= time_shift: 
#             number_windows = int((tmax-tmin)/(time_shift) - (width-time_shift)/time_shift)
#             print('number_windows:', number_windows)

#             if number_windows >= 2: 
#                 activity = {}
#                 delta_activity = {}
#                 dynamic_degree = {}
#                 index = 0
#                 end_t = tmin
#                 while end_t <= tmax and index <= MAX_windows:
#                     start_t = tmin + time_shift*index
#                     end_t = start_t + width
#                     # I should be able to transform activity, delta_activity as 2d arrays.
#                     df_window = df[(df['timestamp'] >= start_t) & (df['timestamp'] < end_t)]
#                     for edge in edges:
#                         if edge in list(df_window['edge'].drop_duplicates()):
#                             activity[edge, end_t] = 1
#                         else:
#                             activity[edge, end_t] = 0
#                         if index > 0:
#                             delta_activity[edge, end_t] = activity[edge, end_t] -  activity[edge, end_t - time_shift]
#                     index += 1

#                 times = np.unique([time for _, time in activity.keys()])
#                 n_active = {}
#                 delta_n_active = {}
#                 delta_n_active_static = {}
#                 delta_n_active_mean_dynamic = {}
#                 mean_dynamic_degree = {}
#                 mean_dynamic_degree_static = {}
#                 for ind in nodes: 
#                     lines = df[(df['sender'] == ind) | (df['recipient'] == ind)]
#                     aux = list(lines['edge'].drop_duplicates())
#                     iter = 0
#                     for time_point in times:
#                         n_active[ind, time_point] = np.sum([activity[e, time_point] == 1 for e in aux]) #np.sum(activity[aux, time_point] == 1)
#                         if iter > 0:
#                             delta_k = n_active[ind, time_point] - n_active[ind, time_point - time_shift]
#                             if n_active[ind, time_point - time_shift] not in delta_n_active:
#                                 delta_n_active[n_active[ind, time_point - time_shift]] = []
#                             delta_n_active[n_active[ind, time_point - time_shift]].append(delta_k)
#                             if agg_degree[ind] not in delta_n_active_static:
#                                 delta_n_active_static[agg_degree[ind]] = []
#                             delta_n_active_static[agg_degree[ind]].append(delta_k)
#                         iter += 1

#                     mean_dynamic_degree[ind] = np.mean([n_active[ind, time_point] for time_point in times])
#                     if agg_degree[ind] not in mean_dynamic_degree_static:
#                         mean_dynamic_degree_static[agg_degree[ind]] = []
#                     mean_dynamic_degree_static[agg_degree[ind]].append(np.mean([n_active[ind, time_point] for time_point in times]))
#                     iter = 0
#                     for time_point in times:
#                         n_active[ind, time_point] = np.sum([activity[e, time_point] == 1 for e in aux]) #np.sum(activity[aux, time_point] == 1)
#                         if iter > 0:   
#                             delta_k = n_active[ind, time_point] - n_active[ind, time_point - time_shift]                 
#                             if mean_dynamic_degree[ind] not in delta_n_active_mean_dynamic:
#                                 delta_n_active_mean_dynamic[mean_dynamic_degree[ind]] = []
#                             delta_n_active_mean_dynamic[mean_dynamic_degree[ind]].append(delta_k)
#                         iter += 1

#                 all_deltaks = np.concatenate(list(delta_n_active.values()))
#                 x, y = np.unique(all_deltaks, return_counts = True)
#                 ax0.bar(x, y/sum(y), alpha = 0.2, color = chosen_color, edgecolor = chosen_color, label = f'eta = {width/int(mean_tau):.2f} <tau>, nu = {(width-time_shift)/width:.2f}')
                
#                 meandeltak = {}
#                 for once, k in enumerate(sorted(delta_n_active.keys())):
#                     if once == 0: 
#                         ax1.scatter([k] * len(delta_n_active[k]), delta_n_active[k], color = chosen_color, alpha = 0.1, s = 10)
#                     else: ax1.scatter([k] * len(delta_n_active[k]), delta_n_active[k], color = chosen_color, alpha = 0.1, s = 10)
#                     meandeltak[k] = np.mean(delta_n_active[k])
#                 ks = list(meandeltak.keys())
#                 if len(ks) > 1: 
#                     (a_plus, b_plus), _ = curve_fit(lin_func, ks, list(meandeltak.values()))
#                     mean_deltak_fit = lin_func(ks, a_plus, b_plus)
#                     ax1.plot(ks, mean_deltak_fit, linewidth = 4, color=chosen_color, label = f'eta = {width/int(mean_tau):.2f} <tau>, nu = {(width-time_shift)/width:.2f}')
#                 ax1.axvline(x = np.mean(list(mean_dynamic_degree.values())), linestyle = '--', color=chosen_color, label = f'mean dyn. degree')
 
#                 meandeltak = {}
#                 for once, k in enumerate(sorted(delta_n_active_static.keys())):
#                     if once == 0: 
#                         ax2.scatter([k] * len(delta_n_active_static[k]), delta_n_active_static[k], color = chosen_color, alpha = 0.1, s = 10)
#                     else: ax2.scatter([k] * len(delta_n_active_static[k]), delta_n_active_static[k], color = chosen_color, alpha = 0.1, s = 10)
#                     meandeltak[k] = np.mean(delta_n_active_static[k])
#                 ks = list(meandeltak.keys())
#                 if len(ks) > 1: 
#                     (a_plus, b_plus), _ = curve_fit(lin_func, ks, list(meandeltak.values()))
#                     mean_deltak_fit = lin_func(ks, a_plus, b_plus)
#                     ax2.plot(ks, mean_deltak_fit, linewidth = 4, color=chosen_color, label = f'eta = {width/int(mean_tau):.2f} <tau>, nu = {(width-time_shift)/width:.2f}')
#                 xlim = ax2.get_xlim()
#                 x_bisector2 = np.linspace(xlim[0], xlim[1], 100)
#                 y_bisector2 = -x_bisector2

#                 meandeltak = {}
#                 for once, k in enumerate(sorted(delta_n_active_mean_dynamic.keys())):
#                     if once == 0: 
#                         ax3.scatter([k] * len(delta_n_active_mean_dynamic[k]), delta_n_active_mean_dynamic[k], color = chosen_color, alpha = 0.1, s = 10)
#                     else: ax3.scatter([k] * len(delta_n_active_mean_dynamic[k]), delta_n_active_mean_dynamic[k], color = chosen_color, alpha = 0.1, s = 10)
#                     meandeltak[k] = np.mean(delta_n_active_mean_dynamic[k])
#                 ks = list(meandeltak.keys())
#                 if len(ks) > 1: 
#                     (a_plus, b_plus), _ = curve_fit(lin_func, ks, list(meandeltak.values()))
#                     mean_deltak_fit = lin_func(ks, a_plus, b_plus)
#                     ax3.plot(ks, mean_deltak_fit, linewidth = 4, color=chosen_color, label = f'eta = {width/int(mean_tau):.2f} <tau>, nu = {(width-time_shift)/width:.2f}')

#                 for static_k in list(mean_dynamic_degree_static.keys()):
#                     ax4.scatter([static_k]*len(list(mean_dynamic_degree_static[static_k])), list(mean_dynamic_degree_static[static_k]), alpha = 0.4, color = chosen_color, label = f'eta = {width/int(mean_tau):.2f} <tau>, nu = {(width-time_shift)/width:.2f}')
#                     xlim = ax4.get_xlim()
#                     x_bisector3 = np.linspace(xlim[0], xlim[1], 100)

#     ax0.set_yscale('log')
#     ax0.axvline(x = 0, linewidth = 1, color='black')
#     ax0.set_ylabel('Probability', fontsize = 25)
#     ax0.legend(fontsize = 18)
#     ax0.set_xlabel('degree change', fontsize = 25)
#     ax0.set_title(f'{dataset_name}', fontsize = 22)

#     ax1.axhline(y = 0, color='k')
#     ax1.set_ylabel('Degree change', fontsize = 23)
#     ax1.legend(fontsize = 18)
#     ax1.set_xlabel('dynamic degree', fontsize = 25)
#     ax1.plot(x_bisector2, y_bisector2, color='black', label='y = -x')
#     ax1.set_title(f'{dataset_name}', fontsize = 22)

#     ax2.axhline(y = 0, color='k')
#     ax2.set_ylabel('Degree change', fontsize = 23)
#     ax2.legend(fontsize = 18)
#     ax2.set_xlabel('static degree k', fontsize = 25)
#     ax2.plot(x_bisector2, y_bisector2, color='black', label='y = -x')
#     ax2.set_title(f'{dataset_name}', fontsize = 22)

#     ax3.axhline(y = 0, color='k')
#     ax3.set_ylabel('Degree change', fontsize = 23)
#     ax3.legend(fontsize = 18)
#     ax3.set_xlabel('mean individual dynamic degree', fontsize = 25)
#     ax3.plot(x_bisector2, y_bisector2, color='black', label='y = -x')
#     ax3.set_title(f'{dataset_name}', fontsize = 22)

#     ax4.plot(x_bisector3, x_bisector3, color='black', label='y = x')
#     ax4.axhline(y=0, color='k')
#     ax4.set_ylabel('mean individual dynamic degree', fontsize = 23)
#     ax4.legend(fontsize = 18)
#     ax4.set_xlabel('static degree', fontsize = 25)
#     ax4.set_title(f'{dataset_name}', fontsize = 22)