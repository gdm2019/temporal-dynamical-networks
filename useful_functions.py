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
    g.reindex_edges()
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
    return df, dataset_name

def df_threshold(df, link_threshold):
    '''It takes a Pandas dataframe with keys timestamp and edge.
    It returns a filter dataset, where edges that have less than link_threshold occurrences in the original dataframe are taken out.'''
    
    edge_counts = df['edge'].value_counts()
    edges_to_keep = edge_counts[edge_counts >= link_threshold].index
    filtered_df = df[df['edge'].isin(edges_to_keep)]

    return filtered_df

def get_time(timestamp):
    '''Assumes that the input timestamp is expressed in seconds.
    It returns the timestamp in the appriate time unit.'''

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
    '''t takes a pandas dataframe with keys timestamp, sender, recipient.
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
    '''It takes the Pandas dataframe with keys edge and timestamp.
    It returns the list of inter-event times for each unique undirected edge in the dataset.'''

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
    '''It computes tmin, tmax, period of the dataset, the number of nodes, for each of them its aggregateed degree, the total number of events, the edge interevent times and the system interevent times.'''

    stuff = []
    if scalar_true == True:
        times = df['timestamp'].astype(float).astype(int)
        tmax = np.max(times)
        tmin = np.min(times)
        num_nodes = len(pd.concat([df['sender'],df['recipient']]).unique())
        num_events = len(df)
        period = get_time(tmax-tmin)
        stuff.append([get_time(tmax), get_time(tmin), num_nodes, num_events, period])

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