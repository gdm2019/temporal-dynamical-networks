# temporal-dynamical-networks
We study the stability of dynamical networks, exploring the wide range of aggregation windows taking us from the temporal network limit to its static limit.

The 45 empirical datasets that we use can be found in the folder 'datasets'. 
They are saved as Pandas dataframes. 
Every line corresponds to an connection event between two nodes. The time of the connection is saved with the key 'timestamp', the sender node id is saved with the key 'sender', its recipient id with the key 'recipient' and the undirected edge (min(sender, recipient), max(sender, recipient)) corresponding to the event with the key 'edge'.

