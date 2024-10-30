# temporal-dynamical-networks
We study the stability of dynamical networks, exploring the wide range of aggregation windows taking us from the temporal network limit to its static limit.

The 45 empirical datasets files that we use could not be uploaded as their size was too big. 

We provide the code to create the files in 'create_dataset_file.py'.
Datasets are saved as Pandas dataframes. 
Every line corresponds to a connection event between two nodes. The time of the connection is saved with the key 'timestamp', the sender node id is saved with the key 'sender', its recipient id with the key 'recipient' and the undirected edge (min(sender, recipient), max(sender, recipient)) corresponding to the event with the key 'edge'.

