# temporal-dynamical-networks
We study the stability of dynamical networks, exploring the wide range of aggregation windows going from the temporal network limit to the static one.

We identified 45 datasets in the Netzschleuder online open repository that had a 'time' edge property.

In our analysis the data is used in the form of Pandas dataframes. 
Every line corresponds to a connection event between two nodes. The time of the connection is saved with the key 'timestamp', the sender node id is saved with the key 'sender', its recipient id with the key 'recipient' and the undirected edge (min(sender, recipient), max(sender, recipient)) corresponding to the event with the key 'edge'.

The 45 dataframe files that we use for the analysis could however not be uploaded as their size was too big. 
Please use the function create_dataset_file(dataset_name) in 'useful_functions', to create this file.
