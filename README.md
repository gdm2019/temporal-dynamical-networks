# temporal-dynamical-networks
We study the stability of dynamical networks, exploring the wide range of aggregation windows going from the temporal network limit to the static one.

We identified 45 datasets in the Netzschleuder online open repository that had a 'time' edge property. We treat them using Python.

In our analysis the data is used in the form of Pandas dataframes. 
Every line corresponds to a connection event between two nodes. The time of the connection is saved with the key 'timestamp', the sender node id is saved with the key 'sender', its recipient id with the key 'recipient' and the undirected edge (min(sender, recipient), max(sender, recipient)) corresponding to the event with the key 'edge'.

The 45 dataframe files that we use for the analysis could not be uploaded as their size is too big. 
Please use the function create_dataset_file(dataset_name) in 'useful_functions' if you wish to create such files.

A custom library was created that includes all functions required for the analysis. The library can be imported with the following command:
import useful_functions as uf

In order to plot the degree change frequency and the degree change spectrum plots (as functions of either dynamic degree, mean dynamic degree or static degree) run the following, after importing the custom library:

dataset_name = 'contact'

df_before = uf.get_g_df(dataset_name)

df = uf.df_threshold(df_before, 10)

MAX_windows = 2000

widths_ref = [1, 1, 5]

time_shifts_ref = [0.2, 1, 2]

uf.get_spectrum(dataset_name, df, MAX_windows, widths_ref, time_shifts_ref)
