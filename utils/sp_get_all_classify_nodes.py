import random

import networkx as nx
import pandas as pd

df = pd.read_csv('./data/Edge_1000.txt', sep=' ', names=['s', 't', 'weight'])
# print(df)
df.s = df.s.astype(int)
df.t = df.t.astype(int)
total_node_list = list(set(df.s.values.tolist() + df.t.values.tolist()))
weights_list = df.values.tolist()
# G = nx.Graph()
# G.add_edges_from()
G = nx.Graph()
G.add_weighted_edges_from(weights_list)
# print(G.edges)
all_spp_dist = nx.all_pairs_shortest_path(G)

# all_spp_dist = nx.all_pairs_dijkstra_path_length(G)
all_spp_dist_list = list(all_spp_dist)
print(all_spp_dist_list[0])
print(all_spp_dist_list[1])

all_samples_list = []
# edges_list = []
for s, t_dict in all_spp_dist_list:
    for t, w in t_dict.items():
        if len(w)<3:
            continue
        else:
            assert s==w[0] and t==w[-1], 'error in matching'
            pos_samples = []
            neg_samples = []
            for i in w[1:-1]:
                pos_samples.append([s,t,i,1])
            neg_candis_all = [j for j in total_node_list if j not in w]
            neg_sample_nodes = random.sample(neg_candis_all, 1*len(w[1:-1]))
            neg_samples = [[s,t,k,0] for k in neg_sample_nodes]
            all_samples_list += pos_samples
            all_samples_list += neg_samples


samples_df = pd.DataFrame(all_samples_list, columns=['s', 't', 'c', 'label'])
print('samples_df shape', samples_df.shape)
samples_df.to_csv('./data/whole_node_pair_samples.csv', index=False, header=True)








