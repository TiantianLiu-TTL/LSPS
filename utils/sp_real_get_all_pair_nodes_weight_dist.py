import networkx as nx
import pandas as pd


dataset_name = "road-NA"

if dataset_name == "brain":
    edge_num = "503"
elif dataset_name == "bio-SC-LC":
    edge_num = "1999"
elif dataset_name == "inf-power":
    edge_num = "4941"
elif dataset_name == "road-NA":
    edge_num = "175813"
elif dataset_name == "web-EPA":
    edge_num = "4253"

df = pd.read_csv('../data/{dataset}/Edge_{edge_num}.txt'.format(dataset=dataset_name, edge_num=edge_num), sep=' ', names=['s', 't', 'weight'])
# print(df)
df.s = df.s.astype(int)
df.t = df.t.astype(int)

weights_list = df.values.tolist()

# G = nx.Graph()
# G.add_edges_from()

G = nx.Graph()
G.add_weighted_edges_from(weights_list)
# print(G.edges)
# all_spp_dist = nx.all_pairs_shortest_path_length(G)

all_spp_dist = nx.all_pairs_dijkstra_path_length(G)

all_spp_dist_list = list(all_spp_dist)
print(all_spp_dist_list[0])
print(all_spp_dist_list[1])

all_pair_dist = []
edges_list = []
for s, t_dict in all_spp_dist_list:
    for t, w in t_dict.items():
        if s!=t:
            all_pair_dist.append([s,t,w])

all_pairs_df = pd.DataFrame(all_pair_dist, columns=['s', 't', 'weight'])
print('shape', all_pairs_df.shape)


all_pairs_df.to_csv('../data/{dataset_name}/whole_edge_dist.csv'.format(dataset_name=dataset_name), index=False, header=True)







