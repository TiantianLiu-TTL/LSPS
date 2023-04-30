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

# all_spp_dist_list = list(all_spp_dist)
# print(all_spp_dist_list[0])
# print(all_spp_dist_list[1])

all_pair_dist = []
edges_list = []

cnt = 0
# print('total length of samples:', all_spp_dist.__len__())

item_counter = 0

for path in all_spp_dist:
    print('path: {cnt}'.format(cnt=str(cnt)))
    s, t_dict = path
    # cnt_inter = 0
    for t, w in t_dict.items():
        if s!=t:
            all_pair_dist.append([s,t,w])
    if cnt >= 100:
        print('{cnt}:go to save and release memory!'.format(cnt=str(cnt)))
        all_pairs_df = pd.DataFrame(all_pair_dist, columns=['s', 't', 'weight'])
        # all_pairs_df.to_csv('../data/{dataset_name}/whole_edge_hop_dist_{cnt}.csv'.format(cnt=str(item_counter), dataset_name=dataset_name),
        #                     index=False, header=True)
        all_pairs_df.to_csv('../data/{dataset_name}/whole_edge_dist_{cnt}.csv'.format(cnt=str(item_counter), dataset_name=dataset_name), index=False,
                            header=True)
        all_pair_dist = []
        cnt = 0

    cnt += 1
    item_counter += 1







