import networkx as nx
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--start", type=int, default=3200)
args = parser.parse_args()
start_place = args.start

dataset_name = "road-NA"
if dataset_name == "brain":
    edge_num = "503"
elif dataset_name == "bio-SC-LC":
    edge_num = "1999"
elif dataset_name == "inf-power":
    edge_num = "4941"
elif dataset_name == "web-EPA":
    edge_num = "4253"
elif dataset_name == "road-NA":
    edge_num = "175813"
df = pd.read_csv('../data/{dataset}/Edge_{edge_num}.txt'.format(dataset=dataset_name, edge_num=edge_num), sep=' ', names=['s', 't', 'weight'])


df.s = df.s.astype(int)
df.t = df.t.astype(int)

weights_list = df.values.tolist()
# G = nx.Graph()
# G.add_edges_from()

G = nx.Graph()
G.add_weighted_edges_from(weights_list)
all_spp_dist = nx.all_pairs_dijkstra_path(G)

# print(2223432)

# all_spp_dist_list = list(all_spp_dist)
# print(all_spp_dist_list[0])
# print(all_spp_dist_list[1])

all_pair_dist = []
edges_list = []

cnt = 0
# print('total length of samples:', all_spp_dist.__len__())

item_counter = 0
for path in all_spp_dist:
    if item_counter < start_place:
        item_counter += 1
        continue
    print('path: {cnt}'.format(cnt = str(cnt)))
    s, t_dict = path
    # cnt_inter = 0
    for t, w in t_dict.items():
        # print('internal', cnt_inter)
        if s!=t:
            hop_dist = len(w)
            all_pair_dist.append([s,t,hop_dist])
        # cnt_inter += 1
    if cnt >= 100:
        print('{cnt}:go to save and release memory!'.format(cnt=str(cnt)))
        all_pairs_df = pd.DataFrame(all_pair_dist, columns=['s', 't', 'weight'])
        all_pairs_df.to_csv('../data/{dataset_name}/whole_edge_hop_dist_{cnt}.csv'.format(cnt=str(item_counter), dataset_name=dataset_name), index=False, header=True)
        all_pair_dist = []
        cnt = 0
    cnt += 1
    item_counter += 1

# for s, t_dict in all_spp_dist_list:
#     for t, w in t_dict.items():
#         if s!=t:
#             hop_dist = len(w)
#             all_pair_dist.append([s,t,hop_dist])

# all_pairs_df = pd.DataFrame(all_pair_dist, columns=['s', 't', 'weight'])
# print('shape', all_pairs_df.shape)
# all_pairs_df.to_csv('../data/{dataset_name}/whole_edge_hop_dist.csv'.format(dataset_name=dataset_name), index=False, header=True)







