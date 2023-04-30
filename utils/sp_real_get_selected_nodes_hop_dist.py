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
elif dataset_name == "road-NA":
    edge_num = "175813"
elif dataset_name == "web-EPA":
    edge_num = "4253"

df = pd.read_csv('../data/{dataset}/Edge_{edge_num}.txt'.format(dataset=dataset_name, edge_num=edge_num), sep=' ', names=['s', 't', 'weight'])
df.s = df.s.astype(int)
df.t = df.t.astype(int)

weights_list = df.values.tolist()

G = nx.Graph()
G.add_weighted_edges_from(weights_list)

for sample_index in range(1,11):
    print('sample_index:', sample_index)
    selected_edges = pd.read_csv('../data/{dataset}/samples_{sample_index}'.format(dataset=dataset_name, sample_index=sample_index), names=['s', 't'], sep=' ')
    all_pair_dist = []
    print('selected edge samples:', len(selected_edges))

    # print('selected edges:', selected_edges)
    cnt = 0
    item_counter = 0
    threshold = 10000

    for index, row in selected_edges.iterrows():
        s, t = row['s'], row['t']
        if s != t:
            path = nx.dijkstra_path(G, s, t)
            hop_dist = len(path)
            all_pair_dist.append([s, t, hop_dist])
            cnt += 1
            # if cnt >= threshold:
            #     print('{cnt}:go to save and release memory!'.format(cnt=str(cnt)))
            #     all_pairs_df = pd.DataFrame(all_pair_dist, columns=['s', 't', 'weight'])
            #     all_pairs_df.to_csv('../data/{dataset_name}/whole_edge_hop_dist_{cnt}.csv'.format(cnt=str(item_counter),
            #                                                                                       dataset_name=dataset_name), index=False, header=True)
            #     all_pair_dist = []
            #     cnt = 0
            item_counter += 1

    all_pairs_df = pd.DataFrame(all_pair_dist, columns=['s', 't', 'weight'])
    all_pairs_df.to_csv('../data/{dataset_name}/whole_edge_hop_dist_sa_{sample_index}.csv'.format(dataset_name=dataset_name, sample_index=sample_index), index=False, header=True)











