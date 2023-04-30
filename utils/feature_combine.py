import pandas as pd

import os, sys


path = '../data/road-NA/samples_0218'

dirs = os.listdir(path)
print(dirs)
weight_list = []
hop_list = []
for fn in dirs:
    if fn.startswith('samples'):
        file_path = os.path.join(path, fn)
        sample_df = pd.read_csv(file_path, sep=' ', names=['s', 't', 'weight', 'hop_dist'])
        print(sample_df.iloc[:10,:])
        temp_weight_df = sample_df.loc[:, ['s', 't', 'weight']]
        temp_hop_df = sample_df.loc[:, ['s', 't', 'hop_dist']]
        weight_list.append(temp_weight_df)
        hop_list.append(temp_hop_df)

weight_df = pd.concat(weight_list, axis=0).reset_index(drop=True)
hop_df = pd.concat(hop_list, axis=0).reset_index(drop=True)

weight_df.weight = weight_df.weight.round(4)
print('len of weight_df', len(weight_df))
weight_df.to_csv(os.path.join(path, 'whole_edge_dist_0218.csv'), header=True, index=False)


hop_df.columns = ['s', 't', 'weight']
print('len of hop_df', len(hop_df))

hop_df.to_csv(os.path.join(path, 'whole_edge_hop_dist_0218.csv'), header=True, index=False)

