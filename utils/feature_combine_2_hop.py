import pandas as pd

import os, sys

# sys.path.append('/home/xiaol/Documents/SPP/')

path = '/home/xiaol/Documents/SPP/data/road-NA'

df_list = []
for i in range(1,5):
    file_path = os.path.join(path, 'whole_edge_hop_dist_sa_{index}.csv'.format(index=str(i)))
    part_df = pd.read_csv(file_path, sep=',', header=0)
    df_list.append(part_df)

df = pd.concat(df_list, axis=0).reset_index(drop=True)

df.to_csv(os.path.join(path, 'whole_edge_hop_dist_sa.csv'), header=True, index=False)


