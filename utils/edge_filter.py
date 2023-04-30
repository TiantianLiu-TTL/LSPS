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
df.s = df.s.astype(int)
df.t = df.t.astype(int)

total_edges = df[['s', 't']]

print('len of df before', len(df))

selected_edges = pd.read_csv('../data/{dataset}/samples.txt'.format(dataset=dataset_name), names=['s', 't'], sep=' ')

selected_list_of_tuples = list(selected_edges.itertuples(index=False, name=None))

print('select samples:', selected_list_of_tuples[:3])

list_of_tuples = list(total_edges.itertuples(index=False, name=None))

print('list of samples:', list_of_tuples[:3])

print('len of selected before', len(selected_list_of_tuples))
common_list_of_tuples = list(filter(lambda x: x in list_of_tuples, selected_list_of_tuples))
print('len of selected after', len(common_list_of_tuples))

common_edge_df = pd.DataFrame(common_list_of_tuples)

common_edges = common_edge_df.values.tolist()

df = df.set_index(['s', 't'])
df = df.loc[common_edges,:]
print('len of df after', len(df))
df.to_csv('../data/{dataset}/common_edges.csv'.format(dataset=dataset_name))