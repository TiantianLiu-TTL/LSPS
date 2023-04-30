import time
import sys
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append('/home/xiao/Documents')
# from utils import load_graph_data, construct_initial_graph
import torch
from sklearn.preprocessing import StandardScaler
from torch import embedding
from models.sp_models import EdgeConv
from models.sp_models import MLPNet
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import copy
import numpy as np
from argparse import ArgumentParser



parser = ArgumentParser()

parser.add_argument("--node", type=int, default=0)

parser.add_argument("--batch", type=int, default=10000)

args = parser.parse_args()


# set hyperparameters:
torch.random.manual_seed(5)
device = torch.device('cuda')
# out_channels = 16
final_out_dim = 1
lr = 0.01
weight_decay = 0.0001
pretrain_epochs = 30
training_way = 0 # 0: all retraining, 1: incremental retraining on new nodes 2: incremental retraining on new nodes and neighbor nodes

# step 1:  load data
# pretrain_df, pos_test_df, online_train_data, min_x_y, max_x_y = load_graph_data()
sc = StandardScaler()

def get_model_size(model):
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('Model Size: {:.3f} MB'.format(size_all_mb))
    return size_all_mb


# def load_data():
#     edge_weights = pd.read_csv('./data/bio-SC-LC/whole_edge_dist.csv', sep=',', header=0)
#     hop_edge_dist = pd.read_csv('./data/bio-SC-LC/whole_edge_hop_dist.csv', sep=',', header=0).astype(float)
#
#     edge_weights.s = edge_weights.s.astype(int)
#     edge_weights.t = edge_weights.t.astype(int)
#
#     node_emb = pd.read_csv('./data/bio-SC-LC/bio_Edge.emb', sep=' ', index_col=0, header=None)
#
#     # node_feats = pd.read_csv('./data/feature.txt', sep='\t',header=None, index_col=0)
#     # node_feats = sc.fit_transform(node_feats)
#     # print(node_feats[:5])
#     return node_emb, edge_weights, hop_edge_dist


def load_data():
    # edge_weights = pd.read_csv('./data/road-NA/whole_edge_dist.csv', sep=',', header=0)
    # hop_edge_dist = pd.read_csv('./data/road-NA/whole_edge_hop_dist.csv', sep=',', header=0).astype(float)

    distance = pd.read_csv(f'./data/road-NA/node{args.node}/samples_weight_{args.node}.txt', sep=' ',
                           names=['s', 't', 'weight', 'hop'])
    distance = distance.loc[(distance['weight'] > 0) & (distance['hop'] > 0), :].reset_index(drop=True)
    edge_weights = distance.loc[:, ['s', 't', 'weight']]
    hop_edge_dist = distance.loc[:, ['s', 't', 'hop']]
    hop_edge_dist.columns = ['s', 't', 'weight']

    edge_weights.s = edge_weights.s.astype(int)
    edge_weights.t = edge_weights.t.astype(int)
    hop_edge_dist.s = hop_edge_dist.s.astype(int)
    hop_edge_dist.t = hop_edge_dist.t.astype(int)

    node_emb = pd.read_csv(f'./data/road-NA/node_{args.node}_Edge.emb', sep=' ', index_col=0, header=None)


    return node_emb, edge_weights, hop_edge_dist

node_emb, edge_weights, hop_edge_dist = load_data()

print('node index:', max(node_emb.index), min(node_emb.index))

print('edge index:', edge_weights.s.max(), edge_weights.s.min(), edge_weights.t.max(), edge_weights.t.min())


# print('node emb:', node_emb[:5])
print('node emb:', node_emb.index)

# X_emb_s = node_emb.loc[edge_weights['s'],:]
# print('X_emb_s', X_emb_s.iloc[:5,:], X_emb_s.shape)





# X_emb = node_emb.values
# X_emb = torch.FloatTensor(X_emb).to(device)
# print('X_emb:', X_emb)
# step2: get features/labels for model input


edge_weights = edge_weights.sample(frac=1).reset_index(drop=True)


ori_indices = edge_weights.loc[:,['s','t']].values.tolist()
# print('ori_indices:', ori_indices)

ori_indices = list(map(lambda x:tuple(x), ori_indices))

hop_edge_dist = hop_edge_dist.set_index(['s','t'])

# only_hop_edge_dist = hop_edge_dist.loc[ori_indices,:].reset_index(drop=True).rename(columns={'weight':'hop_dist'})

hop_edge_dist = hop_edge_dist.loc[ori_indices,:].reset_index()

shp = edge_weights.shape
train_df = edge_weights.iloc[:int(shp[0] * 0.8), :]

train_df_s = node_emb.loc[train_df['s'],:]
train_df_t = node_emb.loc[train_df['t'],:]

train_df_s = torch.FloatTensor(train_df_s.values)
train_df_t = torch.FloatTensor(train_df_t.values)
train_weights = torch.FloatTensor(train_df[['weight']].values).to(device)

hop_shp = hop_edge_dist.shape
print('shp, hop_shp:', shp, hop_shp)

assert shp==hop_shp, 'error, unmatch in data size!'
hop_train_df = hop_edge_dist.iloc[:int(hop_shp[0] * 0.8), :]

hop_train_df_s = node_emb.loc[hop_train_df['s'],:]
hop_train_df_t = node_emb.loc[hop_train_df['t'],:]

hop_train_df_s = torch.FloatTensor(hop_train_df_s.values)
hop_train_df_t = torch.FloatTensor(hop_train_df_t.values)
hop_train_weights = torch.FloatTensor(hop_train_df[['weight']].values).to(device)


valid_df = edge_weights.iloc[int(shp[0] * 0.8):, :]

valid_df_s = node_emb.loc[valid_df['s'],:]
valid_df_t = node_emb.loc[valid_df['t'],:]

valid_df_s = torch.FloatTensor(valid_df_s.values).to(device)
valid_df_t = torch.FloatTensor(valid_df_t.values).to(device)
valid_weights = torch.FloatTensor(valid_df[['weight']].values).to(device)

hop_valid_df = hop_edge_dist.iloc[int(hop_shp[0] * 0.8):, :]

hop_valid_df_s = node_emb.loc[hop_valid_df['s'],:]
hop_valid_df_t = node_emb.loc[hop_valid_df['t'],:]

hop_valid_df_s = torch.FloatTensor(hop_valid_df_s.values).to(device)
hop_valid_df_t = torch.FloatTensor(hop_valid_df_t.values).to(device)
hop_valid_weights = torch.FloatTensor(hop_valid_df[['weight']].values).to(device)

# node_feats = torch.FloatTensor(node_feats).to(device)
# step 3: construct neural network model
# in_channels = len(node_feats[0])
# out_channels = copy.copy(in_channels)

out_channels = 128
regressor = MLPNet(out_channels*2, final_out_dim).to(device)
regressor_2 = MLPNet(out_channels*2, final_out_dim).to(device)
trainable_parameters = list(regressor.parameters()) + list(regressor_2.parameters())
filter_fn = list(filter(lambda p : p.requires_grad, trainable_parameters))
# filter_fn = trainable_parameters

ilter_fn = trainable_parameters

num_of_params = sum(p.numel() for p in filter_fn)

print('number of trainable parameters:', num_of_params)


opt = optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay)



pre_train_losses = []
weight_valid_rmse_error = []
weight_mape_error = []
hop_valid_mape_error = []
hop_valid_rmse_error = []

training_time_list = []
model_size_list = []

st = datetime.now()
batch_size = args.batch

if train_df.shape[0]%batch_size == 0:
    num_train_batch = int(train_df.shape[0]/batch_size)
else:
    num_train_batch = int(train_df.shape[0]/batch_size) + 1

if valid_df.shape[0]%batch_size == 0:
    num_valid_batch = int(valid_df.shape[0] / batch_size)
else:
    num_valid_batch = int(valid_df.shape[0] / batch_size) + 1
# print('1111', train_df['s'].values)
# print('2222', X_emb[train_df['s'].values])
for pre_epoch in range(pretrain_epochs):
    for batch in range(num_train_batch):
        # X_emb = gnn(X, edge_index_ts)
        regressor.train()
        regressor_2.train()

        # train_batch_s = train_df['s'].values[batch*batch_size:(batch+1)*batch_size]
        # train_batch_t = train_df['t'].values[batch*batch_size:(batch+1)*batch_size]

        # print('X_emb train_batch_s:', X_emb[train_batch_s])
        pred = regressor([train_df_s[batch*batch_size:(batch+1)*batch_size].to(device), train_df_t[batch*batch_size:(batch+1)*batch_size].to(device)])

        # hop_train_batch_s = hop_train_df['s'].values[batch*batch_size:(batch+1)*batch_size]
        # hop_train_batch_t = hop_train_df['t'].values[batch*batch_size:(batch+1)*batch_size]
        hop_pred = regressor_2([hop_train_df_s[batch*batch_size:(batch+1)*batch_size].to(device), hop_train_df_t[batch*batch_size:(batch+1)*batch_size].to(device)])
        print('{n} epoch {b} batch train pred shape'.format(n=pre_epoch, b=str(batch)), pred.shape)
        print('{n} epoch {b} batch train pred'.format(n=pre_epoch, b=str(batch)), pred[:10])
        print('{n} epoch {b} batch train true:'.format(n=pre_epoch, b=str(batch)), train_weights[batch*batch_size:(batch+1)*batch_size][:10])

        weight_loss = F.mse_loss(pred, train_weights[batch*batch_size:(batch+1)*batch_size])
        hop_loss = F.mse_loss(hop_pred, hop_train_weights[batch*batch_size:(batch+1)*batch_size])

        print('{n} epoch {b} batch weight loss:'.format(n=pre_epoch, b=str(batch)), weight_loss.item())
        print('{n} epoch {b} batch hop loss:'.format(n=pre_epoch, b=str(batch)), hop_loss.item())
        opt.zero_grad()
        loss = weight_loss + 1*hop_loss
        loss.backward()
        opt.step()
        pre_train_loss = loss.item()
        print('{n} epoch {b} batch train loss:'.format(n=pre_epoch, b=str(batch)), pre_train_loss)

    regressor.eval()
    regressor_2.eval()

    stall_time = datetime.now()
    current_consumed_time = (stall_time - st).total_seconds() / 60
    training_time_list.append(current_consumed_time)
    # gnn_model_size = get_model_size(gnn)
    regressor_model_size = get_model_size(regressor)
    regressor_2_model_size = get_model_size(regressor_2)
    total_model_size = regressor_model_size + regressor_2_model_size
    model_size_list.append(total_model_size)

    with torch.no_grad():
        valid_st_time = datetime.now()

        pred = regressor([valid_df_s, valid_df_t])
        hop_pred = regressor_2([hop_valid_df_s, valid_df_t])

        valid_et_time = datetime.now()
        valid_elapsed_time = (valid_et_time - valid_st_time).total_seconds()
        num_of_valid_samples = pred.shape[0]
        spent_time_per_sample = round((valid_elapsed_time / num_of_valid_samples) * 1e3, 6)

        print('valid_elapsed_time:', valid_elapsed_time)
        print('num_of_valid_samples', num_of_valid_samples)
        print('spent_time_per_sample', spent_time_per_sample)

        print('valid pred', pred[:10])
        print('valid true:', valid_weights[:10])
        weight_valid_loss = F.mse_loss(pred, valid_weights)
        print('valid weight loss:', weight_valid_loss.item())
        weight_valid_rmse_error.append(torch.sqrt(weight_valid_loss).item())

        print('valid hop pred', hop_pred[:10])
        print('valid hop true:', hop_valid_weights[:10])
        hop_valid_loss = F.mse_loss(hop_pred, hop_valid_weights)
        print('valid hop loss:', hop_valid_loss.item())
        hop_valid_rmse_error.append(torch.sqrt(hop_valid_loss).item())

        # weight_valid_mape = torch.sum(torch.abs(pred - valid_weights) / valid_weights) / pred.shape[0]
        weight_valid_mape = torch.sum(torch.abs(pred - valid_weights))/ torch.sum(torch.abs(valid_weights))

        weight_mape_error.append(weight_valid_mape.item())

        # hop_valid_mape = torch.sum(torch.abs(hop_pred - hop_valid_weights) / hop_valid_weights) / hop_pred.shape[0]
        hop_valid_mape = torch.sum(torch.abs(hop_pred - hop_valid_weights))/torch.sum(torch.abs(hop_valid_weights))

        hop_valid_mape_error.append(hop_valid_mape.item())

        # min weight mape
        arg_min_mape, min_mape = min(list(enumerate(weight_mape_error)), key=lambda x: x[1])
        print('min weight mape error: epoch, w_mape, time, size, h_mape, w_rmse, h_rmse', arg_min_mape, min_mape,
              training_time_list[arg_min_mape], model_size_list[arg_min_mape], hop_valid_mape_error[arg_min_mape],
              weight_valid_rmse_error[arg_min_mape], hop_valid_rmse_error[arg_min_mape])

        # min hop mape
        arg_min_hop_mape, min_hop_mape = min(list(enumerate(hop_valid_mape_error)), key=lambda x: x[1])
        print('min hop mape error: epoch, mape, time, size, w_mape, w_rmse, h_rmse', arg_min_hop_mape, min_hop_mape,
              training_time_list[arg_min_hop_mape], model_size_list[arg_min_hop_mape],
              weight_valid_rmse_error[arg_min_hop_mape], hop_valid_rmse_error[arg_min_hop_mape])

        # min weight rmse
        arg_min_rmse, min_rmse = min(list(enumerate(weight_valid_rmse_error)), key=lambda x: x[1])
        print('min weight rmse error: epoch, rmse, time, size, h_rmse, w_mape, h_mape', arg_min_rmse, min_rmse,
              training_time_list[arg_min_rmse],
              model_size_list[arg_min_rmse], hop_valid_rmse_error[arg_min_rmse], weight_mape_error[arg_min_rmse],
              hop_valid_mape_error[arg_min_rmse])

        # min hop rmse
        arg_min_hop_rmse, min_hop_rmse = min(list(enumerate(hop_valid_rmse_error)), key=lambda x: x[1])
        print('min hop rmse error: epoch, rmse, time, size, w_rmse, w_mape, h_mape', arg_min_hop_rmse, min_hop_rmse,
              training_time_list[arg_min_hop_rmse], model_size_list[arg_min_hop_rmse],
              weight_valid_rmse_error[arg_min_hop_rmse], weight_mape_error[arg_min_hop_rmse],
              hop_valid_mape_error[arg_min_hop_rmse])

