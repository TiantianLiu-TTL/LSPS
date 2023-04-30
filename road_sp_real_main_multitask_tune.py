import time
import sys
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append('/home/xiao/Documents')
sys.path.append('/home/xiaol/Documents')

# from utils import load_graph_data, construct_initial_graph
import torch
from sklearn.preprocessing import StandardScaler
from torch import embedding
from models.sp_models import EdgeConv
from models.sp_models import MLPNet
import torch.optim as optim
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import copy
import numpy as np
from datetime import datetime
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--tier", type=int, default=2)

parser.add_argument("--loss_weight", type=str, default='1')

parser.add_argument("--prefix", type=str, default='none')

parser.add_argument("--node", type=int, default=0)

parser.add_argument("--batch", type=int, default=100000)


args = parser.parse_args()

# set hyperparameters:
torch.random.manual_seed(5)
device = torch.device('cuda')
# out_channels = 16
final_out_dim = 1
lr = 0.01
weight_decay = 0.0000
pretrain_epochs = 30000
training_way = 0 # 0: all retraining, 1: incremental retraining on new nodes 2: incremental retraining on new nodes and neighbor nodes

# step 1:  load data
# pretrain_df, pos_test_df, online_train_data, min_x_y, max_x_y = load_graph_data()
sc = StandardScaler()

# def load_data():
#     zero_edge_index = pd.read_csv('./data/bio-SC-LC/edge_2hop2tier_tier_0.txt', sep='\t', names=['s', 't']).T.astype(int)
#     one_edge_index = pd.read_csv('./data/bio-SC-LC/edge_2hop2tier_tier_1.txt', sep='\t', names=['s', 't']).T.astype(int)
#     two_edge_index = pd.read_csv('./data/bio-SC-LC/edge_2hop2tier_tier_2.txt', sep='\t', names=['s', 't']).T
#     edge_weights = pd.read_csv('./data/bio-SC-LC/whole_edge_dist.csv', sep=',', header=0).astype(float)
#     hop_edge_dist = pd.read_csv('./data/bio-SC-LC/whole_edge_hop_dist.csv', sep=',', header=0).astype(float)
#     node_feats = pd.read_csv('./data/bio-SC-LC/feature_2hop2tier.txt', sep='\t',header=None, index_col=0).astype(float)
#
#     print('node feat null:', pd.isnull(node_feats).sum())
#     print('node null samples location:', np.where(node_feats.isna().values == 1))
#     null_values = node_feats.values[np.where(node_feats.isna().values == 1)]
#     print('node null samples:', null_values)
#     node_feats = sc.fit_transform(node_feats)
#     print('node feats:', node_feats.shape)
#     print(node_feats[:5])
#     return zero_edge_index, one_edge_index, two_edge_index, edge_weights, node_feats, hop_edge_dist

def load_data():
    zero_edge_index = pd.read_csv(f'./data/road-NA/node{args.node}/edge_3hop2tier_tier_0.txt', sep='\t', names=['s', 't']).T.astype(int)
    one_edge_index = pd.read_csv(f'./data/road-NA/node{args.node}/edge_3hop2tier_tier_1.txt', sep='\t', names=['s', 't']).T.astype(int)
    two_edge_index = pd.read_csv(f'./data/road-NA/node{args.node}/edge_3hop2tier_tier_2.txt', sep='\t', names=['s', 't']).T.astype(int)

    distance = pd.read_csv(f'./data/road-NA/node{args.node}/samples_weight_{args.node}.txt', sep=' ', names=['s', 't', 'weight', 'hop']).astype(float)

    distance = distance.loc[(distance['weight']>0)&(distance['hop']>0), :].reset_index(drop=True)

    edge_weights = distance.loc[:, ['s', 't', 'weight']]
    hop_edge_dist = distance.loc[:, ['s', 't', 'hop']]
    hop_edge_dist.columns = ['s', 't', 'weight']

    node_feats = pd.read_csv(f'./data/road-NA/node{args.node}/feature_3hop2tier.txt', sep='\t',header=None, index_col=0).astype(float)

    # print('node feat null:', pd.isnull(node_feats).sum())
    # print('node null samples location:', np.where(node_feats.isna().values == 1))
    null_values = node_feats.values[np.where(node_feats.isna().values == 1)]
    print('node null samples:', null_values)
    node_feats = sc.fit_transform(node_feats)
    print('node feat shpe', node_feats.shape)
    print(node_feats[:5])
    return zero_edge_index, one_edge_index, two_edge_index, edge_weights, node_feats, hop_edge_dist

zero_edge_index, one_edge_index, two_edge_index, edge_weights, node_feats, hop_edge_dist = load_data()

# step2: get features/labels for model input
zero_edge_index_arr = zero_edge_index.values.tolist()
zero_edge_index_ts = torch.LongTensor(zero_edge_index_arr).to(device)

one_edge_index_arr = one_edge_index.values.tolist()
one_edge_index_ts = torch.LongTensor(one_edge_index_arr).to(device)

two_edge_index_arr = two_edge_index.values.tolist()
two_edge_index_ts = torch.LongTensor(two_edge_index_arr).to(device)

edge_index = [zero_edge_index_ts, one_edge_index_ts, two_edge_index_ts]
edge_weights = edge_weights.sample(frac=1).reset_index(drop=True)

ori_indices = edge_weights.loc[:,['s','t']].values.tolist()
ori_indices = list(map(lambda x: tuple(x), ori_indices))

hop_edge_dist = hop_edge_dist.set_index(['s', 't'])

# only_hop_edge_dist = hop_edge_dist.loc[ori_indices, :].reset_index(drop=True).rename(columns={'weight': 'hop_dist'})

hop_edge_dist = hop_edge_dist.loc[ori_indices, :].reset_index()

print('edge_weights:', edge_weights.reset_index(drop=True).loc[:5, :])
# print('only_hop_edge_dist:', only_hop_edge_dist.loc[:5, :])

# groud_truth_edge_dist = pd.concat([edge_weights.reset_index(drop=True), only_hop_edge_dist], axis=1)
# groud_truth_edge_dist.to_csv('./data/bio-SC-LC/edge_dist_ground_truth.csv', index=False, header=True)

shp = edge_weights.shape
train_df = edge_weights.iloc[:int(shp[0] * 0.8), :]
train_weights = torch.FloatTensor(train_df[['weight']].values).to(device)

hop_shp = hop_edge_dist.shape
print('shp, hop_shp:', shp, hop_shp)

assert shp == hop_shp, 'error, unmatch in data size!'
hop_train_df = hop_edge_dist.iloc[:int(hop_shp[0] * 0.8), :]
hop_train_weights = torch.FloatTensor(hop_train_df[['weight']].values).to(device)

valid_df = edge_weights.iloc[int(shp[0] * 0.8):, :]
ori_weight_valid_weights = torch.FloatTensor(valid_df[['weight']].values).to(device)

hop_valid_df = hop_edge_dist.iloc[int(hop_shp[0] * 0.8):, :]
ori_hop_valid_weights = torch.FloatTensor(hop_valid_df[['weight']].values).to(device)

node_feats = torch.FloatTensor(node_feats).to(device)
# step 3: construct neural network model
in_channels = len(node_feats[0])

# out_channels = copy.copy(in_channels)

# out_channels_1 = 64
# out_channels_2 = 32
# out_channels_3 = 16

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

# def get_gnn():
#     models = torch.nn.ModuleList()
#     gnn1 = EdgeConv(in_channels=in_channels, out_channels=out_channels_1).to(device)
#     models.append(gnn1)
#     gnn2 = EdgeConv(in_channels=out_channels_1, out_channels=out_channels_2).to(device)
#     models.append(gnn2)
#     gnn3 = EdgeConv(in_channels=out_channels_2, out_channels=out_channels_3).to(device)
#     models.append(gnn3)
#     return models

out_channels_3 = copy.copy(in_channels)
gnn = EdgeConv(in_channels=in_channels, out_channels=out_channels_3).to(device)
# gnn = get_gnn()

regressor = MLPNet(out_channels_3 * 2, final_out_dim).to(device)

regressor_2 = MLPNet(out_channels_3 * 2, final_out_dim).to(device)
trainable_parameters = list(gnn.parameters()) \
                       + list(regressor.parameters()) \
                       + list(regressor_2.parameters())
filter_fn = list(filter(lambda p: p.requires_grad, trainable_parameters))
# filter_fn = trainable_parameters

opt = optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay)

# step 3: pretraining
# pre_X, pre_Y, pre_edge_index, pre_mask = construct_initial_graph(pretrain_df)
# pre_X = pre_X.to(device)
# pre_Y = pre_Y.to(device)
# pre_edge_index = pre_edge_index.to(device)
# pre_mask = pre_mask.to(device)

pre_train_losses = []
X = node_feats
weight_valid_rmse_error = []
weight_mape_error = []

hop_valid_rmse_error = []
hop_valid_mape_error = []

training_time_list = []
model_size_list = []

st = datetime.now()
batch_size = args.batch

if train_df.shape[0] % batch_size == 0:
    num_train_batch = int(train_df.shape[0] / batch_size)
else:
    num_train_batch = int(train_df.shape[0] / batch_size) + 1

if valid_df.shape[0] % batch_size == 0:
    num_valid_batch = int(valid_df.shape[0] / batch_size)
else:
    num_valid_batch = int(valid_df.shape[0] / batch_size) + 1

for pre_epoch in range(pretrain_epochs):
    for batch in range(num_train_batch):
        # print('epoch:{epoch}'.format(epoch=str(pre_epoch)))
        gnn.train()
        X_emb = copy.copy(X)
        for tier in range(args.tier):
            X_emb = gnn(X_emb, edge_index[tier])
        print('training epoch:{epoch}, batch:{batch}:'.format(epoch=str(pre_epoch), batch=str(batch)))
        print('X_emb shape', X_emb.shape)
        print('X_emb:', X_emb[:5])
        regressor.train()
        regressor_2.train()

        train_batch_s = train_df['s'].values[batch * batch_size:(batch + 1) * batch_size]
        train_batch_t = train_df['t'].values[batch * batch_size:(batch + 1) * batch_size]
        pred = regressor([X_emb[train_batch_s], X_emb[train_batch_t]])

        hop_train_batch_s = hop_train_df['s'].values[batch * batch_size:(batch + 1) * batch_size]
        hop_train_batch_t = hop_train_df['t'].values[batch * batch_size:(batch + 1) * batch_size]
        hop_pred = regressor_2([X_emb[hop_train_batch_s], X_emb[hop_train_batch_t]])
        print('{n} epoch {b} batch train pred shape'.format(n=pre_epoch, b=str(batch)), pred.shape)
        print('{n} epoch {b} batch train pred'.format(n=pre_epoch, b=str(batch)), pred[:10])
        print('{n} epoch {b} batch train true:'.format(n=pre_epoch, b=str(batch)), train_weights[batch * batch_size:(batch + 1) * batch_size][:10])

        weight_loss = F.mse_loss(pred, train_weights[batch * batch_size:(batch + 1) * batch_size])
        hop_loss = F.mse_loss(hop_pred, hop_train_weights[batch * batch_size:(batch + 1) * batch_size])

        print('{n} epoch {b} batch weight loss:'.format(n=pre_epoch, b=str(batch)), weight_loss.item())
        print('{n} epoch {b} batch hop loss:'.format(n=pre_epoch, b=str(batch)), hop_loss.item())
        opt.zero_grad()
        loss = weight_loss + 1 * hop_loss
        loss.backward()
        opt.step()
        pre_train_loss = loss.item()
        print('{n} epoch {b} batch train loss:'.format(n=pre_epoch, b=str(batch)), pre_train_loss)

    stall_time = datetime.now()
    current_consumed_time = (stall_time - st).total_seconds() / 60
    training_time_list.append(current_consumed_time)
    # gnn_model_size = get_model_size(gnn)
    regressor_model_size = get_model_size(regressor)
    regressor_2_model_size = get_model_size(regressor_2)
    total_model_size = regressor_model_size + regressor_2_model_size
    model_size_list.append(total_model_size)

    # pre_train_losses.append(pre_train_loss)
    gnn.eval()
    regressor.eval()
    regressor_2.eval()

    with torch.no_grad():
        print('valid epoch:{epoch}'.format(epoch=str(pre_epoch)))

        pred = regressor([X_emb[valid_df['s'].values], X_emb[valid_df['t'].values]])
        hop_pred = regressor_2([X_emb[hop_valid_df['s'].values], X_emb[hop_valid_df['t'].values]])

        print('valid pred', pred[:10])
        print('valid true:', ori_weight_valid_weights[:10])
        weight_valid_loss = F.mse_loss(pred, ori_weight_valid_weights) / pred.shape[0]
        print('valid weight loss:', weight_valid_loss.item())
        weight_valid_rmse_error.append(torch.sqrt(weight_valid_loss).item())

        print('valid hop pred', hop_pred[:10])
        print('valid hop true:', ori_hop_valid_weights[:10])
        hop_valid_loss = F.mse_loss(hop_pred, ori_hop_valid_weights) / hop_pred.shape[0]
        print('valid hop loss:', hop_valid_loss.item())
        hop_valid_rmse_error.append(torch.sqrt(hop_valid_loss).item())

        weight_valid_mape = torch.sum(torch.abs(pred - ori_weight_valid_weights))/ torch.sum(torch.abs(ori_weight_valid_weights))

        weight_mape_error.append(weight_valid_mape.item())

        hop_valid_mape = torch.sum(torch.abs(hop_pred - ori_hop_valid_weights))/torch.sum(torch.abs(ori_hop_valid_weights))
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






