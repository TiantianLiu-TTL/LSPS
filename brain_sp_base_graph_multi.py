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
# set hyperparameters:
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--batch", type=int, default=10000)
parser.add_argument("--epoch", type=int, default=30)


args = parser.parse_args()

torch.random.manual_seed(5)
device = torch.device('cuda')
# out_channels = 16
final_out_dim = 1
lr = 0.01
weight_decay = 0.0001
pretrain_epochs = args.epoch

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

def load_data():
    edge_index = pd.read_csv('./data/brain/Edge_503.txt', sep=' ').iloc[:,:2].T
    print('edge_index samples:', edge_index[:5])
    edge_weights = pd.read_csv('./data/brain/whole_edge_dist.csv', sep=',', header=0)
    hop_edge_dist = pd.read_csv('./data/brain/whole_edge_hop_dist.csv', sep=',', header=0).astype(float)

    node_feats = pd.read_csv('./data/brain/feature_2hop1tier.txt', sep='\t',header=None, index_col=0)
    node_feats = sc.fit_transform(node_feats)
    # print(node_feats[:5])
    return edge_index, edge_weights, node_feats, hop_edge_dist

edge_index, edge_weights, node_feats, hop_edge_dist = load_data()

# step2: get features/labels for model input
edge_index_arr = edge_index.values.tolist()
edge_index_ts = torch.LongTensor(edge_index_arr).to(device)

edge_weights = edge_weights.sample(frac=1).reset_index(drop=True)

ori_indices = edge_weights.loc[:,['s','t']].values.tolist()
# print('ori_indices:', ori_indices)
ori_indices = list(map(lambda x:tuple(x), ori_indices))
hop_edge_dist = hop_edge_dist.set_index(['s','t'])

only_hop_edge_dist = hop_edge_dist.loc[ori_indices,:].reset_index(drop=True).rename(columns={'weight':'hop_dist'})

hop_edge_dist = hop_edge_dist.loc[ori_indices,:].reset_index()

shp = edge_weights.shape
hop_shp = hop_edge_dist.shape
print('shp, hop_shp:', shp, hop_shp)
assert shp==hop_shp, 'error, unmatch in data size!'

train_df = edge_weights.iloc[:int(shp[0] * 0.8), :]

train_weights = torch.FloatTensor(train_df[['weight']].values).to(device)
hop_train_df = hop_edge_dist.iloc[:int(hop_shp[0] * 0.8), :]
hop_train_weights = torch.FloatTensor(hop_train_df[['weight']].values).to(device)


valid_df = edge_weights.iloc[int(shp[0] * 0.8):, :]
valid_weights = torch.FloatTensor(valid_df[['weight']].values).to(device)

hop_valid_df = hop_edge_dist.iloc[int(hop_shp[0] * 0.8):, :]
hop_valid_weights = torch.FloatTensor(hop_valid_df[['weight']].values).to(device)


feat_shp = node_feats.shape
node_feats = torch.rand(feat_shp[0], feat_shp[1]) ###random generate the input feature for GNN
node_feats = torch.FloatTensor(node_feats).to(device)

# step 3: construct neural network model
in_channels = len(node_feats[0])
# out_channels = copy.copy(in_channels)
out_channels = 16

def get_gnn():
    gnn = EdgeConv(in_channels=in_channels, out_channels=out_channels).to(device)
    return gnn

# gnn = EdgeConv(in_channels=in_channels, out_channels=out_channels).to(device)
gnn = get_gnn()

regressor = MLPNet(out_channels*2, final_out_dim).to(device)
regressor_2 = MLPNet(out_channels*2, final_out_dim).to(device)
trainable_parameters = list(gnn.parameters()) \
                       + list(regressor.parameters()) \
                              + list(regressor_2.parameters())
# filter_fn = list(filter(lambda p : p.requires_grad, trainable_parameters))
filter_fn = trainable_parameters

opt = optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay)


pre_train_losses = []
X = node_feats
weight_valid_rmse_error = []
hop_valid_rmse_error = []

weight_mape_error = []
hop_valid_mape_error = []
training_time_list = []
model_size_list = []

st = datetime.now()

for pre_epoch in range(pretrain_epochs):
    gnn.train()
    regressor.train()
    regressor_2.train()

    opt.zero_grad()
    X_emb = copy.copy(X)
    for layer in range(2):
        X_emb = gnn(X_emb, edge_index_ts)
    print('X_emb shape', X_emb.shape)
    print('X_emb:', X_emb[:5])
    pred = regressor([X_emb[train_df['s'].values], X_emb[train_df['t'].values]])
    hop_pred = regressor_2([X_emb[hop_train_df['s'].values], X_emb[hop_train_df['t'].values]])

    print('train pred shape', pred.shape)
    print('train pred', pred[:10])
    print('train true:', train_weights[:10])
    loss = F.mse_loss(pred, train_weights)
    hop_loss = F.mse_loss(hop_pred, hop_train_weights)
    loss = loss + hop_loss
    loss.backward()
    opt.step()
    pre_train_loss = loss.item()
    print('{n} epoch train loss:'.format(n=pre_epoch), pre_train_loss)
    pre_train_losses.append(pre_train_loss)
    gnn.eval()
    regressor.eval()
    regressor_2.eval()

    stall_time = datetime.now()
    current_consumed_time = (stall_time - st).total_seconds() / 60
    training_time_list.append(current_consumed_time)
    gnn_model_size = get_model_size(gnn)
    regressor_model_size = get_model_size(regressor)
    regressor_2_model_size = get_model_size(regressor_2)
    total_model_size = gnn_model_size + regressor_model_size + regressor_2_model_size
    model_size_list.append(total_model_size)

    with torch.no_grad():
        pred = regressor([X_emb[valid_df['s'].values], X_emb[valid_df['t'].values]])
        hop_pred = regressor_2([X_emb[hop_valid_df['s'].values], X_emb[hop_valid_df['t'].values]])

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

        #min weight mape
        arg_min_mape, min_mape = min(list(enumerate(weight_mape_error)), key=lambda x: x[1])
        print('min weight mape error: epoch, w_mape, time, size, h_mape, w_rmse, h_rmse', arg_min_mape, min_mape, training_time_list[arg_min_mape],
              model_size_list[arg_min_mape], hop_valid_mape_error[arg_min_mape], weight_valid_rmse_error[arg_min_mape], hop_valid_rmse_error[arg_min_mape])
        #min hop mape
        arg_min_hop_mape, min_hop_mape = min(list(enumerate(hop_valid_mape_error)), key=lambda x: x[1])
        print('min hop mape error: epoch, mape, time, size, w_mape, w_rmse, h_rmse', arg_min_hop_mape, min_hop_mape,
              training_time_list[arg_min_hop_mape], model_size_list[arg_min_hop_mape], weight_valid_rmse_error[arg_min_hop_mape], hop_valid_rmse_error[arg_min_hop_mape])

        #min weight rmse
        arg_min_rmse, min_rmse = min(list(enumerate(weight_valid_rmse_error)), key=lambda x: x[1])
        print('min weight rmse error: epoch, rmse, time, size, h_rmse, w_mape, h_mape', arg_min_rmse, min_rmse, training_time_list[arg_min_rmse],
              model_size_list[arg_min_rmse], hop_valid_rmse_error[arg_min_rmse], weight_mape_error[arg_min_rmse], hop_valid_mape_error[arg_min_rmse])

        #min hop rmse
        arg_min_hop_rmse, min_hop_rmse = min(list(enumerate(hop_valid_rmse_error)), key=lambda x: x[1])
        print('min hop rmse error: epoch, rmse, time, size, w_rmse, w_mape, h_mape', arg_min_hop_rmse, min_hop_rmse,
              training_time_list[arg_min_hop_rmse], model_size_list[arg_min_hop_rmse], weight_valid_rmse_error[arg_min_hop_rmse], weight_mape_error[arg_min_hop_rmse], hop_valid_mape_error[arg_min_hop_rmse])






