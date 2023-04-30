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
from sklearn.neighbors import NearestNeighbors
import copy
import numpy as np
# set hyperparameters:
torch.random.manual_seed(5)
device = torch.device('cuda')
# out_channels = 16
final_out_dim = 1
lr = 0.01
weight_decay = 0.0001
pretrain_epochs = 10000
ontrain_epochs = 10000
training_way = 0 # 0: all retraining, 1: incremental retraining on new nodes 2: incremental retraining on new nodes and neighbor nodes

# step 1:  load data
# pretrain_df, pos_test_df, online_train_data, min_x_y, max_x_y = load_graph_data()
sc = StandardScaler()

def load_data():
    edge_weights = pd.read_csv('./data/bio-SC-LC/whole_edge_hop_dist.csv', sep=',', header=0)
    edge_weights.s = edge_weights.s.astype(int)
    edge_weights.t = edge_weights.t.astype(int)

    node_emb = pd.read_csv('./data/bio-SC-LC/brain_Edge.emb', sep=' ', index_col=0, header=None)

    # node_feats = pd.read_csv('./data/feature.txt', sep='\t',header=None, index_col=0)
    # node_feats = sc.fit_transform(node_feats)
    # print(node_feats[:5])
    return node_emb, edge_weights

node_emb, edge_weights = load_data()

# print('node emb:', node_emb[:5])
print('node emb:', node_emb.index)

X_emb = node_emb.loc[range(node_emb.shape[0]), :].values
X_emb = torch.FloatTensor(X_emb).to(device)
print('X_emb:', X_emb)
# step2: get features/labels for model input

edge_weights = edge_weights.sample(frac=1)
shp = edge_weights.shape
train_df = edge_weights.iloc[:int(shp[0] * 0.8), :]

train_weights = torch.FloatTensor(train_df[['weight']].values).to(device)
valid_df = edge_weights.iloc[int(shp[0] * 0.8):, :]
valid_weights = torch.FloatTensor(valid_df[['weight']].values).to(device)

# node_feats = torch.FloatTensor(node_feats).to(device)

# step 3: construct neural network model
# in_channels = len(node_feats[0])
# out_channels = copy.copy(in_channels)

out_channels = 128


regressor = MLPNet(out_channels*2, final_out_dim).to(device)
trainable_parameters = list(regressor.parameters())
filter_fn = list(filter(lambda p : p.requires_grad, trainable_parameters))
# filter_fn = trainable_parameters

opt = optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay)

#step 3: pretraining
# pre_X, pre_Y, pre_edge_index, pre_mask = construct_initial_graph(pretrain_df)
# pre_X = pre_X.to(device)
# pre_Y = pre_Y.to(device)
# pre_edge_index = pre_edge_index.to(device)
# pre_mask = pre_mask.to(device)

pre_train_losses = []
valid_mse_error = []
valid_rmse_error = []
weight_mape_error = []

print('1111', train_df['s'].values)
print('2222', X_emb[train_df['s'].values])
for pre_epoch in range(pretrain_epochs):
    regressor.train()
    opt.zero_grad()
    pred = regressor([X_emb[train_df['s'].values], X_emb[train_df['t'].values]])
    print('train pred shape', pred.shape)
    print('train pred', pred[:10])
    print('train true:', train_weights[:10])
    loss = F.mse_loss(pred, train_weights)
    loss.backward()
    opt.step()
    pre_train_loss = loss.item()
    print('{n} epoch train loss:'.format(n=pre_epoch), pre_train_loss)
    pre_train_losses.append(pre_train_loss)
    regressor.eval()
    with torch.no_grad():
        pred = regressor([X_emb[valid_df['s'].values], X_emb[valid_df['t'].values]])
        print('valid pred', pred[:10])
        print('valid true:', valid_weights[:10])
        valid_loss = F.mse_loss(pred, valid_weights)
        print('valid loss:', valid_loss.item())
        # valid_mse_error.append(valid_loss.item())
        # valid_rmse_error.append(torch.sqrt(valid_loss))
        # print('min mse error:', min(valid_mse_error))
        # print('min rmse error:', min(valid_rmse_error))

        weight_valid_mape = torch.sum(torch.abs(pred - valid_weights) / valid_weights) / pred.shape[0]
        weight_mape_error.append(weight_valid_mape.item())
        arg_min_mape, min_mape = min(list(enumerate(weight_mape_error)), key=lambda x: x[1])
        print('min mape error (hop):', arg_min_mape, min_mape)




