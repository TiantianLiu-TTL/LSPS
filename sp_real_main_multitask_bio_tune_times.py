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
from sp_util import load_data
# set hyperparameters:
torch.random.manual_seed(5)
device = torch.device('cuda')
# out_channels = 16
final_out_dim = 1
lr = 0.01
weight_decay = 0.0001
pretrain_epochs = 100000
ontrain_epochs = 100000
training_way = 0 # 0: all retraining, 1: incremental retraining on new nodes 2: incremental retraining on new nodes and neighbor nodes

# step 1:  load data
# pretrain_df, pos_test_df, online_train_data, min_x_y, max_x_y = load_graph_data()
sc = StandardScaler()

def load_data():
    zero_edge_index = pd.read_csv('./data/bio-SC-LC/edge_2hop2tier_tier_0.txt', sep='\t', names=['s', 't']).T.astype(int)
    one_edge_index = pd.read_csv('./data/bio-SC-LC/edge_2hop2tier_tier_1.txt', sep='\t', names=['s', 't']).T.astype(int)
    two_edge_index = pd.read_csv('./data/bio-SC-LC/edge_2hop2tier_tier_2.txt', sep='\t', names=['s', 't']).T
    edge_weights = pd.read_csv('./data/bio-SC-LC/whole_edge_dist.csv', sep=',', header=0).astype(float)
    hop_edge_dist = pd.read_csv('./data/bio-SC-LC/whole_edge_hop_dist.csv', sep=',', header=0).astype(float)
    node_feats = pd.read_csv('./data/bio-SC-LC/feature_2hop2tier.txt', sep='\t',header=None, index_col=0).astype(float)

    print('node feat null:', pd.isnull(node_feats).sum())
    print('node null samples location:', np.where(node_feats.isna().values == 1))
    null_values = node_feats.values[np.where(node_feats.isna().values == 1)]
    print('node null samples:', null_values)
    node_feats = sc.fit_transform(node_feats)
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
edge_weights = edge_weights.sample(frac=1)

ori_indices = edge_weights.values.tolist()
ori_indices = list(map(lambda x:tuple(x), ori_indices))

hop_edge_dist = hop_edge_dist.set_index(['s','t'])

only_hop_edge_dist = hop_edge_dist.loc[ori_indices,:].reset_index(drop=True).rename(columns={'weight':'hop_dist'})

hop_edge_dist = hop_edge_dist.loc[ori_indices,:].reset_index()

print('edge_weights:', edge_weights.reset_index(drop=True).loc[:5,:])
print('only_hop_edge_dist:', only_hop_edge_dist.loc[:5,:])

groud_truth_edge_dist = pd.concat([edge_weights.reset_index(drop=True), only_hop_edge_dist], axis=1)
groud_truth_edge_dist.to_csv('./data/bio-SC-LC/edge_dist_ground_truth.csv', index=False, header=True)

shp = edge_weights.shape
train_df = edge_weights.iloc[:int(shp[0] * 0.8), :]
train_weights = torch.FloatTensor(train_df[['weight']].values).to(device)

hop_shp = hop_edge_dist.shape
print('shp, hop_shp:', shp, hop_shp)

assert shp==hop_shp, 'error, unmatch in data size!'
hop_train_df = hop_edge_dist.iloc[:int(hop_shp[0] * 0.8), :]
hop_train_weights = torch.FloatTensor(hop_train_df[['weight']].values).to(device)

valid_df = edge_weights.iloc[int(shp[0] * 0.8):, :]
valid_weights = torch.FloatTensor(valid_df[['weight']].values).to(device)

hop_valid_df = hop_edge_dist.iloc[int(hop_shp[0] * 0.8):, :]
hop_valid_weights = torch.FloatTensor(hop_valid_df[['weight']].values).to(device)

node_feats = torch.FloatTensor(node_feats).to(device)
# step 3: construct neural network model
in_channels = len(node_feats[0])
# out_channels = copy.copy(in_channels)

out_channels_1 = 64
out_channels_2 = 32
out_channels_3 = 16

def get_gnn():
    models = torch.nn.ModuleList()
    gnn1 = EdgeConv(in_channels=in_channels, out_channels=out_channels_1).to(device)
    models.append(gnn1)
    gnn2 = EdgeConv(in_channels=out_channels_1, out_channels=out_channels_2).to(device)
    models.append(gnn2)
    gnn3 = EdgeConv(in_channels=out_channels_2, out_channels=out_channels_3).to(device)
    models.append(gnn3)
    return models

# gnn = EdgeConv(in_channels=in_channels, out_channels=out_channels).to(device)
gnn = get_gnn()

regressor = MLPNet(out_channels_3*2, final_out_dim).to(device)

regressor_2 = MLPNet(out_channels_3*2, final_out_dim).to(device)
trainable_parameters = list(gnn.parameters()) \
                       + list(regressor.parameters()) \
                              + list(regressor_2.parameters())
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
X = node_feats
# valid_mse_error = []
# valid_rmse_error = []
weight_mape_error = []

# hop_valid_mse_error = []
# hop_valid_rmse_error = []
hop_valid_mape_error = []

for pre_epoch in range(pretrain_epochs):
    print('epoch:{epoch}'.format(epoch=str(pre_epoch)))
    gnn.train()
    regressor.train()
    opt.zero_grad()
    X_emb = copy.copy(X)
    for tier in range(3):
        X_emb = gnn[tier](X_emb, edge_index[tier])
    print('X_emb shape', X_emb.shape)
    print('X_emb:', X_emb[:5])
    pred = regressor([X_emb[train_df['s'].values], X_emb[train_df['t'].values]])
    hop_pred = regressor_2([X_emb[hop_train_df['s'].values], X_emb[hop_train_df['t'].values]])

    print('train pred shape', pred.shape)
    print('train pred', pred[:10])
    print('train true:', train_weights[:10])
    loss = F.mse_loss(pred, train_weights)/10000000
    hop_loss = F.mse_loss(hop_pred, hop_train_weights)/10000000
    print('{n} epoch weight loss:'.format(n=pre_epoch), loss.item())
    print('{n} epoch hop loss:'.format(n=pre_epoch), hop_loss.item())

    loss = loss + 1*hop_loss
    loss.backward()
    opt.step()
    pre_train_loss = loss.item()
    print('{n} epoch train loss:'.format(n=pre_epoch), pre_train_loss)
    pre_train_losses.append(pre_train_loss)
    gnn.eval()
    regressor.eval()
    with torch.no_grad():
        pred = regressor([X_emb[valid_df['s'].values], X_emb[valid_df['t'].values]])
        hop_pred = regressor_2([X_emb[hop_valid_df['s'].values], X_emb[hop_valid_df['t'].values]])

        print('valid pred', pred[:10])
        print('valid true:', valid_weights[:10])
        valid_loss = F.mse_loss(pred, valid_weights)/pred.shape[0]

        print('valid hop pred', hop_pred[:10])
        print('valid hop true:', hop_valid_weights[:10])

        weight_valid_mape = torch.sum(torch.abs(pred - valid_weights)/valid_weights)/pred.shape[0]
        weight_mape_error.append(weight_valid_mape.item())
        arg_min_mape, min_mape = min(list(enumerate(weight_mape_error)), key=lambda x:x[1])
        print('min mape error:', arg_min_mape, min_mape)

        hop_valid_loss = F.mse_loss(hop_pred, hop_valid_weights)/pred.shape[0]
        # valid_loss = valid_loss + hop_valid_loss
        # print('valid loss:', valid_loss.item())
        # valid_mse_error.append(valid_loss.item())
        # valid_rmse_error.append(torch.sqrt(valid_loss))

        hop_valid_mape = torch.sum(torch.abs(hop_pred - hop_valid_weights)/hop_valid_weights)/hop_pred.shape[0]
        hop_valid_mape_error.append(hop_valid_mape.item())
        arg_min_hop_mape, min_hop_mape = min(list(enumerate(hop_valid_mape_error)), key=lambda x:x[1])
        print('min hop mape error:', arg_min_hop_mape, min_hop_mape)

        # arg_min_mse, min_mse = min(list(enumerate(valid_mse_error)), key=lambda x:x[1])
        # arg_min_rmse, min_rmse = min(list(enumerate(valid_rmse_error)), key=lambda x:x[1])

        # print('min mse error:', arg_min_mse, min_mse)
        # print('min rmse error:', arg_min_rmse, min_rmse)

        # print('hop valid loss:', hop_valid_loss.item())
        # hop_valid_mse_error.append(hop_valid_loss.item())
        # hop_valid_rmse_error.append(torch.sqrt(hop_valid_loss))

        # arg_hop_min_mse, min_hop_mse = min(list(enumerate(hop_valid_mse_error)), key=lambda x: x[1])
        # arg_hop_min_rmse, min_hop_rmse = min(list(enumerate(hop_valid_rmse_error)), key=lambda x: x[1])

        # print('min hop mse error:',  arg_hop_min_mse, min_hop_mse)
        # print('min hop rmse error:', arg_hop_min_rmse, min_hop_rmse)

        # pred = regressor(x_embd)
        # valid_indices = torch.where(new_eval_mask == 1)[0]
        # valid_loss = F.mse_loss(pred[valid_indices], new_Y[valid_indices]) / (len(valid_indices) + 1e-5)
        # valid_loss = valid_loss.item()
        # print('{n} epoch valid loss:'.format(n=on_epoch), valid_loss)
        # valid_loss_list.append(valid_loss)
        # y_pred = (new_eval_mask * pred).data.cpu().numpy()
        # y_true = (torch.nan_to_num(new_eval_mask * new_Y, nan=0)).data.cpu().numpy()
        # y_pred = y_pred * (max_x_y - min_x_y) + min_x_y
        # y_true = y_true * (max_x_y - min_x_y) + min_x_y
        #
        # # y_pred = y_pred * std_x_y + mean_x_y
        # # y_true = y_true * std_x_y + mean_x_y
        # print('pred', y_pred[valid_indices.data.cpu().numpy()])
        # print('true', y_true[valid_indices.data.cpu().numpy()])
        # p_sqrt_dist = np.sqrt(np.sum(np.square(y_pred - y_true), axis=1))
        # # print('p_sqrt_dist', p_sqrt_dist)
        # point_error_list.append(round(np.sum(p_sqrt_dist) / (np.sum(new_eval_mask.data.cpu().numpy()) + 1e-5), 6))
        # print('predicted point error:',
        #       round(np.sum(p_sqrt_dist) / (np.sum(new_eval_mask.data.cpu().numpy()) + 1e-5), 6))


'''
step 4: online incremental retraining
current_node_index = pre_X.shape[0]
current_X = copy.copy(pre_X)
current_edge_index = copy.copy(pre_edge_index)
current_mask = copy.copy(pre_mask)
current_Y = copy.copy(pre_Y)
Feature_len = pre_X.shape[1]

for batch_i, batch_df in enumerate(online_train_data):
    print('online batch_df shp:', batch_df.shape)
    # batch_df = batch_df.sample(frac=1)
    neigh = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    neigh.fit(current_X.data.cpu().numpy())
    temp_X = batch_df.iloc[:, :Feature_len].values
    temp_Y = batch_df.loc[:, ['x_', 'y_']].values
    temp_dist_matrix, temp_neigh_matrix = neigh.kneighbors(temp_X, return_distance=True)
    print('current X temp X shp', current_X.shape, temp_X.shape)
    print('neighbor matrix:', temp_neigh_matrix)
    temp_edge_index = []
    temp_edge_attri = []
    print('temp X shape', temp_X.shape)
    print('current edge index shp', current_edge_index.shape)
    for i in range(temp_X.shape[0]):
        new_node_edge_index = [[current_node_index+i, j] for j in temp_neigh_matrix[i, :]] + [[j, current_node_index+i] for j in temp_neigh_matrix[i,:]]
        new_node_edge_attri = list(temp_dist_matrix[i,:]) + list(temp_dist_matrix[i,:])
        temp_edge_index += new_node_edge_index
        # temp_edge_attri += new_node_edge_attri
    if training_way == 0: #0: all retraining, 1: incremental retraining on new nodes
        temp_edge_index_ts = torch.LongTensor(temp_edge_index).transpose(0,1).to(device)
        print('temp_edge_index_ts shp:', temp_edge_index_ts.shape)
        new_edge_index = torch.cat([current_edge_index, temp_edge_index_ts], dim=1)
        # ra_indices = torch.randperm(new_edge_index.shape[1])
        # print('ra indices:', ra_indices)
        # new_edge_index = new_edge_index[:, ra_indices]
    elif training_way == 1:
        temp_edge_index_ts = torch.LongTensor(temp_edge_index).transpose(0, 1).to(device)
        new_edge_index = copy.copy(temp_edge_index_ts)
    print('{n}:new_edge_index:'.format(n=training_way), new_edge_index.shape)
    node_index_list = list(set(torch.ravel(new_edge_index).tolist()))
    print('node index list:', len(node_index_list))
    temp_X_ts = torch.FloatTensor(temp_X).to(device)
    print('current X shp:', current_X.shape)

    new_X = torch.cat([current_X, temp_X_ts], dim=0)
    temp_mask = torch.zeros((temp_X.shape[0], 1), dtype=float).to(device)
    temp_eval_mask = torch.FloatTensor((batch_df['x_'].notnull()).astype(float).values.reshape((-1,1))).to(device)
    new_mask = torch.cat([current_mask, temp_mask], dim=0)
    new_eval_mask = torch.cat([torch.zeros(current_mask.shape[0], 1).to(device), temp_eval_mask], dim=0)
    temp_Y_ts = torch.FloatTensor(temp_Y).to(device)
    new_Y = torch.cat([current_Y, temp_Y_ts], dim=0)

    # new_X = new_X.to(device)
    # new_Y = new_Y.to(device)
    # new_edge_index = new_edge_index.to(device)
    new_mask = new_mask.to(device)
    # new_eval_mask = new_eval_mask.to(device)

    # current retraining course
    online_train_losses = []
    valid_loss_list = []
    point_error_list = []
    online_training_st = time.time()
    pred_Y_epoch_list = []
    cnt = 0
    for on_epoch in range(ontrain_epochs):
        gnn.train()
        regressor.train()
        opt.zero_grad()
        # print('new_X', new_X[:5])
        # print('new_edge_index:', new_edge_index[:5])
        x_embd = gnn(new_X, new_edge_index)
        # print('x embed:', x_embd)
        # print('x emb just new:', x_embd[current_node_index-5:(current_node_index+5)])
        pred = regressor(x_embd)
        # print('new pred:', pred)
        # print('pred samples:', pred[:5])
        train_indices = torch.where(new_mask==1)[0]
        loss = F.mse_loss(pred[train_indices], new_Y[train_indices])/len(train_indices)

        # print('new mask shp', new_mask.shape)
        # print('new mask:', new_mask[:5])
        # print('new Y mask:', (new_mask*new_Y))
        #
        # print('new pred mask:', (new_mask*pred))
        # part_loss = F.mse_loss((new_mask*pred)[:100], (new_mask*new_Y)[:100])
        # print('part loss:', part_loss.item())

        online_train_loss = loss.item()
        print('{n} epoch train loss:'.format(n=on_epoch), online_train_loss)
        loss.backward()
        opt.step()
        online_train_losses.append(online_train_loss)
        new_constructed_Y = torch.nan_to_num(new_Y*new_mask, nan=0) + pred*(1 - new_mask)
        print('new constructed Y:', new_constructed_Y)
        new_constructed_Y = new_constructed_Y.data.cpu().numpy()*(max_x_y - min_x_y) + min_x_y
        # new_constructed_Y = new_constructed_Y.data.cpu().numpy()*std_x_y + mean_x_y

        pred_Y_epoch_list.append(new_constructed_Y)
        if torch.sum(new_eval_mask, dim=0).item() == 0:
            continue
        # online validation
        with torch.no_grad():
            gnn.eval()
            regressor.eval()
            x_embd = gnn(new_X, new_edge_index)
            pred = regressor(x_embd)
            valid_indices = torch.where(new_eval_mask == 1)[0]
            valid_loss = F.mse_loss(pred[valid_indices], new_Y[valid_indices])/(len(valid_indices) + 1e-5)
            valid_loss = valid_loss.item()
            print('{n} epoch valid loss:'.format(n=on_epoch), valid_loss)
            valid_loss_list.append(valid_loss)
            y_pred = (new_eval_mask*pred).data.cpu().numpy()
            y_true = (torch.nan_to_num(new_eval_mask*new_Y, nan=0)).data.cpu().numpy()
            y_pred = y_pred*(max_x_y-min_x_y) + min_x_y
            y_true = y_true*(max_x_y-min_x_y) + min_x_y

            # y_pred = y_pred * std_x_y + mean_x_y
            # y_true = y_true * std_x_y + mean_x_y
            print('pred', y_pred[valid_indices.data.cpu().numpy()])
            print('true', y_true[valid_indices.data.cpu().numpy()])
            p_sqrt_dist = np.sqrt(np.sum(np.square(y_pred - y_true), axis=1))
            # print('p_sqrt_dist', p_sqrt_dist)
            point_error_list.append(round(np.sum(p_sqrt_dist) / (np.sum(new_eval_mask.data.cpu().numpy()) + 1e-5), 6))
            print('predicted point error:', round(np.sum(p_sqrt_dist) / (np.sum(new_eval_mask.data.cpu().numpy()) + 1e-5), 6))
        # cnt += 1
        # if cnt > 3:
        #     break
    print('min point_error', min(point_error_list))
    if not point_error_list:
        current_Y = torch.FloatTensor(pred_Y_epoch_list[-1]).to(device)
    else:
        assert len(point_error_list) == ontrain_epochs, 'error in online training'
        argmin_index = np.argmin(point_error_list)
        current_Y = torch.FloatTensor(pred_Y_epoch_list[argmin_index]).to(device)
    # calculate time cost each time
    online_training_et = time.time()
    consumed_time = online_training_et-online_training_st
    print('{n} online training consumed time:{t}'.format(n=batch_i, t=consumed_time))

    fig1, ax1 = plt.subplots()
    ax1.plot(range(len(online_train_losses[:150])), online_train_losses[:150])
    plt.savefig('./on_train_loss.png')

    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(valid_loss_list[:150])), valid_loss_list[:150])
    plt.savefig('./on_valid_loss.png')
    #update "current" parameters
    current_node_index = new_X.shape[0]
    current_X = copy.copy(new_X)
    if training_way == 0:
        current_edge_index = copy.copy(new_edge_index)
    elif training_way == 1:
        current_edge_index = torch.cat([current_edge_index, temp_edge_index_ts], dim=1)
    current_mask = torch.ones((new_X.shape[0], 1), dtype=float).to(device)
    break


'''






