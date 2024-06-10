import numpy as np
import os
import argparse
from module import LLAutoEncoder, ALAutoEncoder, AAAutoEncoder
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from utils import *
import minisom

parser = argparse.ArgumentParser(description='SOM-based Explainable Recommendation model(ASC)')
parser.add_argument('--rawdata_path', type=str, default='data/raw',
                    help='path for raw data')
parser.add_argument('--input_dim', type=int, default=104,
                    help='the dim of input data')
parser.add_argument('--feature_dim', type=int, default=104,
                    help='the number of features')
parser.add_argument('--hidden_dim', type=int, default=128,
                    help='the dim of the AE hidden layer')
parser.add_argument('--latend_dim', type=int, default=32,
                    help='the dim of the AE encoder output')
parser.add_argument('--output_dim', type=int, default=104,
                    help='the dim of the AE decoder output')
parser.add_argument('--lr', type=float, default=0.001,
                    help='the learning rate of AE')
parser.add_argument('--ae_epochs', type=int, default=200,
                    help='the number of epoch to train AE')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--m', type=int, default=20,
                    help='the size of som')
parser.add_argument('--n', type=int, default=20,
                    help='the size of som')
parser.add_argument('--som_dim', type=int, default=104,
                    help='the dim of som node')
parser.add_argument('--som_epochs', type=int, default=20,
                    help='the epochs to train som')
parser.add_argument('--n_mask', type=int, default=1000,
                    help='the number of masks')
parser.add_argument('--mask_rate', type=float, default=0.1,
                    help='mask rate in counterfactual inference')
parser.add_argument('--process_rawdata', action='store_true', default=True,
                    help='True to process raw data')
parser.add_argument('--ae', type=int, default=0,
                    help='0: MLP-MLP, 1: transformer encoder-MLP')
parser.add_argument('--model_param_path', type=str, default='model/pths/',
                    help='the path of AE param')
parser.add_argument('--som_train_itera', type=int, default=20,
                    help='train som itera')
parser.add_argument('--sigma', type=int, default=3,
                    help='som sigma')
parser.add_argument('--encoder', action='store_true', default=False)
parser.add_argument('--head_n', type=int, default=4)
parser.add_argument('--ffn_hidden', type=int, default=1024)
parser.add_argument('--layer_n', type=int, default=2)
parser.add_argument('--drop_prob', type=float, default=0.2)
args = parser.parse_args()

if args.process_rawdata is True:
    data_process(args.rawdata_path, args.feature_dim)

fea_d = args.feature_dim
input_d = args.input_dim
hidden_d = args.hidden_dim
latend_d = args.latend_dim
output_d = args.output_dim
lr = args.lr
total_epochs = args.ae_epochs
batch_size = args.batch_size

m, n, dim = args.m, args.n, args.som_dim
num_epoch = args.som_epochs
sigma = args.sigma

head_n = args.head_n
ffn_hidden = args.ffn_hidden
layer_n = args.layer_n
drop_prob = args.drop_prob

n_umask = args.n_mask
mask_rate = args.mask_rate
model_param_path = args.model_param_path

itera = args.som_train_itera

# 训练自编码器
if args.encoder == 0:
    user_autoencoder = LLAutoEncoder(input_d, hidden_d, latend_d, output_d)
    item_autoencoder = LLAutoEncoder(input_d, hidden_d, latend_d, output_d)
elif args.encoder == 1:
    user_autoencoder = ALAutoEncoder(head_n, ffn_hidden, layer_n, input_d, drop_prob, latend_d, hidden_d, output_d)
    item_autoencoder = ALAutoEncoder(head_n, ffn_hidden, layer_n, input_d, drop_prob, latend_d, hidden_d, output_d)

criterion = nn.MSELoss()
optimizer_u = torch.optim.Adam(user_autoencoder.parameters(), lr=lr)
optimizer_i = torch.optim.Adam(item_autoencoder.parameters(), lr=lr)

ufs = data_loader('pro/user_fea_norm.txt')
ufs = torch.Tensor(ufs).unsqueeze(dim=1)
ifs = data_loader('pro/item_fea_norm.txt')
ifs = torch.Tensor(ifs).unsqueeze(dim=1)

print("----- 训练物品特征自编码器 -----")
train_loader = DataLoader(ifs, batch_size=batch_size, shuffle=True)
for epoch in range(total_epochs):
    total_loss = 0.
    for _ifs in train_loader:
        inputs = _ifs
        optimizer_i.zero_grad()
        outputs = item_autoencoder(inputs)
        loss = criterion(outputs, inputs)
        total_loss += loss
        loss.backward()
        optimizer_i.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, total_epochs, total_loss))

print("----- 训练用户特征自编码器 -----")
train_loader = DataLoader(ufs, batch_size=batch_size, shuffle=True)
for epoch in range(total_epochs):
    total_loss = 0.
    for _ifs in train_loader:
        inputs = _ifs
        optimizer_u.zero_grad()
        outputs = user_autoencoder(inputs)
        loss = criterion(outputs, inputs)
        total_loss += loss
        loss.backward()
        optimizer_u.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, total_epochs, total_loss))

print("----- 保存用户编码器参数 -----")
torch.save(user_autoencoder.encoder, model_param_path+'uf_encoder.pth')
print("----- 保存用户解码器参数 -----")
torch.save(user_autoencoder.decoder, model_param_path+'uf_decoder.pth')
print("----- 保存物品编码器参数 -----")
torch.save(item_autoencoder.encoder, model_param_path+'if_encoder.pth')
print("----- 保存物品解码器参数 -----")
torch.save(item_autoencoder.decoder, model_param_path+'if_decoder.pth')

# 对用户, 项目特征进行编码
print("----- 编码用户特征 -----")
uf_encoder = torch.load(model_param_path+'uf_encoder.pth')
uf_decoder = torch.load(model_param_path+'uf_decoder.pth')
users = data_loader('pro/user_fea_norm.txt')
users_enc = []
users_hiddenfea = []
for u in users:
    u = torch.Tensor(u)
    u = u.unsqueeze(dim=0).unsqueeze(dim=1)
    enc = uf_encoder(u)
    users_hiddenfea.append(enc.detach().numpy()[0][0])
    enc = uf_decoder(enc)
    users_enc.append(enc.detach().numpy()[0][0])
print("----- 保存用户特征编码 -----")
np.savetxt("data/pro/users_hiddenfea.txt", users_hiddenfea, fmt='%f', delimiter=',')
np.savetxt("data/pro/users_enc.txt", users_enc, fmt='%f', delimiter=',')

print("----- 编码项目特征 -----")
if_encoder = torch.load(model_param_path+'if_encoder.pth')
if_decoder = torch.load(model_param_path+'if_decoder.pth')
items = data_loader('pro/item_fea_norm.txt')
items_enc = []
items_hiddenfea = []
for i in items:
    i = torch.Tensor(i)
    i = i.unsqueeze(dim=0).unsqueeze(dim=1)
    enc = if_encoder(i)
    items_hiddenfea.append(enc.detach().numpy()[0][0])
    enc = if_decoder(enc)
    items_enc.append(enc.detach().numpy()[0][0])
print("----- 保存项目特征编码 -----")
np.savetxt("data/pro/items_hiddenfea.txt", items_hiddenfea, fmt='%f', delimiter=',')
np.savetxt("data/pro/items_enc.txt", items_enc, fmt='%f', delimiter=',')

# 训练自组织映射网络
som = minisom.MiniSom(m, n, fea_d, sigma=sigma, learning_rate=lr, neighborhood_function='gaussian')
train_data = data_loader('pro/items_enc.txt')
som.random_weights_init(train_data)
som.train_batch(train_data, itera, verbose=False)
weight = som.get_weights()

# 反事实推理
result = []

print("----- 反事实推理 -----")
users = data_loader('pro/users_hiddenfea.txt')
uf_decoder = torch.load(model_param_path+'uf_decoder.pth')

for i in range(20):
    user = users[i]
    bmus = []
    masks = []
    ori_user = uf_decoder(torch.Tensor(user))
    g, h = som.winner(ori_user.detach().numpy())
    print("----- 生成掩码 user {} -----".format(i))
    for j in range(n_umask):
        mask = np.random.rand(latend_d)
        for k in range(len(mask)):
            if mask[k] < mask_rate:
                mask[k] = 0
            else:
                mask[k] = 1
        masks.append(mask)
        mask_user = user * mask
        confac_user = uf_decoder(torch.Tensor(mask_user))
        g_m, h_m = som.winner(confac_user.detach().numpy())
        A = weight[g][h]
        B = weight[g_m][h_m]
        cosine = np.dot(A, B) / (np.linalg.norm(A)*np.linalg.norm(B))
        cos_bmu = np.insert(weight[g_m][h_m], 0, cosine)
        bmus.append(cos_bmu)
    bmus = np.array(bmus)
    ori_user_bmu = np.insert(weight[g][h], 0, 0.)
    bmus = np.insert(bmus, 0, ori_user_bmu).reshape(n_umask+1, fea_d+1)
    idx = bmus[:,0].argsort()
    bmus = bmus[idx]
    np.savetxt('data/cont_infe/user_bmus_{}.txt'.format(i), bmus, fmt='%f', delimiter=',')
    masks = np.insert(masks, 0, np.ones(latend_d)).reshape(n_umask+1, latend_d)
    masks = np.array(masks)
    masks = masks[idx]
    np.savetxt('data/cont_infe/user_mask_matrix_{}.txt'.format(i), masks, fmt='%f', delimiter=',')
    unique_bmus = np.unique(bmus, axis=0)
    unique_masks = []
    for j in range(1, len(unique_bmus)):
        for k in range(len(masks)):
            if bmus[k][0] == unique_bmus[j][0]:
                unique_masks.append(masks[k])
                break
    unique_bmus = np.array(unique_bmus)
    unique_masks = np.array(unique_masks)
    np.savetxt('data/cont_infe/user_bmus_unique_{}.txt'.format(i), unique_bmus, fmt='%f', delimiter=',')
    np.savetxt('data/cont_infe/user_mask_matrix_unique_{}.txt'.format(i), unique_masks, fmt='%f', delimiter=',')

# 生成解释
f = open("data/raw/yelp_recursive.featuremap", 'r')
fea_dict = {}
for line in f.readlines():
    data = line.split('=')
    idx = int(data[0])
    fea_dict[idx] = data[1]

uf_decoder = torch.load("model/pths/uf_decoder.pth")
if_decoder = torch.load("model/pths/if_decoder.pth")

n = 10
explanations = []
ndcg_u = []
ndcg_i = []
HR_u = []
HR_i = []
MRR_u = []
MRR_i = []
for i in range(30): 
    masks = data_loader("cont_infe/user_mask_matrix_unique_{}.txt".format(i))
    user_enc = data_loader("pro/users_hiddenfea.txt")
    
    # 生成物品端 explanation
    ibmus = data_loader("cont_infe/user_bmus_unique_{}.txt".format(i))
    ibmu = ibmus[0][1:]
    ibmu_res_max = ibmus[1][1:]
    res = ibmu - ibmu_res_max
    idx_i = np.argsort(-res)
    exp_i = np.arange(0, 104)
    exp_i = exp_i[idx_i]
    
    # 生成用户视角 explanation
    user = user_enc[i].copy()
    for i in range(1, len(masks)):
        mask = masks[i]
        for j in range(len(user)):
            if mask[j] != 0.:
                user[j] = 0.
    user = torch.Tensor(user)
    ufd = uf_decoder(user).detach().numpy()
    idx_u = np.argsort(-ufd)
    exp_u = np.arange(0, 104)
    exp_u = exp_u[idx_u]
    
    # 用户特征 explanation
    user_ori = user_enc[i].copy()
    user_ori = torch.Tensor(user_ori)
    ufs = uf_decoder(user_ori).detach().numpy()
    idx = np.argsort(-ufs)
    exp = np.arange(0, 104)
    exp = exp[idx]
    
    
    ndcg_u.append(ndcg(exp[:n], exp_u[:n], n))
    ndcg_i.append(ndcg(exp[:n], exp_i[:n], n))
    HR_u.append(hr(exp[:n], exp_u[:n], n))
    HR_i.append(hr(exp[:n], exp_i[:n], n))
    MRR_u.append(mrr(exp[:n], exp_u[:n], n))
    MRR_i.append(mrr(exp[:n], exp_i[:n], n))
    
#     disc = ["用户特征", "用户掩码视角", "物品残差视角"]
#     explanations.append('{:<10} {:<10} {:<10}'.format(disc[0], disc[1], disc[2]))
#     # print('{:<10} {:<10} {:<10}'.format(disc[0], disc[1], disc[2]))
#     for i in range(10):
#         explanations.append('{:<15} {:<15} {:<15}'.format(fea_dict[exp[i]], fea_dict[exp_u[i]], fea_dict[exp_i[i]]))
# #         print('{:<15} {:<15} {:<15}'.format(fea_dict[exp[i]], fea_dict[exp_u[i]], fea_dict[exp_i[i]]))
# explanations = np.array(explanations)
# with open('data/cont_infe/explanations.txt', 'a+', encoding='utf-8') as f:
#     for exp in explanations:
#         f.write(str(exp)+'\n')


ndcg_u = np.array(ndcg_u)
print('ndcg_u@{}: '.format(n), np.mean(ndcg_u))
ndcg_i = np.array(ndcg_i)
print('ndcg_i@{}: '.format(n), np.mean(ndcg_i))
HR_u = np.array(HR_u)
print('HR_u@{}: '.format(n), np.mean(HR_u))
HR_i = np.array(HR_i)
print('HR_i@{}: '.format(n), np.mean(HR_i))
MRR_u = np.array(MRR_u)
print('MRR_u@{}: '.format(n), np.mean(MRR_u))
MRR_i = np.array(MRR_i)
print('MRR_i@{}: '.format(n), np.mean(MRR_i))

result.append('ndcg_u@{}: {}\n'.format(n, np.mean(ndcg_u)))
result.append('ndcg_i@{}: {}\n'.format(n, np.mean(ndcg_i)))
result.append('HR_u@{}: {}\n'.format(n, np.mean(HR_u)))
result.append('HR_i@{}: {}\n'.format(n, np.mean(HR_i)))
result.append('MRR_u@{}: {}\n'.format(n, np.mean(MRR_u)))
result.append('MRR_i@{}: {}\n'.format(n, np.mean(MRR_i)))

with open('rec.txt', 'a+', encoding='utf-8') as f:
    for res in result:
        f.write(str(res))
