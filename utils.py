import os
import numpy as np

def data_loader(file_path):
    f = open("data/{}".format(file_path), 'r')
    features = []
    for line in f.readlines():
        features_s = line.split(',')
        features_i = [float(fea) for fea in features_s]
        features.append(features_i)
    return np.array(features) 

# 返回 BMU在网格中的坐标 (g, h)
def find_BMU(SOM, x):
    distSq = (np.square(SOM - x)).sum(axis=2)
    return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

# 给定一个简单训练样本更新 SOM 单元的权重
def update_weights(SOM, train_ex, learn_rate, radius_sq, BMU_cood, step=3):
    g, h = BMU_cood
    # 如果半径接近 0，那么只更新 BMU
    if radius_sq < 1e-3:
        SOM[g, h, :] += learn_rate * (train_ex - SOM[g, h, :])
        return SOM
    
    # 更新 BMU 在一个小区域内的所有单元
    for i in range(max(0, g-step), min(SOM.shape[0], g+step)):
        for j in range(max(0, h-step), min(SOM.shape[1], h+step)):
            dist_sq = np.square(i - g) + np.square(j - h)
            dist_func = np.exp(-dist_sq / 2 / radius_sq)
            SOM[i,j,:] += learn_rate * dist_func * (train_ex - SOM[i,j,:])   
    return SOM 


def train_SOM(SOM, train_data, learn_rate = .1, radius_sq = 1, 
             lr_decay = .1, radius_decay = .1, epochs = 10):    
    learn_rate_0 = learn_rate
    radius_0 = radius_sq
    for epoch in np.arange(0, epochs):
        np.random.shuffle(train_data)      
        for train_ex in train_data:
            g, h = find_BMU(SOM, train_ex)
            SOM = update_weights(SOM, train_ex, 
                                 learn_rate, radius_sq, (g,h))
        # Update learning rate and radius
        learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
        radius_sq = radius_0 * np.exp(-epoch * radius_decay)            
    return SOM

# 数据预处理
def data_process(path, fea_d):
    user_fea = {}
    item_fea = {}
    f = open(path+'/yelp_recursive.entry', 'r')
    for line in f.readlines():
        uirf = line.split(',')
        u, i, r = int(uirf[0]), int(uirf[1]), int(uirf[2]) / 5.
        # 若用户和项目不在字典中，加入字典
        if u not in user_fea:
            user_fea[u] = [0.] * fea_d
        if i not in item_fea:
            item_fea[i] = [0.] * fea_d
        # 从评论中处理特征
        fea_raw = uirf[3].split(' ')
        fea_review = []
        for frw in fea_raw:
            fr_pro_split = frw.split(':')
            fea_review.append([int(fr_pro_split[0]), int(fr_pro_split[1])])
        for frv in fea_review:
            # 针对用户和项目的特征计算不相同
            user_fea[u][frv[0]] += float(abs(frv[1]))
            item_fea[i][frv[0]] += float(frv[1] * r)

    user_n = len(user_fea)
    item_n = len(item_fea)
    ufea, ifea = [], []
    for i in range(user_n):
        ufea.append(user_fea[i])
    for i in range(item_n):
        ifea.append(item_fea[i])
    ufea = np.array(ufea)
    ifea = np.array(ifea)
    np.savetxt('data/pro/user_fea.txt', ufea, fmt='%f', delimiter=',')
    np.savetxt('data/pro/item_fea.txt', ifea, fmt='%f', delimiter=',')

    # 特征归一化到 0 - 2
    uf_max = np.max(ufea, axis=1)
    uf_min = np.min(ufea, axis=1)
    if_max = np.max(ifea, axis=1)
    if_min = np.min(ifea, axis=1)
    for i in range(user_n):
        ufea[i] = (ufea[i] - uf_min[i]) / (uf_max[i] - uf_min[i]) * 2 
    for i in range(item_n):
        ifea[i] = (ifea[i] - if_min[i]) / (if_max[i] - if_min[i]) * 2 

    save_path = 'data/pro'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.savetxt('data/pro/user_fea_norm.txt', ufea, fmt='%f', delimiter=',')
    np.savetxt('data/pro/item_fea_norm.txt', ifea, fmt='%f', delimiter=',')

def ndcg(ture, rec, n):
    rel = {}
    for i in range(n):
        rel[ture[i]] = n - i
    gain = np.arange(1, n+1)
    log = [np.log2(i+1) for i in range(1, n+1)]
    idcg = 0.
    for i in range(n):
        idcg += rel[ture[i]] * (1. / log[i])
    dcg = 0.
    for i in range(n):
        if rec[i] in rel:
            dcg += rel[rec[i]] * (1. / log[i])
    return dcg / idcg

def mrr(true, rec, n):
    res = 0.
    for i in range(n):
        if rec[i] == true[0]:
            res += 1. / float(i+1)
        if rec[i] == true[1]:
            res += 1. / float(i+1)
        if rec[i] == true[2]:
            res += 1. / float(i+1)
    res *= (18. / 11.) 
    return res / 3.

def hr(true, rec, n):
    hit = 0.
    for fea in rec:
        if fea in true:
            hit += 1
    return hit / float(n)
