import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import pickle
import random
from collections import defaultdict

def node_split(features, split, seed):
    refer = features[100].reshape(1, -1)
    distances = cdist(features, refer, metric='euclidean').flatten()
    sorted_indices = np.argsort(distances)
    train = sorted_indices[:int(len(features)*split)]
    test = sorted_indices[int(len(features)*split):] 

    # tsne = TSNE(n_components=2, random_state=seed)
    # features_tsne = tsne.fit_transform(features) # x,y

    # tsne_group_1 = features_tsne[train, :]
    # tsne_group_2 = features_tsne[test, :]
    # plt.figure(figsize=(8, 6))
    # plt.scatter(tsne_group_1[:, 0], tsne_group_1[:, 1], c='red', label='Train Node', edgecolor='k', alpha=0.7)
    # plt.scatter(tsne_group_2[:, 0], tsne_group_2[:, 1], c='blue', label='Test Node', edgecolor='k', alpha=0.7)

    # plt.title('Distribution Visualization of Train Node and Test Node')

    # plt.legend()
    # plt.savefig("figure/disease distribution.png", dpi=300)
    # plt.show()
    return train, test

def graph_split(g_train, d_train, g_fea, d_fea,
        g_g,
        d_d,
        g_d,
        neg_g_d,
        path):
    sort_g_train = np.sort(g_train)
    sort_d_train = np.sort(d_train)
    pd.DataFrame.from_dict({'gene':sort_g_train,'disease':sort_d_train}, orient='index').transpose().to_excel(path + 'node_index_map.xlsx', index=False)

    train_g_fea = g_fea[sort_g_train, :]
    train_d_fea = d_fea[sort_d_train, :]

    train_g_g = g_g[np.isin(g_g[:, 0], sort_g_train) & np.isin(g_g[:, 1], sort_g_train)]
    train_d_d = d_d[np.isin(d_d[:, 0], sort_d_train) & np.isin(d_d[:, 1], sort_d_train)]
    train_g_d = g_d[np.isin(g_d[:, 0], sort_d_train) & np.isin(g_d[:, 1], sort_g_train)]
    train_neg_g_d = neg_g_d[np.isin(neg_g_d[:, 0], sort_d_train) & np.isin(neg_g_d[:, 1], sort_g_train)]

    return {'g_fea':train_g_fea, 'd_fea':train_d_fea,
            'g_g':train_g_g, 'd_d':train_d_d, 'g_d':train_g_d, 'neg_g_d':train_neg_g_d}


def get_node(relation1, relation2, node1, node2):
    if relation2 is None :
        return np.sort(np.unique(relation1[np.isin(relation1[:, 1], node1), 0])).astype(np.int64)

    filt1 = relation1[np.isin(relation1[:, 1], node1), 0]
    filt2 = relation2[np.isin(relation2[:, 1], node2), 0]
    return np.sort(np.unique(np.concatenate((filt1, filt2)))).astype(np.int64)

def split_data(path, splitRate, seed):
    # if checkDataset(path):
    #     with open(path + 'train.pkl', 'rb') as f:
    #         train_data = pickle.load(f)
    #     with open(path + 'test.pkl', 'rb') as f:
    #         test_data = pickle.load(f)
    #     g_num = len(pd.read_csv('Dataset/gene_feature.csv').values)
    #     d_num = len(pd.read_csv('Dataset/dis_sim_fea.csv', header=None).values)
    #     m_num = len(pd.read_csv('Dataset/mirna_fea.csv', header=None).values)     
    #     return train_data, test_data, g_num, d_num, m_num

    g_fea = pd.read_csv('Dataset/gene_feature.csv').values
    scaler = RobustScaler()
    g_fea = scaler.fit_transform(g_fea)
    d_l_fea = pd.read_csv('Dataset/dis_biobert_fea.csv', header=None).values
    d_sim_fea = pd.read_csv('Dataset/dis_sim_fea.csv', header=None).values
    d_fea = np.hstack((d_l_fea, d_sim_fea))

    g_train, g_test = node_split(g_fea, splitRate, seed)
    d_train, d_test = node_split(d_fea, splitRate, seed)

    # all of them are edge index
    g_g = pd.read_csv('Dataset/gene_adj.tsv', sep='\t', header=None).iloc[:, :2].values
    d_d = pd.read_csv('Dataset/dis_adj.csv', header=None).values
    d_d = torch.nonzero(torch.triu(torch.tensor(d_d) > 0), as_tuple=False)
    g_d = pd.read_csv('Dataset/disease_gene_pos.csv').values
    neg_g_d = np.empty((0, 2), dtype=int)
    g_num, d_num = len(g_fea), len(d_fea)
    gd_net= np.zeros((g_num, d_num))
    gd_net[g_d[:,1].astype(int), g_d[:, 0].astype(int)] = g_d[:, 2]
    g_fea = np.hstack((g_fea, gd_net))
    d_fea = np.hstack((d_fea, gd_net.T))

    train_data = graph_split(g_train,d_train,g_fea,d_fea,
        g_g,d_d,
        g_d,neg_g_d,
        path+'train')
    test_data= graph_split(g_test,d_test,g_fea,d_fea,
        g_g,d_d,
        g_d,neg_g_d,
        path+'test')
    data_save(path, train_data, test_data)
    return train_data, test_data, g_num, d_num

def data_save(folder, train, test):
    train_path = folder + 'train.pkl'
    test_path = folder + 'test.pkl'

    with open(train_path, 'wb') as f:
        pickle.dump(train, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test, f)

def checkDataset(file_path):
    if os.path.exists(file_path):
        files = os.listdir(file_path)
        if files:
            return True
        return False
    else:
        os.makedirs(file_path)
        return False


def graph_constract(data, map, seed, flag=1):
    node_file = pd.read_excel(map)
    node_dict = []
    for column in node_file.columns: # 0gene 1disease 2miRNA
        column_data = node_file[column].dropna().astype(int).tolist()
        node_dict.append(column_data)
    node_dict = [np.array(node_dict[i]) for i in range(len(node_dict))]
    
    graph = HeteroData()
    graph['gene'].x = torch.tensor(data['g_fea'], dtype=torch.float32)
    graph['disease'].x = torch.tensor(data['d_fea'], dtype=torch.float32)

    # edge index mapping
    def get_edge_index(data_key, node_idx_1, node_idx_2):
        rows = np.array(data[data_key])
        return np.column_stack((np.searchsorted(node_dict[node_idx_1], rows[:, 0]), 
                                np.searchsorted(node_dict[node_idx_2], rows[:, 1])))
    
    g_g_edge_index = get_edge_index('g_g', 0, 0)
    d_d_edge_index = get_edge_index('d_d', 1, 1)
    g_d_edge_index = get_edge_index('g_d', 1, 0)
    neg_g_d_edge_index = get_edge_index('neg_g_d', 1, 0)

    graph['gene', 'to', 'gene'].edge_index = torch.tensor(g_g_edge_index, dtype=torch.long).T
    graph['disease', 'to', 'disease'].edge_index = torch.tensor(d_d_edge_index, dtype=torch.long).T
    np.random.seed(seed)
    np.random.shuffle(g_d_edge_index)

    message_g_d = g_d_edge_index[0:int(len(g_d_edge_index) * 0.5)]
    leave_g_d = g_d_edge_index[int(len(g_d_edge_index) * 0.5):]

    graph['disease', 'to', 'gene'].edge_index = torch.tensor(message_g_d, dtype=torch.long).T
    graph = T.ToUndirected()(graph)

    if flag == 1:
        message_edge = pd.DataFrame(message_g_d)
        message_edge.to_csv('Dataset/Data/message_train_edge.csv', index=False, header=['disease', 'gene'])
        return graph, len(data['g_fea']), len(data['d_fea']), g_g_edge_index, d_d_edge_index, message_g_d, leave_g_d, neg_g_d_edge_index
    
    else:
        message_edge = pd.DataFrame(message_g_d)
        message_edge.to_csv('Dataset/Data/message_test_edge.csv', index=False, header=['disease', 'gene'])
        test_edge = pd.DataFrame(leave_g_d)
        test_edge.to_csv('Dataset/Data/test_pos_edge.csv', index=False, header=['disease', 'gene'])
        return graph, len(data['g_fea']), len(data['d_fea']), g_g_edge_index, d_d_edge_index, message_g_d, leave_g_d, neg_g_d_edge_index

def Random_Sampleing(g_num, d_num, message_edge, pos_edge, core_neg_edge, flag = 1, seed=42):
    random.seed(seed)
    
    message_edge_set = set(map(tuple, message_edge))
    pos_edge_set = set(map(tuple, pos_edge))
    core_neg_edge_set = set(map(tuple, core_neg_edge))
    
    if flag == 1:
        new_neg_edge = []
        while len(new_neg_edge) < len(pos_edge) - len(core_neg_edge):
            g_node = random.randint(0, g_num-1)
            d_node = random.randint(0, d_num-1)
            pair = (d_node, g_node)
            if pair in pos_edge_set or pair in core_neg_edge_set or pair in message_edge_set:
                continue
            new_neg_edge.append(pair)
        return np.vstack((core_neg_edge, np.array(new_neg_edge)))

    else:
        new_neg_edge = []
        while len(new_neg_edge) < len(pos_edge):
            g_node = random.randint(0, g_num-1)
            d_node = random.randint(0, d_num-1)
            pair = (d_node, g_node)
            if pair in pos_edge_set or pair in core_neg_edge_set or pair in message_edge_set:
                continue
            new_neg_edge.append(pair)
        return np.array(new_neg_edge)