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

def load_and_prepare_data(dataset, seed):
    print("--- Loading and Preparing Data ---")
    data_path = "Dataset/" + dataset
    
    # Load edges
    disesae_adj_df = pd.read_csv(data_path + "disease_mesh_sim.csv", header=None).values
    disesae_adj = torch.nonzero(torch.triu(torch.tensor(disesae_adj_df) > 0), as_tuple=False).numpy()
    gene_adj = pd.read_csv(data_path + "gene_adj.csv").iloc[:, :2].values
    
    # Load features
    gene_fea_raw = pd.read_csv(data_path + "gene_expression.csv").values
    scaler = RobustScaler()
    g_fea = scaler.fit_transform(gene_fea_raw)
    d_l_fea = pd.read_csv(data_path + "disease_biobert_fea.csv", header=None).values

    # Load and split disease-gene associations
    if dataset == "DisGeNET/":
        dis_gene_df = pd.read_csv(data_path + "DisGeNET.csv")
    elif "DISEASES" in dataset:
        dis_gene_df = pd.read_csv(data_path + "DISEASES.csv")
    disease_num = max(dis_gene_df['disease'].values.squeeze()) + 1
    gene_num = max(dis_gene_df['gene'].values.squeeze()) + 1
    
    dis_gene_assoc = dis_gene_df.values
    np.random.seed(seed)
    np.random.shuffle(dis_gene_assoc)
    
    # Split associations: 50% for message passing, 50% for supervision (training/validation/testing)
    split_idx = int(len(dis_gene_assoc) * 0.5)
    message_dis_gene_assoc = dis_gene_assoc[0:split_idx]
    supervision_dis_gene_assoc = dis_gene_assoc[split_idx:]
    
    # Sample negative edges for the supervision set
    supervision_pos_edges = supervision_dis_gene_assoc[:, :2].astype(int)
    message_edges_for_sampling = message_dis_gene_assoc[:, :2].astype(int)
    neg_g_d = Random_Sampleing(gene_num, disease_num, message_edges_for_sampling, supervision_pos_edges, np.empty((0, 2), dtype=int), 1, seed).astype(int)
    
    print(f"Data loaded. Disease nodes: {disease_num}, Gene nodes: {gene_num}")
    print(f"Message passing d-g edges: {len(message_dis_gene_assoc)}")
    print(f"Supervision d-g edges: {len(supervision_dis_gene_assoc)}")
    
    return (g_fea, d_l_fea, gene_adj, disesae_adj, disease_num, gene_num,
            message_dis_gene_assoc, supervision_dis_gene_assoc, neg_g_d)

def build_graph_for_fold(base_g_fea, base_d_fea, gene_adj, disease_adj,
                         message_assoc, training_assoc, assoc, g_num, d_num):
    feature_construction_edges = np.vstack([
        message_assoc[:, :2],
        training_assoc[:, :2]
    ]).astype(int)
    
    feature_construction_assoc = np.vstack([message_assoc, assoc])

    # Calculate disease similarity using the combined training edges
    d_sim_fea = cal_disease_sim(feature_construction_edges, d_num, g_num)

    gd_net = np.zeros((g_num, d_num))
    gd_net[feature_construction_assoc[:, 1].astype(int), feature_construction_assoc[:, 0].astype(int)] = feature_construction_assoc[:, 2]


    g_fea = np.hstack((base_g_fea, gd_net))
    d_fea = np.hstack((base_d_fea, d_sim_fea, gd_net.T))

    #  Build the HeteroData graph object
    graph = HeteroData()
    graph['gene'].x = torch.tensor(g_fea, dtype=torch.float32)
    graph['disease'].x = torch.tensor(d_fea, dtype=torch.float32)
    
    message_edges = feature_construction_assoc[:, :2].astype(int)
    graph['gene', 'to', 'gene'].edge_index = torch.tensor(gene_adj, dtype=torch.long).T
    graph['disease', 'to', 'disease'].edge_index = torch.tensor(disease_adj, dtype=torch.long).T
    graph['disease', 'to', 'gene'].edge_index = torch.tensor(message_edges, dtype=torch.long).T
    
    graph = T.ToUndirected()(graph)
    
    return graph


def build_graph_for_other_dataset(gene_expression, disease_llm_feature, gene_adj, disease_adj,
                                  message_edge, assoc_dg, assoc_gd, old_d_num, old_g_num, d_num, g_num):
    # note that assoc must be the associations between test diseases and train genes
    train_assoc_net = pd.read_csv("Dataset/DisGeNET/DisGeNET.csv")
    d_sim_fea = cal_disease_sim_for_other_dataset(train_assoc_net, assoc_dg, d_num, old_g_num, 1.0)
    
    dg_net = np.zeros((d_num, old_g_num))
    gd_net = np.zeros((g_num, old_d_num))
    assoc_np = assoc_dg.values
    dg_net[assoc_np[:, 0].astype(int), assoc_np[:, 1].astype(int)] = assoc_np[:, 2]
    assoc_np = assoc_gd.values
    gd_net[assoc_np[:, 1].astype(int), assoc_np[:, 0].astype(int)] = assoc_np[:, 2]
    
    g_fea = np.hstack((gene_expression, gd_net))
    d_fea = np.hstack((disease_llm_feature, d_sim_fea, dg_net))
    
    #  Build the HeteroData graph object
    graph = HeteroData()
    graph['gene'].x = torch.tensor(g_fea, dtype=torch.float32)
    graph['disease'].x = torch.tensor(d_fea, dtype=torch.float32)
    
    message_edges = message_edge[:, :2].astype(int)
    graph['gene', 'to', 'gene'].edge_index = torch.tensor(gene_adj, dtype=torch.long).T
    graph['disease', 'to', 'disease'].edge_index = torch.tensor(disease_adj, dtype=torch.long).T
    graph['disease', 'to', 'gene'].edge_index = torch.tensor(message_edges, dtype=torch.long).T
    
    graph = T.ToUndirected()(graph)
    
    return graph
    

def graph_constract(dataset, seed):
    data_path = "Dataset/" + dataset
    
    # edge
    disesae_adj = pd.read_csv(data_path+"disease_mesh_sim.csv", header=None).values
    disesae_adj = torch.nonzero(torch.triu(torch.tensor(disesae_adj) > 0), as_tuple=False)
    gene_adj = pd.read_csv(data_path+"gene_adj.csv").iloc[:, :2].values
    
    # feature
    gene_fea = pd.read_csv(data_path+"gene_expression.csv").values
    scaler = RobustScaler()
    g_fea = scaler.fit_transform(gene_fea)
    d_l_fea = pd.read_csv(data_path+"disease_biobert_fea.csv", header=None).values

    if dataset == "DisGeNET/":
        dis_gene = pd.read_csv(data_path+"DisGeNET.csv")
        disease_num = max(dis_gene['disease'].values.squeeze()) + 1
        gene_num = max(dis_gene['gene'].values.squeeze()) + 1
        
        dis_gene = dis_gene.values
        np.random.seed(seed)
        np.random.shuffle(dis_gene)
        message_dis_gene = dis_gene[:, :2].astype(int)[0:int(len(dis_gene) * 0.5)] # edge
        
        d_sim_fea = cal_disease_sim(message_dis_gene, disease_num, gene_num)
        # feature concat
        gd_net= np.zeros((gene_num, disease_num))
        gd_net[dis_gene[:,1].astype(int), dis_gene[:, 0].astype(int)] = dis_gene[:, 2]
        d_fea = np.hstack((d_l_fea, d_sim_fea))
        g_fea = np.hstack((g_fea, gd_net))
        d_fea = np.hstack((d_fea, gd_net.T))
        
        graph = HeteroData()
        graph['gene'].x = torch.tensor(g_fea, dtype=torch.float32)
        graph['disease'].x = torch.tensor(d_fea, dtype=torch.float32)
        
        
        graph['gene', 'to', 'gene'].edge_index = torch.tensor(gene_adj, dtype=torch.long).T
        graph['disease', 'to', 'disease'].edge_index = torch.tensor(disesae_adj, dtype=torch.long).T
        graph['disease', 'to', 'gene'].edge_index = torch.tensor(message_dis_gene, dtype=torch.long).T
        graph = T.ToUndirected()(graph)
        
        pos_g_d = dis_gene[:, :2][int(len(dis_gene) * 0.5):].astype(int)
        neg_g_d = Random_Sampleing(gene_num, disease_num, message_dis_gene, pos_g_d, np.empty((0, 2), dtype=int), 1, seed).astype(int)
        
        return graph, disease_num, gene_num, pos_g_d, neg_g_d

def cal_disease_sim(dis_gene, disease_num, gene_num, gamma_d_prime=1.0):
    ip = np.zeros((disease_num, gene_num))
    for d, g in dis_gene:
        ip[d, g] = 1
    
    norm_vals = np.sum(ip**2, axis=1)
    gamma_d = gamma_d_prime / (np.mean(norm_vals[norm_vals > 0]))
    d_sim_fea = np.zeros((disease_num, disease_num))
    for i in range(0, disease_num):
        for j in range(0, disease_num):
            diff = ip[i] - ip[j]
            d_sim_fea[i, j] = np.exp(-gamma_d * np.dot(diff, diff))
            
    return d_sim_fea

def cal_disease_sim_for_other_dataset(train_assoc_net, assoc, d_num, g_num, gamma_d_prime=1.0):
    old_disease_ids = train_assoc_net['disease'].unique()
    num_old_diseases = len(old_disease_ids)
    old_disease_map = {uid: i for i, uid in enumerate(old_disease_ids)}
    
    new_disease_ids = assoc['disease'].unique()
    if d_num != len(new_disease_ids):
        print(f"Warning: Provided d_num ({d_num}) does not match unique diseases in assoc.csv ({len(new_disease_ids)}). Using the latter.")
        d_num = len(new_disease_ids)
    new_disease_map = {uid: i for i, uid in enumerate(new_disease_ids)}
    
    ip_old = np.zeros((num_old_diseases, g_num))
    ip_new = np.zeros((d_num, g_num))
    
    for index, row in train_assoc_net.iterrows():
        disease_idx = old_disease_map[row['disease']]
        gene_idx = int(row['gene'])
        ip_old[disease_idx, gene_idx] = 1

    for index, row in assoc.iterrows():
        disease_idx = new_disease_map[row['disease']]
        gene_idx = int(row['gene'])
        ip_new[disease_idx, gene_idx] = 1
        
    squared_norms_old = np.linalg.norm(ip_old, axis=1)**2
    avg_norm_sq = np.mean(squared_norms_old)
    if avg_norm_sq == 0:
        gamma_d = 1.0
    else:
        gamma_d = gamma_d_prime / avg_norm_sq
    
    similarity_matrix = np.zeros((d_num, num_old_diseases))
    for i in range(d_num):
        for j in range(num_old_diseases):
            ip_new_vec = ip_new[i, :]
            ip_old_vec = ip_old[j, :]

            diff_norm_sq = np.sum((ip_new_vec - ip_old_vec)**2)
            similarity = np.exp(-gamma_d * diff_norm_sq)
            similarity_matrix[i, j] = similarity
            
    return similarity_matrix
    

def graph_scl(train, test):
    scaler_g, scaler_d, scaler_m, scaler_l, scaler_c = MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
    train.x_dict['gene'] = scaler_g.fit_transform(train.x_dict['gene'])
    test.x_dict['gene'] = scaler_g.transform(test.x_dict['gene'])
    train.x_dict['disease'] = scaler_d.fit_transform(train.x_dict['disease'])
    test.x_dict['disease'] = scaler_d.transform(test.x_dict['disease'])
    train.x_dict['miRNA'] = scaler_m.fit_transform(train.x_dict['miRNA'])
    test.x_dict['miRNA'] = scaler_m.transform(test.x_dict['miRNA'])
    train.x_dict['lncRNA'] = scaler_l.fit_transform(train.x_dict['lncRNA'])
    test.x_dict['lncRNA'] = scaler_l.transform(test.x_dict['lncRNA'])
    train.x_dict['circRNA'] = scaler_c.fit_transform(train.x_dict['circRNA'])
    test.x_dict['circRNA'] = scaler_c.transform(test.x_dict['circRNA'])


def Structual_SampleNeg(g_num, d_num, g_g, d_d, pos_edge, core_neg_edge, val_edge, seed):
    random.seed(seed)

    g_adj = defaultdict(set)
    for u, v in g_g:
        g_adj[u].add(v)
        g_adj[v].add(u)
        
    d_adj = defaultdict(set)
    for u, v in d_d:
        d_adj[u].add(v)
        d_adj[v].add(u)

    forbidden_indices = set()
    if pos_edge.size > 0: forbidden_indices.update(pos_edge[:, 0] * g_num + pos_edge[:, 1])
    if core_neg_edge.size > 0: forbidden_indices.update(core_neg_edge[:, 0] * g_num + core_neg_edge[:, 1])
    if val_edge.size > 0: forbidden_indices.update(val_edge[:, 0] * g_num + val_edge[:, 1])
    
    core_d_neighborhood = set()
    if core_neg_edge.size > 0:
        core_diseases = set(np.unique(core_neg_edge[:, 0]))
        for d_node in core_diseases:
            core_d_neighborhood.update(d_adj[d_node])
        core_d_neighborhood.update(core_diseases)

    core_g_neighborhood = set()
    if core_neg_edge.size > 0:
        core_genes = set(np.unique(core_neg_edge[:, 1]))
        for g_node in core_genes:
            core_g_neighborhood.update(g_adj[g_node])
        core_g_neighborhood.update(core_genes)

    num_candidates = (len(pos_edge) - len(core_neg_edge)) * 2
    if num_candidates <= 0:
        return np.array(core_neg_edge)
        
    candidate_indices = set()
    n_total_possible = d_num * g_num
    while len(candidate_indices) < num_candidates:
        batch_size = int((num_candidates - len(candidate_indices)) * 1.2) + 10
        candidates = np.random.randint(0, n_total_possible, size=batch_size)
        for cand_id in candidates:
            if cand_id not in forbidden_indices and cand_id not in candidate_indices:
                candidate_indices.add(cand_id)
                if len(candidate_indices) >= num_candidates:
                    break

    scored_candidates = []
    for cand_id in candidate_indices:
        d_cand = cand_id // g_num
        g_cand = cand_id % g_num

        cand_d_neighbors = d_adj[d_cand]
        cand_g_neighbors = g_adj[g_cand]

        overlap_d = len(cand_d_neighbors.intersection(core_d_neighborhood))
        overlap_g = len(cand_g_neighbors.intersection(core_g_neighborhood))
        
        score = overlap_d + overlap_g
        scored_candidates.append(((d_cand, g_cand), score))
        
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    num_to_select = len(pos_edge) - len(core_neg_edge)
    selected_neg_edges = np.array([item[0] for item in scored_candidates[:num_to_select]])

    if core_neg_edge.size > 0:
        return np.vstack((core_neg_edge, selected_neg_edges))
    else:
        return selected_neg_edges

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