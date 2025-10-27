import numpy as np
import torch
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch_geometric.data import HeteroData
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from matplotlib.lines import Line2D

def obtain_embbedings(graph, pos_edge, neg_edge):
    pos_embedding = []
    neg_embedding = []
    if isinstance(graph, HeteroData):
        for edge in pos_edge:
            pos_embedding.append(torch.cat((graph['disease'].x[edge[0]], graph['gene'].x[edge[1]])).detach().cpu().numpy())
        for edge in neg_edge:
            neg_embedding.append(torch.cat((graph['disease'].x[edge[0]], graph['gene'].x[edge[1]])).detach().cpu().numpy())

    else:
        for edge in pos_edge:
            pos_embedding.append(np.hstack((graph['disease'][edge[0]], graph['gene'][edge[1]])))
        for edge in neg_edge:
            neg_embedding.append(np.hstack((graph['disease'][edge[0]], graph['gene'][edge[1]])))
    return pos_embedding, neg_embedding

def visualize_embeddings_with_umap(pos_embedding, neg_embedding, ax, title):
    def _to_2d_numpy(embeddings_input):
        if isinstance(embeddings_input, list):
            if isinstance(embeddings_input[0], torch.Tensor):
                return torch.stack(embeddings_input).detach().cpu().numpy()
            else:
                return np.array(embeddings_input)
        elif isinstance(embeddings_input, torch.Tensor):
            return embeddings_input.detach().cpu().numpy()
        elif isinstance(embeddings_input, np.ndarray):
            return embeddings_input
        else:
            raise TypeError("Input embeddings must be a list of Tensors/arrays or a single Tensor/array.")

    pos_embedding_np = _to_2d_numpy(pos_embedding)
    neg_embedding_np = _to_2d_numpy(neg_embedding)

    all_emb = np.vstack((pos_embedding_np, neg_embedding_np))
    labels = np.array([1] * len(pos_embedding_np) + [0] * len(neg_embedding_np))
    
    perplexity =  min(30, len(labels) - 1)

    reducer = TSNE(n_components=2, random_state=2341, perplexity=perplexity)
    embedding_2d = reducer.fit_transform(all_emb)
    
    num_points = len(all_emb)
    np.random.seed(42)
    shuffled_indices = np.random.permutation(num_points)

    embedding_2d_shuffled = embedding_2d[shuffled_indices]
    labels_shuffled = labels[shuffled_indices]

    scatter = ax.scatter(embedding_2d_shuffled[:, 0], embedding_2d_shuffled[:, 1], 
                         c=labels_shuffled, 
                         cmap='coolwarm', 
                         s=30, 
                         alpha=0.7)
    ax.set_title(title, fontsize=12)
    ax.grid(False)

    handles, _ = scatter.legend_elements(prop='colors', alpha=0.8)
    ax.legend(handles, ['Negative Samples', 'Positive Samples'], loc='best')
    
def visualize_embeddings_with_umap_v2(pos_embedding, neg_embedding, ax, title):
    def _to_2d_numpy(embeddings_input):
        if isinstance(embeddings_input, list):
            if isinstance(embeddings_input[0], torch.Tensor):
                return torch.stack(embeddings_input).detach().cpu().numpy()
            else:
                return np.array(embeddings_input)
        elif isinstance(embeddings_input, torch.Tensor):
            return embeddings_input.detach().cpu().numpy()
        elif isinstance(embeddings_input, np.ndarray):
            return embeddings_input
        else:
            raise TypeError("Input embeddings must be a list of Tensors/arrays or a single Tensor/array.")

    pos_embedding_np = _to_2d_numpy(pos_embedding)
    neg_embedding_np = _to_2d_numpy(neg_embedding)

    all_emb = np.vstack((pos_embedding_np, neg_embedding_np))
    labels = np.array([1] * len(pos_embedding_np) + [0] * len(neg_embedding_np))
    
    perplexity =  min(30, len(labels) - 1)

    reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embedding_2d = reducer.fit_transform(all_emb)
    
    df = pd.DataFrame({
        'dim1': embedding_2d[:, 0],
        'dim2': embedding_2d[:, 1],
        'label': labels
    })
    
    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]
    sns.kdeplot(
        data=neg_df, x='dim1', y='dim2', ax=ax,
        fill=True,
        alpha=0.6,
        cmap='Blues',
        levels=6
    )
    
    sns.kdeplot(
        data=pos_df, x='dim1', y='dim2', ax=ax,
        fill=True, 
        alpha=0.6,
        cmap='Reds',
        levels=6
    )
    
    
    num_points = len(all_emb)
    np.random.seed(42)
    shuffled_indices = np.random.permutation(num_points)

    embedding_2d_shuffled = embedding_2d[shuffled_indices]
    labels_shuffled = labels[shuffled_indices]

    scatter = ax.scatter(embedding_2d_shuffled[:, 0], embedding_2d_shuffled[:, 1], 
                         c=labels_shuffled, 
                         cmap='coolwarm', 
                         s=30, 
                         alpha=0.7)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)

    handles, _ = scatter.legend_elements(prop='colors', alpha=0.8)
    ax.legend(handles, ['Negative Samples', 'Positive Samples'], loc='best')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Positive Samples',
                              markerfacecolor='#E65B5B', markersize=10), # 暖色代表
                       Line2D([0], [0], marker='o', color='w', label='Negative Samples',
                              markerfacecolor='#6A8DC3', markersize=10)] # 冷色代表

def get_embeddings(folds_dict, search_node_type):
    row, col = folds_dict[0][search_node_type].shape
    average = np.zeros((row, col))
    for fold in folds_dict:
        average += fold[search_node_type]
    return average/len(folds_dict)