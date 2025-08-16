import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch_geometric.data import HeteroData
from scipy.spatial.distance import pdist, squareform

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

    reducer = TSNE(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(all_emb)
    
    num_points = len(all_emb)
    shuffled_indices = np.random.permutation(num_points)

    embedding_2d_shuffled = embedding_2d[shuffled_indices]
    labels_shuffled = labels[shuffled_indices]

    scatter = ax.scatter(embedding_2d_shuffled[:, 0], embedding_2d_shuffled[:, 1], 
                         c=labels_shuffled, 
                         cmap='coolwarm', 
                         s=30, 
                         alpha=0.7)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    handles, _ = scatter.legend_elements(prop='colors', alpha=0.8)
    ax.legend(handles, ['Negative Samples', 'Positive Samples'], loc='best')

def get_embeddings(folds_dict, search_node_type):
    row, col = folds_dict[0][search_node_type].shape
    average = np.zeros((row, col))
    for fold in folds_dict:
        average += fold[search_node_type]
    return average/len(folds_dict)