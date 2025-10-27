import torch
import pandas as pd
from utils import *
from processing_data import *
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def load_visualization_data(fold_num, data_path):
    viz_file_path = os.path.join(data_path, f'visualization_data_fold_{fold_num}.pt')
    if not os.path.exists(viz_file_path):
        print(f"Error: Visualization data file not found at {viz_file_path}")
        return None
    print(f"Loading data from {viz_file_path}...")
    return torch.load(viz_file_path)

def plot_one_disease(disease_index, fold_num, data_path='results/My/'):
    viz_data = load_visualization_data(fold_num, data_path)

    initial_emb_dict = {k: torch.from_numpy(v) for k, v in viz_data['initial_embeddings'].items()}
    learned_emb_dict = {k: torch.from_numpy(v) for k, v in viz_data['learned_embeddings'].items()}

    pos_edges = viz_data['val_pos_edges']
    neg_edges = viz_data['val_neg_edges']
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    
    filter_pos_edges = pos_edges[pos_edges[:, 0] == disease_index]
    filter_neg_edges = neg_edges[neg_edges[:, 0] == disease_index]
    
    original_pos_emb, original_neg_emb = obtain_embbedings(initial_emb_dict, filter_pos_edges, filter_neg_edges)
    best_pos_emb, best_neg_emb = obtain_embbedings(learned_emb_dict, filter_pos_edges, filter_neg_edges)
    visualize_embeddings_with_umap_v2(
        original_pos_emb, 
        original_neg_emb, 
        ax=axes[0], 
        title='Initial Embeddings'
    )
    visualize_embeddings_with_umap_v2(
        best_pos_emb, 
        best_neg_emb, 
        ax=axes[1],
        title='Learned Embeddings (Best Epoch)'
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('figure/'+ str(disease_index)+ '.png', dpi=300)
    
def plot_comparison_for_fold(fold_num, data_path='results/My/'):
    print(f"--- Generating comparison plot for Fold {fold_num} ---")
    
    viz_data = load_visualization_data(fold_num, data_path)
    if not viz_data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    initial_emb_dict = {k: torch.from_numpy(v) for k, v in viz_data['initial_embeddings'].items()}
    original_pos_emb, original_neg_emb = obtain_embbedings(initial_emb_dict, viz_data['val_pos_edges'], viz_data['val_neg_edges'])
    visualize_embeddings_with_umap(
        original_pos_emb, 
        original_neg_emb, 
        ax=axes[0], 
        title='Initial Embeddings'
    )

    fig_init, ax_init = plt.subplots(figsize=(8, 6))
    visualize_embeddings_with_umap(original_pos_emb, original_neg_emb, ax=ax_init, title=f'Initial - Fold {fold_num}')
    fig_init.savefig(f"figure/fold_{fold_num}_initial.png", dpi=300)
    plt.close(fig_init)

    learned_embeddings = viz_data['learned_embeddings']
    if learned_embeddings:
        learned_emb_dict = {k: torch.from_numpy(v) for k, v in learned_embeddings.items()}
        best_pos_emb, best_neg_emb = obtain_embbedings(learned_emb_dict, viz_data['val_pos_edges'], viz_data['val_neg_edges'])
        visualize_embeddings_with_umap(
            best_pos_emb, 
            best_neg_emb, 
            ax=axes[1],
            title='Learned Embeddings (Best Epoch)'
        )
        fig_learn, ax_learn = plt.subplots(figsize=(8, 6))
        visualize_embeddings_with_umap(best_pos_emb, best_neg_emb, ax=ax_learn, title=f'Learned (Best) - Fold {fold_num}')
        fig_learn.savefig(f"figure/fold_{fold_num}_learned.png", dpi=300)
        plt.close(fig_learn)
    else:
        axes[1].text(0.5, 0.5, 'No learned embeddings found.', ha='center', va='center')
        axes[1].set_title('Learned Embeddings (Best Epoch)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("figure/fold embedding.png", dpi=300)
    plt.show()
    
if __name__ == '__main__':
    for i in range(218):
        plot_one_disease(i, 1)
        
    plot_comparison_for_fold(1)