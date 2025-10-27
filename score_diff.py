import argparse
import torch
import random
import torch.optim as optim
from processing_data import *
from model import *
from metrics import *
from utils import *
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200,
                    help='Training Epochs')
parser.add_argument('--num_layers', type=int, default=2,
                    help='graph layers')
parser.add_argument('--heads', type=int, default=2,
                    help='Muti Heads')
parser.add_argument('--gene_input_size', type=int, default=68,
                    help='gene fea num')
parser.add_argument('--disease_input_size', type=int, default=986,
                    help='disease fea num')
parser.add_argument('--hidden_dim', type=int, default=256,
                help='hidden_Channels')
parser.add_argument('--output_dim', type=int, default=64,
                help='Output_Channels')
parser.add_argument('--dropout', type=float, default=0.25,
                help='dropout')
parser.add_argument('--lr', type=float, default= 0.00096)
parser.add_argument('--weight_decay', type=float, default=0.0006,
                    help='l2 reg')
parser.add_argument('--seed', type=int,default=2341,
                    help='random seed')
parser.add_argument('--patience', type=int, default=10,
                help='Early stopping patience')
args = parser.parse_args()
print('args:',args)

EPOCHS = args.epoch
NUM_LAYERS = args.num_layers
HEADS = args.heads
LR = args.lr
GENE_DIM = args.gene_input_size
DISEASE_DIM = args.disease_input_size
HIDDEN_DIM = args.hidden_dim
OUTPUT_DIM = args.output_dim
DROPOUT = args.dropout
WEIGHT_DECAY = args.weight_decay
SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = "results/My/"

train_dataset = "Dataset/DisGeNET/"
# Load edges
disesae_adj_df = pd.read_csv(train_dataset + "disease_mesh_sim.csv", header=None).values
disesae_adj = torch.nonzero(torch.triu(torch.tensor(disesae_adj_df) > 0), as_tuple=False).numpy()
gene_adj = pd.read_csv(train_dataset + "gene_adj.csv").iloc[:, :2].values

# Load features
gene_fea_raw = pd.read_csv(train_dataset + "gene_expression.csv").values
scaler = RobustScaler()
g_fea = scaler.fit_transform(gene_fea_raw)
d_l_fea = pd.read_csv(train_dataset + "disease_biobert_fea.csv", header=None).values

# score > 0
dis_gene_df = pd.read_csv(train_dataset + "DisGeNET.csv")
taus = [i/20 for i in range(20)]

counts = []
for tau in taus:
    n_edges = (dis_gene_df['score'] >= tau).sum()
    counts.append(n_edges)

dis_gene_assoc = dis_gene_df.values
np.random.seed(SEED)
np.random.shuffle(dis_gene_assoc)
split_idx = int(len(dis_gene_assoc) * 0.5)
message_dis_gene_assoc = dis_gene_assoc[0:split_idx]
supervision_dis_gene_assoc = dis_gene_assoc[split_idx:]

# negtive sample
supervision_pos_edges = supervision_dis_gene_assoc[:, :2].astype(int)
message_edges_for_sampling = message_dis_gene_assoc[:, :2].astype(int)
disease_num = max(dis_gene_df['disease'].values.squeeze()) + 1
gene_num = max(dis_gene_df['gene'].values.squeeze()) + 1
neg_g_d = Random_Sampleing(gene_num, disease_num, message_edges_for_sampling, supervision_pos_edges, np.empty((0, 2), dtype=int), 1, SEED).astype(int)

k_fold = KFold(n_splits=5, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(k_fold.split(supervision_dis_gene_assoc)):
        
    test_pos = supervision_dis_gene_assoc[val_idx][:, :2].astype(int)
    test_neg = neg_g_d[val_idx]

    train_pos = supervision_dis_gene_assoc[train_idx]
    train_neg = neg_g_d[train_idx]
    break

train_pos_df = pd.DataFrame(train_pos, columns=["disease", "gene", "score"])
train_neg_df = pd.DataFrame(train_neg, columns=["disease", "gene"])
message_df = pd.DataFrame(message_dis_gene_assoc, columns=["disease", "gene", "score"])

# load feature for node
gene_fea_raw = pd.read_csv(train_dataset + "gene_expression.csv").values
scaler = RobustScaler()
g_fea = scaler.fit_transform(gene_fea_raw)
d_l_fea = pd.read_csv(train_dataset + "disease_biobert_fea.csv", header=None).values
gene_adj = pd.read_csv(train_dataset + "gene_adj.csv").iloc[:, :2].values
disesae_adj_df = pd.read_csv(train_dataset + "disease_mesh_sim.csv", header=None).values
disesae_adj = torch.nonzero(torch.triu(torch.tensor(disesae_adj_df) > 0), as_tuple=False).numpy()

results_list = []
for tau in taus:
    kept_pos = train_pos_df[train_pos_df["score"] > tau]
    kept_msg = message_df[message_df["score"] > tau]
    
    n_removed = len(train_pos_df) - len(kept_pos)
    if n_removed > 0:
        kept_neg = train_neg_df.sample(len(train_neg_df) - n_removed, random_state=SEED)
    else:
        kept_neg = train_neg_df.copy()
        
    train_graph = build_graph_for_fold(g_fea, d_l_fea, gene_adj, disesae_adj,
        kept_msg.values, kept_pos.values, supervision_dis_gene_assoc, gene_num, disease_num)
    
    pos_edge = kept_pos[["disease", "gene"]].values.astype(int)
    neg_edge = kept_neg[["disease", "gene"]].values.astype(int)
    model = M2HGNN(GENE_DIM, DISEASE_DIM, gene_num, disease_num, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, HEADS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    
    best_auc = 0
    best_aupr = 0
    best_f1 = 0
    patience_counter = 0
    for epoch in range(EPOCHS):
        print(f"Tau: {tau}, Epoch: {epoch + 1:03d}", end='\t')
        model.train()
        optimizer.zero_grad()
        loss, train_pred, target, _, _ = model(train_graph.to(device), pos_edge, neg_edge, GENE_DIM, DISEASE_DIM, device)
    
        loss.backward()
        optimizer.step()
        train_metrics = calculate_metrics(target, train_pred.detach().cpu().numpy())
        print("loss_train= {:.4f}".format(loss.detach().cpu().numpy()),
        "train_auc={:.4f}".format(train_metrics['auc'].item()),
        "train_aupr={:.4f}".format(train_metrics['aupr'].item()),
        "train_f1={:.4f}".format(train_metrics['f1'].item()),
        "train_acc={:.4f}".format(train_metrics['acc'].item()),
        "train_recall={:.4f}".format(train_metrics['recall'].item()),
        "train_precision={:.4f}".format(train_metrics['precision'].item()),
        end='\t')

        model.eval()
        with torch.no_grad():
            loss, test_pred, target, embeddings, x_dict = model.forward(train_graph.to(device), test_pos, test_neg, GENE_DIM, DISEASE_DIM, device)
            test_metrics = calculate_metrics(target, test_pred.detach().cpu().numpy())
            print("loss_test= {:.4f}".format(loss.detach().cpu().numpy()),
            "test_auc={:.4f}".format(test_metrics['auc'].item()),
            "test_aupr={:.4f}".format(test_metrics['aupr'].item()),
            "test_f1={:.4f}".format(test_metrics['f1'].item()),
            "test_acc={:.4f}".format(test_metrics['acc'].item()), 
            "test_recall={:.4f}".format(test_metrics['recall'].item()),
            "test_precision={:.4f}".format(test_metrics['precision'].item()))
            
            current_auc = test_metrics['auc'].item()
            if current_auc > best_auc:
                best_auc = current_auc
                best_aupr = test_metrics['aupr'].item()
                best_f1 = test_metrics['f1'].item()
                patience_counter = 0
                print("best_auc:{:.4f}".format(best_auc))
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    results_list.append({
        "tau": tau,
        "best_auc": best_auc,
        "best_aupr": best_aupr,
        "best_f1": best_f1,
        "removed_pos": n_removed,
        "msg_edges": len(kept_msg),
        "train_pos": len(kept_pos),
        "train_neg": len(kept_neg)
    })

results_df = pd.DataFrame(results_list)
results_df.to_csv("results/My/tau_experiment_results.csv", index=False)
results_df = pd.read_csv("results/My/tau_experiment_results.csv")


def plot_performance_vs_threshold(df, save_path):
    print("Generating Plot 1: Performance vs. Threshold...")
    
    best_tau_row = df.loc[df['best_auc'].idxmax()]
    best_tau = best_tau_row['tau']
    best_auc = best_tau_row['best_auc']

    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    sns.set_style("whitegrid")
    
    sns.lineplot(data=df, x='tau', y='best_auc', ax=ax1, label='Test AUC', marker='o', color='C0')
    sns.lineplot(data=df, x='tau', y='best_aupr', ax=ax1, label='Test AUPR', marker='s', color='C1')
    ax1.set_xlabel('Score Threshold (τ)', fontsize=14, weight='bold')
    ax1.set_ylabel('Performance (AUC / AUPR)', fontsize=14, weight='bold')
    ax1.tick_params(axis='y', labelcolor='C0')
    
    ax1.axvline(x=best_tau, color='red', linestyle='--', 
                label=f'Best AUC (τ={best_tau:.2f})')
    
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='tau', y='train_pos', ax=ax2, label='Positive Training Edges', 
                 linestyle='--', color='C2', marker='x', legend=False)
    ax2.set_ylabel('Number of Positive Training Edges', fontsize=14, weight='bold')
    ax2.tick_params(axis='y', labelcolor='C2')
    ax2.set_yscale('log')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=12)
    
    axins = inset_axes(ax1, width="30%", height="30%", loc='right',
                       bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax1.transAxes, borderpad=0.6)
    sns.lineplot(data=df, x='tau', y='best_auc', ax=axins, color='C0', marker='o', legend=False)
    sns.lineplot(data=df, x='tau', y='best_aupr', ax=axins, color='C1', marker='s', legend=False)
    axins.set_xlim(0, 0.2)
    axins.set_ylim(0.875, 0.90)
    axins.tick_params(axis='both', labelsize=8)
    axins.axvline(x=best_tau, color='red', linestyle='--', lw=1)
    mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.6", lw=0.8)
    axins.set_title("Zoomed View (τ ≤ 0.2)", fontsize=9)
    axins.grid(False)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_score_distribution(original_data_path, save_path):
    print("Generating Plot 2: Score Distribution...")
    
    try:
        original_df = pd.read_csv(original_data_path)
    except FileNotFoundError:
        print(f"Warning: Could not find original data file at {original_data_path}. Skipping Plot 2.")
        return

    scores = original_df['score']
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    sns.set_style("ticks")

    sns.histplot(scores, bins=50, kde=True, ax=ax1, label='Score Distribution', 
                 color='C0', stat='density')
    ax1.set_xlabel('DisGeNET Score', fontsize=14, weight='bold')
    ax1.set_ylabel('Density', fontsize=14, weight='bold')
    ax1.legend(loc='upper left', fontsize=12)

    ax2 = ax1.twinx()

    sorted_scores = sorted(scores.unique())
    total_count = len(scores)
    percent_kept = [(scores >= tau).sum() / total_count * 100 for tau in sorted_scores]
    
    ax2.plot(sorted_scores, percent_kept, color='C1', linestyle='--', 
             label='Cumulative % of Associations Kept (≥ Score)')
    ax2.set_ylabel('Percentage of Associations Kept (%)', fontsize=14, weight='bold')
    ax2.tick_params(axis='y', labelcolor='C1')
    ax2.legend(loc='upper right', fontsize=12)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_performance_vs_quantity(df, save_path):
    print("Generating Plot 3: Performance vs. Quantity...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_style("whitegrid")

    best_point = df.loc[df['best_auc'].idxmax()]

    scatter = sns.scatterplot(data=df, x='train_pos', y='best_auc', hue='tau', 
                              palette='viridis_r', s=150, ax=ax, legend='full')

    ax.scatter(best_point['train_pos'], best_point['best_auc'], s=400, 
               facecolors='none', edgecolors='red', linewidth=2, 
               label=f'Best AUC (τ={best_point["tau"]:.2f})')
    
    ax.invert_xaxis()
    
    ax.set_xlabel('Number of Positive Training Edges\n(<-- More Data | Higher Quality -->)', fontsize=14, weight='bold')
    ax.set_ylabel('Best Test AUC', fontsize=14, weight='bold')
    
    handles, labels = ax.get_legend_handles_labels()
    best_label_idx = [i for i, s in enumerate(labels) if 'Best AUC' in s]
    if best_label_idx:
        idx = best_label_idx[0]
        handles.append(handles.pop(idx))
        labels.append(labels.pop(idx))
        
    ax.legend(handles=handles, labels=labels, title='τ Value / Best', 
              bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=13)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

plot_performance_vs_threshold(results_df, os.path.join('figure', "plot_1_performance_vs_threshold.png"))

plot_score_distribution("Dataset/DisGeNET/DisGeNET.csv", os.path.join('figure', "plot_2_score_distribution.png"))

plot_performance_vs_quantity(results_df, os.path.join('figure', "plot_3_performance_vs_quantity.png"))

print("\nAll plots generated successfully and saved to:")