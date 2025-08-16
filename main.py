import argparse
import torch
import random
import torch.optim as optim
from processing_data import *
from model import *
from metrics import *
from utils import *
from ranger import Ranger
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

if __name__ == '__main__' :

    #Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200,
                        help='Training Epochs')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='graph layers')
    parser.add_argument('--heads', type=int, default=2,
                        help='Muti Heads')
    parser.add_argument('--gene_input_size', type=int, default=68,
                        help='gene fea num')
    parser.add_argument('--disease_input_size', type=int, default=1080,
                        help='disease fea num')
    parser.add_argument('--miRNA_input_size', type=int, default=714,
                        help='miRNA fea num')
    parser.add_argument('--hidden_dim', type=int, default=256,
                    help='hidden_Channels')
    parser.add_argument('--output_dim', type=int, default=64,
                    help='Output_Channels')
    parser.add_argument('--dropout', type=float, default=0.30,
                    help='dropout')
    parser.add_argument('--lr', type=float, default= 0.00064)
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
    MIRNA_DIM = args.miRNA_input_size
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

    train_data, test_data, g_num, d_num = split_data(path='Dataset/Data/', splitRate=0.7, seed=SEED)
    train_graph, train_g_num, train_d_num, g_g, d_d, message_train_edge, train_val_pos_edge, train_neg_edge = graph_constract(train_data, map='Dataset/Data/trainnode_index_map.xlsx', seed=SEED)
    test_graph, test_g_num, test_d_num, _, _, message_test_edge, test_pos_edge, test_neg_edge = graph_constract(test_data, map='Dataset/Data/testnode_index_map.xlsx', seed=SEED, flag = 0)
    test_neg_edge = Random_Sampleing(test_g_num, test_d_num, message_test_edge, test_pos_edge, test_neg_edge, 1, SEED)
    test_edge = pd.DataFrame(test_neg_edge)
    test_edge.to_csv('Dataset/Data/test_neg_edge.csv', index=False, header=['disease', 'gene'])

    # the code above has created the test data, and wll create 5-cv
    train_val_neg_edge = Random_Sampleing(train_g_num, train_d_num, message_train_edge, train_val_pos_edge, train_neg_edge, 1 ,SEED)
    k_fold = KFold(n_splits=5, shuffle=True, random_state=SEED)  

    all_folds_best_val_metrics = []
    all_folds_best_pred = [] # five arrays -> disease gene score
    all_folds_best_dict = []

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(train_val_pos_edge)):
        print(f"\n---------- Fold {fold + 1}/{5} ----------")
        fold_train_pos_edge = train_val_pos_edge[train_idx]
        fold_train_neg_edge = train_val_neg_edge[train_idx]
        fold_val_pos_edge = train_val_pos_edge[val_idx]
        val_neg_edge = train_val_neg_edge[val_idx]
        pd_train_pos_edge = pd.DataFrame(fold_train_pos_edge)
        pd_train_pos_edge.to_csv('Dataset/Data/fold_' +  str(fold+1) + 'train_pos_edge.csv', index=False, header=['disease', 'gene'])
        pd_train_neg_edge = pd.DataFrame(fold_train_neg_edge)
        pd_train_neg_edge.to_csv('Dataset/Data/fold_' +  str(fold+1) + 'train_neg_edge.csv', index=False, header=['disease', 'gene'])
        pd_val_pos_edge = pd.DataFrame(fold_val_pos_edge)
        pd_val_pos_edge.to_csv('Dataset/Data/fold_' +  str(fold+1) + 'val_pos_edge.csv', index=False, header=['disease', 'gene'])
        pd_val_neg_edge = pd.DataFrame(val_neg_edge)
        pd_val_neg_edge.to_csv('Dataset/Data/fold_' +  str(fold+1) + 'val_neg_edge.csv', index=False, header=['disease', 'gene'])

        model = FGNNHG(GENE_DIM, DISEASE_DIM, g_num, d_num, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, HEADS).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_metrics_this_fold = {}
        best_dict = {}
        best_pred = None
        best_auc_fold = 0
        patience_counter = 0

        for epoch in range(EPOCHS):
            print(f"Fold: {fold + 1}, Epoch: {epoch + 1:03d}", end='\t')
            model.train()
            optimizer.zero_grad()
            loss, train_pred, target, _, _ = model(train_graph.to(device), fold_train_pos_edge, fold_train_neg_edge, GENE_DIM, DISEASE_DIM, device, np.vstack((val_neg_edge, fold_val_pos_edge)), train_g_num, train_d_num, g_g, d_d, fold, seed=SEED+epoch, random=0)
            
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
                loss, valid_pred, target, embeddings, x_dict = model.forward(train_graph.to(device), fold_val_pos_edge, val_neg_edge, GENE_DIM, DISEASE_DIM, device, None, None, None, None, None, None, None)
                val_metrics = calculate_metrics(target, valid_pred.detach().cpu().numpy())
                print("loss_test= {:.4f}".format(loss.detach().cpu().numpy()),
                "test_auc={:.4f}".format(val_metrics['auc'].item()),
                "test_aupr={:.4f}".format(val_metrics['aupr'].item()),
                "test_f1={:.4f}".format(val_metrics['f1'].item()),
                "test_acc={:.4f}".format(val_metrics['acc'].item()), 
                "test_recall={:.4f}".format(val_metrics['recall'].item()),
                "test_precision={:.4f}".format(val_metrics['precision'].item()))

                current_auc = val_metrics['auc'].item()
                if current_auc > best_auc_fold:
                    best_pred = train_pred.detach().cpu().numpy()[0:train_pred.shape[0]//2]
                    for node in x_dict:
                        x_dict[node] = x_dict[node].detach().cpu().numpy()
                    best_dict = x_dict
                    best_auc_fold = current_auc
                    patience_counter = 0
                    torch.save(model.state_dict(), f"{save_path}best_model_fold_{fold+1}.pth")
                    print(f"Saved new best model with AUC: {best_auc_fold:.4f}")
                    best_metrics_this_fold = val_metrics
                    best_pos_emb, best_neg_emb = obtain_embbedings(x_dict, fold_val_pos_edge, val_neg_edge)
                else:
                    patience_counter += 1
                
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch+1} for fold {fold+1}")
                    break

        visualization_data = {
            'initial_embeddings': {k: v.cpu().numpy() for k, v in train_graph.x_dict.items()},
            'learned_embeddings': best_dict,
            'val_pos_edges': fold_val_pos_edge,
            'val_neg_edges': val_neg_edge
        }
        
        viz_file_path = os.path.join(save_path, f'visualization_data_fold_{fold+1}.pt')
        torch.save(visualization_data, viz_file_path)

        disease_gene_score = np.hstack((fold_train_pos_edge, best_pred.reshape(-1, 1)))
        all_folds_best_pred.append(disease_gene_score)#合并预测的值与对应的disease-gene对
        all_folds_best_val_metrics.append(best_metrics_this_fold)
        all_folds_best_dict.append(best_dict)
    
    for fold in range(len(all_folds_best_pred)):
        np.save(save_path + 'train_fold_' + str(fold+1) + '_predition', all_folds_best_pred[fold])
    disease_embed = get_embeddings(all_folds_best_dict, 'disease')
    gene_embed = get_embeddings(all_folds_best_dict, 'gene')
    np.save(save_path + 'train_disease_embeddings.npy', disease_embed)
    np.save(save_path + 'train_gene_embeddings.npy', gene_embed)

    print(f"\n\n{'='*20} 5-Fold Cross-Validation Summary {'='*20}")
    if all_folds_best_val_metrics:
        val_metrics_df = pd.DataFrame(all_folds_best_val_metrics).drop(columns=['fpr', 'tpr', 'precision_vec', 'recall_vec'])
        avg_val_metrics = val_metrics_df.mean()
        std_val_metrics = val_metrics_df.std()
        
        print("--- Average Validation Performance ---")
        for metric in avg_val_metrics.index:
            print(f"{metric.upper()}: {avg_val_metrics[metric]:.4f} ± {std_val_metrics[metric]:.4f}")
    else:
        print("No validation metrics were recorded.")

    print(f"\n\n{'='*20} Final Testing on Hold-Out Test Set {'='*20}")
    all_folds_test_metrics = []

    for fold_idx in range(1, 6):
        print(f"\n--- Testing with model from Fold {fold_idx} ---")
        model_path = f"{save_path}best_model_fold_{fold_idx}.pth"

        model = FGNNHG(GENE_DIM, DISEASE_DIM, g_num, d_num, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, HEADS).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            loss, pred, target, _, _ = model.forward(test_graph.to(device), test_pos_edge, test_neg_edge, GENE_DIM, DISEASE_DIM, device, None, None, None, None, None, None, None)
            test_metrics = calculate_metrics(target, pred.detach().cpu().numpy())
            np.save(save_path + 'test_pred.npy', pred.cpu().detach().numpy())
            np.save(save_path + 'test_truth.npy', np.array(target))
            
            print("Test Results -> AUC: {:.4f}, AUPR: {:.4f}, F1: {:.4f}, Accuracy: {:.4f}, Recall: {:.4f}, Precision: {:.4f}".format(
                test_metrics['auc'].item(),
                test_metrics['aupr'].item(),
                test_metrics['f1'].item(),
                test_metrics['acc'].item(),
                test_metrics['recall'].item(),
                test_metrics['precision'].item()
            ))
            
            all_folds_test_metrics.append(test_metrics)


    test_metrics_df = pd.DataFrame(all_folds_test_metrics).drop(columns=['fpr', 'tpr', 'precision_vec', 'recall_vec'])
    avg_test_metrics = test_metrics_df.mean()
    std_test_metrics = test_metrics_df.std()
    
    for metric in avg_test_metrics.index:
        print(f"{metric.upper()}: {avg_test_metrics[metric]:.4f} ± {std_test_metrics[metric]:.4f}")