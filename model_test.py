from model import *
from metrics import *
from processing_data import *
import argparse
import torch
import random

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
    parser.add_argument('--disease_input_size', type=int, default=986,
                        help='disease fea num')
    parser.add_argument('--gene_num', type=int, default=6465,
                        help='train gene num')
    parser.add_argument('--disease_num', type=int, default=218,
                        help='train disease num')
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
    GENE_NUM = args.gene_num
    DISEASE_NUM = args.disease_num
    HIDDEN_DIM = args.hidden_dim
    OUTPUT_DIM = args.output_dim
    DROPOUT = args.dropout
    WEIGHT_DECAY = args.weight_decay
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    save_path = "results/My/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_datasets = ["DISEASES/d cold start/", "DISEASES/g cold start/", "DISEASES/d-g cold start/"] 
    for cold_data_path in test_datasets:
        print("*"*10 + "test for " + cold_data_path + "*"*10)
        (base_g_fea, base_d_l_fea, gene_adj, disesae_adj, test_d_num, test_g_num,
        message_dis_gene_assoc, pos_edge, neg_edge
        ) = load_and_prepare_data(cold_data_path, SEED)
        
        assoc_dg = pd.read_csv("Dataset/" + cold_data_path + "assoc_dg.csv")
        assoc_gd = pd.read_csv("Dataset/" + cold_data_path + "assoc_gd.csv")
        test_graph = build_graph_for_other_dataset(base_g_fea, base_d_l_fea, gene_adj, disesae_adj,
                                    message_dis_gene_assoc, assoc_dg, assoc_gd, 218, 6465, test_d_num, test_g_num)
        
        for fold_idx in range(1, 6):
            print(f"\n--- Testing with model from Fold {fold_idx} ---")
            model_path = f"{save_path}best_model_fold_{fold_idx}.pth"
            model = M2HGNN(GENE_DIM, DISEASE_DIM, GENE_NUM, DISEASE_NUM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, HEADS).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            with torch.no_grad():
                print("Test for " + cold_data_path)
                
                loss, pred, target, _, _ = model.forward(test_graph.to(device), pos_edge.astype(int), neg_edge.astype(int), GENE_DIM, DISEASE_DIM, device)
                test_metrics = calculate_metrics(target, pred.detach().cpu().numpy())
                print("test_auc={:.4f}".format(test_metrics['auc'].item()),
                "test_aupr={:.4f}".format(test_metrics['aupr'].item()),
                "test_f1={:.4f}".format(test_metrics['f1'].item()),
                "test_acc={:.4f}".format(test_metrics['acc'].item()), 
                "test_recall={:.4f}".format(test_metrics['recall'].item()),
                "test_precision={:.4f}".format(test_metrics['precision'].item()))