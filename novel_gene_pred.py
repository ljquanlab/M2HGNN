from model import *
from processing_data import *
import torch
import re
from collections import defaultdict
import gseapy as gp
from gseapy.plot import *
from sklearn.model_selection import KFold

if __name__ == '__main__':
    GENE_DIM = 68
    DISEASE_DIM = 986
    g_num = 6465
    d_num = 218
    HIDDEN_DIM = 256
    OUTPUT_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.25
    HEADS = 2
    SEED = 2341
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'results/My/best_model_fold_1.pth'
    model = M2HGNN(GENE_DIM, DISEASE_DIM, g_num, d_num, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, HEADS).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    train_dataset = "DisGeNET/"
    (base_g_fea, base_d_l_fea, gene_adj, disesae_adj, train_d_num, train_g_num,
     message_dis_gene_assoc, pos_edge, neg_edge
    ) = load_and_prepare_data(train_dataset, SEED)
    
    k_fold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train_idx, val_idx) in enumerate(k_fold.split(pos_edge)):
        fold_train_pos_edge = pos_edge[train_idx]
        continue
    train_graph = build_graph_for_fold(
            base_g_fea, base_d_l_fea, gene_adj, disesae_adj,
            message_dis_gene_assoc, fold_train_pos_edge,
            pos_edge, train_g_num, train_d_num
        )
    
    
    def generate_edge_for_disease(target_disease_index, gene_num):
        return [[target_disease_index, i] for i in range(gene_num)]
    fold_train_pos_edge = set(map(tuple, fold_train_pos_edge[:, :2].astype(int)))
    filtered_edge = generate_edge_for_disease(100, 6465)
    print(len(filtered_edge))
    filtered_edge = [item for item in filtered_edge if tuple(item) not in fold_train_pos_edge]
    print(len(filtered_edge))
    
    with torch.no_grad():
        loss, pred, _, _, _, = model.forward(train_graph.to(device), filtered_edge, [], GENE_DIM, DISEASE_DIM, device)
        
    pred = pred.detach().cpu().numpy()
    sorted_indices = np.argsort(pred)[::-1]

    top_10_values = pred[sorted_indices[:20]]
    top_10_indices = sorted_indices[:20]
    gene_index = [filtered_edge[index][1] for index in top_10_indices]
    pcg_genes = pd.read_csv('Raw_Data/pcg_gene.csv').iloc[:, 1].tolist()
    
    train_map_filepath = "Dataset/trainnode_index_map.xlsx"
    train_node_dict = pd.read_excel(train_map_filepath)
    train_disease_map = train_node_dict['disease'].dropna().astype(int)
    train_gene_map = train_node_dict['gene'].dropna().astype(int)
    pcg_genes = pd.read_csv('Raw_Data/pcg_gene.csv').iloc[:, 1].tolist()
    disease_name = pd.read_csv('Raw_Data/disease_name.csv').iloc[:, 1].tolist()
    
    target_disease = disease_name[train_disease_map.iloc[100]]
    print("target disease:", target_disease)
    
    gene_list = [pcg_genes[train_gene_map[index]] for index in gene_index]
    print("scores:", top_10_values)
    print(gene_list)
    
    # code for generate GMT file to analyse gene ontology
    annotation_file_path = 'Raw_Data/uniprotkb_Homo_sapiens_AND_reviewed_tru_2025_06_23.xlsx'
    go_col = 'Gene Ontology (biological process)' # can be replaced by: Gene Ontology (biological process) Gene Ontology (cellular component) Gene Ontology (molecular function) Gene Ontology (GO)

    try:
        df = pd.read_excel(annotation_file_path)
    except FileNotFoundError:
        print(f"error:can't find the target file'{annotation_file_path}'.")
        exit()

    go_to_genes = defaultdict(list)
    for index, row in df.iterrows():
        gene_name_str = row.get('Gene Names')
        if pd.isna(gene_name_str):
            continue
        primary_gene_name = gene_name_str.split(' ')[0]

        go_terms_str = row.get(go_col)
        if pd.isna(go_terms_str):
            continue

        go_terms = go_terms_str.split(';')
        for term in go_terms:
            term = term.strip()
            if term:
                match = re.match(r'(.+)\s[\[\(]GO:(\d+)[\]\)]', term)
                if match:
                    go_full_name = f"{match.group(1).strip()} ({match.group(2)})"
                    go_to_genes[go_full_name].append(primary_gene_name)

    print(f"process successfully, find {len(go_to_genes)} single GO biological process.")
    
    res_path = 'results/My/'
    output_gmt_path = res_path + 'go_bp.gmt'

    with open(output_gmt_path, 'w') as f:
        for go_term, genes in go_to_genes.items():
            gene_str = '\t'.join(sorted(list(set(genes))))
            f.write(f"{go_term}\tna\t{gene_str}\n")

    print(f"GMT file has been created in: {output_gmt_path}")

    background_genes = pd.unique(df['Gene Names'].dropna().apply(lambda x: x.split(' ')[0]))
    print(f"Background Gene Set includes {len(background_genes)} genes.")

    res_path = 'results/My/'
    gmt_file = res_path + 'go_bp.gmt'

    gene_ontology_output_file = res_path + 'PE_Enrichment_Analysis_Results'

    print(f"\nstart using gseapy ...")
    enr = gp.enrichr(gene_list=gene_list,
                    gene_sets=gmt_file,
                    background=background_genes,
                    outdir=gene_ontology_output_file,
                    cutoff=0.5 
                    )
    print("\n(Sorted by Adjusted P-value ):")
    results_df = enr.results.sort_values(by='Adjusted P-value', ascending=True)
    print(results_df[['Term', 'Overlap', 'Adjusted P-value', 'Odds Ratio', 'Genes']].head(15))

    fig, (ax_bar, ax_dot) = plt.subplots(
        1, 2,
        figsize=(18, 8),
        sharey=True 
    )

    dot_bar = DotPlot(
        df=enr.res2d,
        y="Term",
        hue="Adjusted P-value",
        thresh=0.05,
        n_terms=10,
        ax=ax_bar,
        figsize=(4, 6)
    )
    ax_dot.grid(False)
    ax_bar = dot_bar.barh(color=["salmon"])
    term_order = dot_bar.data["Term"].tolist()

    dotplot(
        df=enr.res2d,
        column="Adjusted P-value",
        x="Combined Score",
        y="Term",
        y_order=term_order,
        title="Biological Process",
        cutoff=0.05,
        top_term=10,
        size=6,
        ax=ax_dot,
        cmap="viridis_r",
        show_ring=False
    )

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.25)
    ax_dot.grid(False)
    ax_bar.grid(False)
    plt.savefig("figure/combined_enrichment_plot.png", bbox_inches="tight", dpi=300)
    plt.show()