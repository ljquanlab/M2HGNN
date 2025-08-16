import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from processing_data import Structual_SampleNeg, Random_Sampleing
from torch_geometric.nn import HeteroConv, GATv2Conv, GCNConv

class FGNNHG(nn.Module):
    def __init__(self, gene_input_size, disease_input_size, g_num, d_num, hidden_size, output_size, num_layers, dropout, heads):
        super().__init__()

        self.g_gate = MultimodalGating(gene_input_size, d_num, hidden_size, heads, dropout)
        self.d_gate = MultimodalGating(disease_input_size, g_num, hidden_size, heads, dropout)
        
        self.g_lin = nn.Linear(hidden_size, output_size)
        self.d_lin = nn.Linear(hidden_size, output_size)

        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ('gene', 'to', 'gene'): GCNConv(-1, output_size),
                    ('disease', 'to', 'disease'):GCNConv(-1, output_size),
                    ('disease', 'to', 'gene'):GATv2Conv((-1, -1), output_size, add_self_loops=False),
                    ('gene', 'rev_to', 'disease'):GATv2Conv((-1, -1), output_size, add_self_loops=False),
                }
            )
            self.bns.append(nn.BatchNorm1d(output_size))
            self.convs.append(conv)
            self.se_blocks.append(SEBlock(c=output_size, r=16))

        self.dropout = nn.Dropout(dropout)
        self.loss = nn.BCELoss()
        self.mlp = MLP(output_size*2, 1, dropout)

    def forward(self, graph, pos_edge, neg_edge, G_feanum, D_feanum, device, val_neg_edge, part_g_num, part_d_num, g_g, d_d, fold, seed, random = 0):
        # if val_neg_edge is not None:
        #     if random == 0:
        #         neg_edge = Random_Sampleing(part_g_num, part_d_num, pos_edge, val_neg_edge, flag=0)
        #         train_neg_edge = pd.DataFrame(neg_edge)
        #         train_neg_edge.to_csv('Dataset/Data/fold_' +  str(fold+1) + 'train_neg_edge.csv', index=False, header=['disease', 'gene'])
        #     else:
        #         neg_edge = Structual_SampleNeg(part_g_num, part_d_num, g_g, d_d, pos_edge, neg_edge, val_neg_edge, seed)#pos neg数量相同

        x_dict = graph.x_dict
        x_dict['gene'] = self.g_gate(x_dict['gene'][:, 0:G_feanum], x_dict['gene'][:, G_feanum:])
        x_dict['disease'] = self.d_gate(x_dict['disease'][:, 0:D_feanum], x_dict['disease'][:, D_feanum:])
        res_x_dict = x_dict
        res_x_dict['gene'] = self.g_lin(res_x_dict['gene'])
        res_x_dict['disease'] = self.d_lin(res_x_dict['disease'])

        edge_index = graph.edge_index_dict
        for i,conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index)
            for node in x_dict:
                x_dict[node] = self.bns[i](x_dict[node])
                x_dict[node] = F.relu(x_dict[node])
                x_dict[node] = self.se_blocks[i](x_dict[node])
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        for key in res_x_dict.keys():
            x_dict[key] = res_x_dict[key] + x_dict[key]

        target = []
        posnum = len(pos_edge)
        negnum = len(neg_edge)
        conbs = torch.zeros(posnum + negnum, x_dict['disease'][0].size(0) + x_dict['gene'][0].size(0)).to(device)

        for i, pair in enumerate(pos_edge):
            disease_emb = x_dict['disease'][pair[0]]
            gene_emb = x_dict['gene'][pair[1]]
            conb = torch.cat((disease_emb, gene_emb), dim=0)
            conbs[i] = conb
            target.append(1)
        for i, pair in enumerate(neg_edge):
            disease_emb = x_dict['disease'][pair[0]]
            gene_emb = x_dict['gene'][pair[1]]
            conb = torch.cat((disease_emb, gene_emb), dim=0)
            conbs[i+posnum] = conb
            target.append(0)
        
        embeddings = conbs.clone().detach().cpu()
        conbs = self.mlp(conbs)
        conbs = torch.squeeze(conbs)
        pe_loss = self.loss(conbs, torch.tensor(target, dtype=torch.float32).to(device))

        return  pe_loss, conbs, target, embeddings, x_dict
    
class MLP(nn.Module):
    def __init__(self, input_size,output_size,dropout):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(input_size // 2, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.linear(x)
        del x
        return out

class MultimodalGating(nn.Module):
    def __init__(self, dim1, dim2, hidden_dim, num_heads, dropout):
        super().__init__()

        self.modal_f1 = nn.Sequential(
            nn.Linear(dim1, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.modal_f2 = nn.Sequential(
            nn.Linear(dim2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.dropout = nn.Dropout(p=dropout)

        self.gate = nn.Linear(hidden_dim * 3, 1)

    def forward(self, ref_fea, rel_fea):
        ref_encode = F.relu(self.modal_f1(ref_fea))
        rel_encode = F.relu(self.modal_f2(rel_fea))

        ref_encode = self.dropout(ref_encode)
        rel_encode = self.dropout(rel_encode)

        attn_output, _ = self.cross_attn(
            query=ref_encode.unsqueeze(1),
            key=rel_encode.unsqueeze(1),
            value=rel_encode.unsqueeze(1)
        )
        attn_output = attn_output.squeeze(1)  # [B, H]

        ref_cat_rel = torch.concat((ref_encode, rel_encode, attn_output), dim=1)
        gate_rate = torch.sigmoid(self.gate(ref_cat_rel))

        return gate_rate * ref_encode + (1 - gate_rate) * rel_encode
    
class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.squeeze(x.unsqueeze(0).transpose(1, 2))
        y = y.transpose(1, 2)
        y = self.excitation(y).squeeze(0) 
        return x * y