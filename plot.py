import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ptitprince as pt
from matplotlib.patches import Patch

models = ['full model', 'w/o MF', 'w/o RES', 'w/o HGNN', 'w/o MF & HGNN', 'w/o RES & HGNN', 'w/o MF & RES']
metrics = ['auc', 'aupr', 'f1']

data1 = {
    'full model':
        {
            'auc':[0.8974, 0.0042],
            'aupr':[0.8979, 0.0047],
            'f1':[0.8121, 0.0044]
        },
    'w/o MF':
        {
            'auc':[0.8640, 0.0146],
            'aupr':[0.8686, 0.0121],
            'f1':[0.6790, 0.0703]
        },
    'w/o RES':
        {
            'auc':[0.8388, 0.0298],
            'aupr':[0.8290, 0.0272],
            'f1':[0.7108, 0.0787]
        },
    'w/o HGNN':
        {
            'auc':[0.8868, 0.0051],
            'aupr':[0.8870, 0.0050],
            'f1':[0.8010, 0.0071]
        },
    'w/o MF & HGNN':
        {
            'auc':[0.8469, 0.0053],
            'aupr':[0.8572, 0.0065],
            'f1':[0.7570, 0.0057]
        },
    'w/o RES & HGNN':
        {
            'auc':[0.8291, 0.0163],
            'aupr':[0.8346, 0.0126],
            'f1':[0.7541, 0.0206]
        },
    'w/o MF & RES':
        {
            'auc':[0.8077, 0.0128],
            'aupr':[0.8018, 0.0141],
            'f1':[0.6515, 0.1318]
        },
}

df_list = []
for model_name, metric_data in data1.items():
    for metric_name, values in metric_data.items():
        df_list.append({
            'model': model_name,
            'metric': metric_name,
            'value': values[0],
            'error': values[1]
        })
df = pd.DataFrame(df_list)
fig, ax = plt.subplots(figsize=(6, 6)) 
bar_width = 0.2
group_gap = 0.8 
index = np.arange(len(models)) * group_gap

colors = ['#71b7ed', '#fae69e', '#f2b56f']
metric_colors = {
    'auc': '#71b7ed',
    'aupr': '#fae69e',
    'f1': '#f2b56f'
}
for i, metric in enumerate(metrics):
    subset = df[df['metric'] == metric]
    subset = subset.set_index('model').reindex(models).reset_index()
    offset = (i - (len(metrics) - 1) / 2.0) * bar_width
    x_pos = index + offset

    ax.bar(x_pos, subset['value'], bar_width, 
           yerr=subset['error'], capsize=4,
           label=metric.upper(), color=metric_colors[metric])
    
ax.set_ylabel('Score', fontsize=14)
ax.set_xticks(index)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
ax.set_ylim(0.51, 1.0) 

ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(title='Metrics', fontsize=12)

plt.tight_layout()
plt.savefig("figure/ablation of model components", dpi=300)
plt.show()

models = ['full model', 'w/o GE', 'w/o DL', 'w/o GE & DL']
metrics = ['auc', 'aupr', 'f1']

data1 = {
    'full model':
        {
            'auc':[0.8974, 0.0042],
            'aupr':[0.8979, 0.0047],
            'f1':[0.8121, 0.0044]
        },
    'w/o GE':
        {
            'auc':[0.8770, 0.0079],
            'aupr':[0.8813, 0.0067],
            'f1':[0.7861, 0.0115]
        },
    'w/o DL':
        {
            'auc':[0.8732, 0.0198],
            'aupr':[0.8785, 0.0176],
            'f1':[0.7673, 0.0251]
        },
    'w/o GE & DL':
        {
            'auc':[0.8636, 0.0138],
            'aupr':[0.8611, 0.0104],
            'f1':[0.7598, 0.0145]
        },
}

df_list = []
for model_name, metric_data in data1.items():
    for metric_name, values in metric_data.items():
        df_list.append({
            'model': model_name,
            'metric': metric_name,
            'value': values[0],
            'error': values[1]
        })
df = pd.DataFrame(df_list)

plt.figure(figsize=(6, 6))
sns.set(style="whitegrid", font_scale=1.2)

markers = ['p', 'D', '*']
for i in range(len(metrics)):
    metric = metrics[i]
    subset = df[df['metric'] == metric]
    plt.errorbar(
        subset['model'], 
        subset['value'], 
        label=metric.upper(),
        marker=markers[i],
        markersize = 10,
        capsize=6, 
        linestyle='-'
    )

plt.grid(False)
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.ylim(0.65, 0.95)
plt.legend()
plt.tight_layout()
plt.savefig("figure/ablation of node features", dpi=300)
plt.show()


def prepare_data(df, dataset_name):
    data_agg = (
        df[df['Dataset'] == dataset_name]
        .pivot(index='Model', columns='Metric', values='Value')
        .reindex(columns=['auc', 'aupr', 'f1'])
    )
    data_agg = data_agg.sort_values(by='auc', ascending=False)
    return data_agg

def plot_radar_chart(df_agg, title, dataset):
    labels = df_agg.columns.values
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=True))

    y_min, y_max = 0.6, 0.9
    ax.set_ylim(y_min, y_max)

    for index, row in df_agg.iterrows():
        model_name = index
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=model_name, linewidth=2, marker='o', markersize=7)
        ax.fill(angles, values, alpha=0.2)

    zorder_outer_ring = 10
    ax.plot(np.linspace(0, 2 * np.pi, 200), [y_max] * 200, color="black", lw=2, zorder=zorder_outer_ring)

    ax.set_yticklabels([])
    ax.set_xticklabels([])

    yticks = np.arange(y_min, y_max, 0.05)
    for tick in yticks:
        ax.text(np.pi/2, tick, f'{tick:.2f}', ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0))

    ax.text(angles[0], y_max * 1.08, labels[0], ha='center', va='center', size=14)
    ax.text(angles[1], y_max * 1.08, labels[1], ha='center', va='center', size=14)
    ax.text(angles[2], y_max * 1.08, labels[2], ha='center', va='center', size=14)

    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15), fontsize=10)
    ax.spines['polar'].set_visible(False)
    fig.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    plt.savefig(f"figure/result for {dataset} set.png", dpi=300)
    plt.show()

data = [
    #My full model
    ('M2HGNN', 'dataset1', 'auc', 0.8950), ('M2HGNN', 'dataset1', 'aupr', 0.8965), ('M2HGNN', 'dataset1', 'f1', 0.8167),
    ('M2HGNN', 'dataset1', 'auc', 0.8969), ('M2HGNN', 'dataset1', 'aupr', 0.8989), ('M2HGNN', 'dataset1', 'f1', 0.8109),
    ('M2HGNN', 'dataset1', 'auc', 0.9028), ('M2HGNN', 'dataset1', 'aupr', 0.9035), ('M2HGNN', 'dataset1', 'f1', 0.8135),
    ('M2HGNN', 'dataset1', 'auc', 0.9011), ('M2HGNN', 'dataset1', 'aupr', 0.9009), ('M2HGNN', 'dataset1', 'f1', 0.8152),
    ('M2HGNN', 'dataset1', 'auc', 0.8912), ('M2HGNN', 'dataset1', 'aupr', 0.8897), ('M2HGNN', 'dataset1', 'f1', 0.8042),
    
    ('M2HGNN', 'dataset2', 'auc', 0.8599), ('M2HGNN', 'dataset2', 'aupr', 0.8607), ('M2HGNN', 'dataset2', 'f1', 0.6774),
    
    ('M2HGNN', 'dataset3', 'auc', 0.8322), ('M2HGNN', 'dataset3', 'aupr', 0.8362), ('M2HGNN', 'dataset3', 'f1', 0.6877),
    
    ('M2HGNN', 'dataset4', 'auc', 0.8537), ('M2HGNN', 'dataset4', 'aupr', 0.8539), ('M2HGNN', 'dataset4', 'f1', 0.6671),
    
    # HNEEM
    ('HNEEM', 'dataset1', 'auc', 0.8547), ('HNEEM', 'dataset1', 'aupr', 0.8681), ('HNEEM', 'dataset1', 'f1', 0.7677),
    ('HNEEM', 'dataset1', 'auc', 0.8569), ('HNEEM', 'dataset1', 'aupr', 0.8626), ('HNEEM', 'dataset1', 'f1', 0.7682),
    ('HNEEM', 'dataset1', 'auc', 0.8534), ('HNEEM', 'dataset1', 'aupr', 0.8669), ('HNEEM', 'dataset1', 'f1', 0.7625),
    ('HNEEM', 'dataset1', 'auc', 0.8575), ('HNEEM', 'dataset1', 'aupr', 0.8657), ('HNEEM', 'dataset1', 'f1', 0.7678),
    ('HNEEM', 'dataset1', 'auc', 0.8540), ('HNEEM', 'dataset1', 'aupr', 0.8587), ('HNEEM', 'dataset1', 'f1', 0.7608),
    
    ('HNEEM', 'dataset2', 'auc', 0.8334), ('HNEEM', 'dataset2', 'aupr', 0.8354), ('HNEEM', 'dataset2', 'f1', 0.7414),
    
    ('HNEEM', 'dataset3', 'auc', 0.8127), ('HNEEM', 'dataset3', 'aupr', 0.8294), ('HNEEM', 'dataset3', 'f1', 0.7263),
    
    ('HNEEM', 'dataset4', 'auc', 0.8261), ('HNEEM', 'dataset4', 'aupr', 0.8228), ('HNEEM', 'dataset4', 'f1', 0.7557),
    
    # GlaHGCL
    ('GlaHGCL', 'dataset1', 'auc', 0.7951), ('GlaHGCL', 'dataset1', 'aupr', 0.8054), ('GlaHGCL', 'dataset1', 'f1', 0.7242),
    ('GlaHGCL', 'dataset1', 'auc', 0.8105), ('GlaHGCL', 'dataset1', 'aupr', 0.7934), ('GlaHGCL', 'dataset1', 'f1', 0.7414),
    ('GlaHGCL', 'dataset1', 'auc', 0.7732), ('GlaHGCL', 'dataset1', 'aupr', 0.7526), ('GlaHGCL', 'dataset1', 'f1', 0.7259),
    ('GlaHGCL', 'dataset1', 'auc', 0.7932), ('GlaHGCL', 'dataset1', 'aupr', 0.7862), ('GlaHGCL', 'dataset1', 'f1', 0.7307),
    ('GlaHGCL', 'dataset1', 'auc', 0.6420), ('GlaHGCL', 'dataset1', 'aupr', 0.6586), ('GlaHGCL', 'dataset1', 'f1', 0.6335),
    
    ('GlaHGCL', 'dataset2', 'auc', 0.7854), ('GlaHGCL', 'dataset2', 'aupr', 0.7896), ('GlaHGCL', 'dataset2', 'f1', 0.7237),
    
    ('GlaHGCL', 'dataset3', 'auc', 0.7458), ('GlaHGCL', 'dataset3', 'aupr', 0.7467), ('GlaHGCL', 'dataset3', 'f1', 0.6980),

    ('GlaHGCL', 'dataset4', 'auc', 0.7510), ('GlaHGCL', 'dataset4', 'aupr', 0.7487), ('GlaHGCL', 'dataset4', 'f1', 0.6032),

    # dgn2vec
    ('dgn2vec', 'dataset1', 'auc', 0.7098), ('dgn2vec', 'dataset1', 'aupr', 0.7155), ('dgn2vec', 'dataset1', 'f1', 0.6910),
    ('dgn2vec', 'dataset1', 'auc', 0.7151), ('dgn2vec', 'dataset1', 'aupr', 0.7201), ('dgn2vec', 'dataset1', 'f1', 0.6923),
    ('dgn2vec', 'dataset1', 'auc', 0.7166), ('dgn2vec', 'dataset1', 'aupr', 0.7128), ('dgn2vec', 'dataset1', 'f1', 0.6854),
    ('dgn2vec', 'dataset1', 'auc', 0.7173), ('dgn2vec', 'dataset1', 'aupr', 0.7164), ('dgn2vec', 'dataset1', 'f1', 0.6914),
    ('dgn2vec', 'dataset1', 'auc', 0.7032), ('dgn2vec', 'dataset1', 'aupr', 0.7037), ('dgn2vec', 'dataset1', 'f1', 0.6884),
    
    ('dgn2vec', 'dataset2', 'auc', 0.6972), ('dgn2vec', 'dataset2', 'aupr', 0.6957), ('dgn2vec', 'dataset2', 'f1', 0.7320),
    
    ('dgn2vec', 'dataset3', 'auc', 0.6645), ('dgn2vec', 'dataset3', 'aupr', 0.6839), ('dgn2vec', 'dataset3', 'f1', 0.7004),
    
    ('dgn2vec', 'dataset4', 'auc', 0.7044), ('dgn2vec', 'dataset4', 'aupr', 0.7137), ('dgn2vec', 'dataset4', 'f1', 0.7280),

    # GCN
    ('GCN', 'dataset1', 'auc', 0.8002), ('GCN', 'dataset1', 'aupr', 0.8119), ('GCN', 'dataset1', 'f1', 0.7082),
    ('GCN', 'dataset1', 'auc', 0.7589), ('GCN', 'dataset1', 'aupr', 0.7681), ('GCN', 'dataset1', 'f1', 0.6805),
    ('GCN', 'dataset1', 'auc', 0.7659), ('GCN', 'dataset1', 'aupr', 0.7637), ('GCN', 'dataset1', 'f1', 0.6807),
    ('GCN', 'dataset1', 'auc', 0.8257), ('GCN', 'dataset1', 'aupr', 0.8364), ('GCN', 'dataset1', 'f1', 0.7297),
    ('GCN', 'dataset1', 'auc', 0.7608), ('GCN', 'dataset1', 'aupr', 0.7774), ('GCN', 'dataset1', 'f1', 0.6964),
    
    ('GCN', 'dataset2', 'auc', 0.8269), ('GCN', 'dataset2', 'aupr', 0.8305), ('GCN', 'dataset2', 'f1', 0.6871),

    ('GCN', 'dataset3', 'auc', 0.8292), ('GCN', 'dataset3', 'aupr', 0.8322), ('GCN', 'dataset3', 'f1', 0.6307),
    
    ('GCN', 'dataset4', 'auc', 0.8444), ('GCN', 'dataset4', 'aupr', 0.8382), ('GCN', 'dataset4', 'f1', 0.7648),

    # DeepWalk
    ('DeepWalk', 'dataset1', 'auc', 0.8263), ('DeepWalk', 'dataset1', 'aupr', 0.8293), ('DeepWalk', 'dataset1', 'f1', 0.4384),
    ('DeepWalk', 'dataset1', 'auc', 0.8337), ('DeepWalk', 'dataset1', 'aupr', 0.8219), ('DeepWalk', 'dataset1', 'f1', 0.4299),
    ('DeepWalk', 'dataset1', 'auc', 0.8278), ('DeepWalk', 'dataset1', 'aupr', 0.8255), ('DeepWalk', 'dataset1', 'f1', 0.4263),
    ('DeepWalk', 'dataset1', 'auc', 0.8323), ('DeepWalk', 'dataset1', 'aupr', 0.8317), ('DeepWalk', 'dataset1', 'f1', 0.4479),
    ('DeepWalk', 'dataset1', 'auc', 0.8264), ('DeepWalk', 'dataset1', 'aupr', 0.8211), ('DeepWalk', 'dataset1', 'f1', 0.4340),
    
    ('DeepWalk', 'dataset2', 'auc', 0.8083), ('DeepWalk', 'dataset2', 'aupr', 0.7701), ('DeepWalk', 'dataset2', 'f1', 0.3471),
    
    ('DeepWalk', 'dataset3', 'auc', 0.7580), ('DeepWalk', 'dataset3', 'aupr', 0.7322), ('DeepWalk', 'dataset3', 'f1', 0.4089),
    
    ('DeepWalk', 'dataset4', 'auc', 0.7432), ('DeepWalk', 'dataset4', 'aupr', 0.6950), ('DeepWalk', 'dataset4', 'f1', 0.2568),

    # HerGePred
    ('HerGePred', 'dataset1', 'auc', 0.7572), ('HerGePred', 'dataset1', 'aupr', 0.7638), ('HerGePred', 'dataset1', 'f1', 0.7114),
    ('HerGePred', 'dataset1', 'auc', 0.7548), ('HerGePred', 'dataset1', 'aupr', 0.7658), ('HerGePred', 'dataset1', 'f1', 0.7086),
    ('HerGePred', 'dataset1', 'auc', 0.7477), ('HerGePred', 'dataset1', 'aupr', 0.7587), ('HerGePred', 'dataset1', 'f1', 0.7027),
    ('HerGePred', 'dataset1', 'auc', 0.7636), ('HerGePred', 'dataset1', 'aupr', 0.7729), ('HerGePred', 'dataset1', 'f1', 0.7129),
    ('HerGePred', 'dataset1', 'auc', 0.7427), ('HerGePred', 'dataset1', 'aupr', 0.7517), ('HerGePred', 'dataset1', 'f1', 0.7014),
    
    ('HerGePred', 'dataset2', 'auc', 0.7832), ('HerGePred', 'dataset2', 'aupr', 0.7382), ('HerGePred', 'dataset2', 'f1', 0.7518),
    
    ('HerGePred', 'dataset3', 'auc', 0.8301), ('HerGePred', 'dataset3', 'aupr', 0.8235), ('HerGePred', 'dataset3', 'f1', 0.7748),
    
    ('HerGePred', 'dataset4', 'auc', 0.8201), ('HerGePred', 'dataset4', 'aupr', 0.7948), ('HerGePred', 'dataset4', 'f1', 0.7670),
    
    # DGP-PGTN
    ('DGP-PGTN', 'dataset1', 'auc', 0.7152), ('DGP-PGTN', 'dataset1', 'aupr', 0.7334), ('DGP-PGTN', 'dataset1', 'f1', 0.5852),
    ('DGP-PGTN', 'dataset1', 'auc', 0.7197), ('DGP-PGTN', 'dataset1', 'aupr', 0.7385), ('DGP-PGTN', 'dataset1', 'f1', 0.5717),
    ('DGP-PGTN', 'dataset1', 'auc', 0.7283), ('DGP-PGTN', 'dataset1', 'aupr', 0.7381), ('DGP-PGTN', 'dataset1', 'f1', 0.6211),
    ('DGP-PGTN', 'dataset1', 'auc', 0.7410), ('DGP-PGTN', 'dataset1', 'aupr', 0.7564), ('DGP-PGTN', 'dataset1', 'f1', 0.6842),
    ('DGP-PGTN', 'dataset1', 'auc', 0.7213), ('DGP-PGTN', 'dataset1', 'aupr', 0.7341), ('DGP-PGTN', 'dataset1', 'f1', 0.5182),
    
    ('DGP-PGTN', 'dataset2', 'auc', 0.6015), ('DGP-PGTN', 'dataset2', 'aupr', 0.6360), ('DGP-PGTN', 'dataset2', 'f1', 0.5611),
    
    ('DGP-PGTN', 'dataset3', 'auc', 0.6960), ('DGP-PGTN', 'dataset3', 'aupr', 0.7210), ('DGP-PGTN', 'dataset3', 'f1', 0.5783),
    
    ('DGP-PGTN', 'dataset4', 'auc', 0.5804), ('DGP-PGTN', 'dataset4', 'aupr', 0.5729), ('DGP-PGTN', 'dataset4', 'f1', 0.4214),
]

df = pd.DataFrame(data, columns=['Model', 'Dataset', 'Metric', 'Value'])
df_dataset1 = df[df['Dataset'] == 'dataset1'].copy()

f, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.1)
sns.set_style("whitegrid")


pt.RainCloud(data=df_dataset1, x='Metric', y='Value', hue='Model',
             palette='Set2',
             width_viol = .8,
             width_box = .2,
             orient='v',
             move = .22
             )

# plt.title('Performance Comparison on Development Set', fontsize=16)
ax.grid(False)

plt.xlabel("")
plt.ylabel('Score', fontsize=12)
plt.tight_layout()
plt.savefig("figure/result for development set", dpi=300, bbox_inches='tight')
plt.show()


data_dataset2 = prepare_data(df, 'dataset2')
plot_radar_chart(data_dataset2, 'Model Performance on Dataset2', "disease-cold")

data_dataset3 = prepare_data(df, 'dataset3')
plot_radar_chart(data_dataset3, 'Model Performance on Dataset3', "gene-cold")

data_dataset3 = prepare_data(df, 'dataset4')

model_order = ['M2HGNN', 'HNEEM', 'GlaHGCL', 'dgn2vec', 'GCN', 'DeepWalk','DGP-PGTN', 'HerGePred']

plt.figure(figsize=(8, 4))
sns.heatmap(
    data_dataset3, 
    annot=True,
    fmt=".4f",
    cmap="Blues",
    linewidths=.5,
    cbar_kws={'label': 'Mean Score'}
)
# plt.title('Mean Performance Heatmap on Dataset4', fontsize=16)
plt.xlabel("")
plt.ylabel('Model', fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0) 
plt.savefig("figure/result for disease & gene cold set", dpi=300)
plt.show()