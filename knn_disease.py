import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict, OrderedDict
from sklearn.metrics.pairwise import cosine_similarity

def get_category(TreeNum):
    if pd.isna(TreeNum):
        return 'Uncategorized'
    
    top_layer_categorty = [cls.strip().split('.')[0] for cls in TreeNum.split(',')] #get top layer of mesh tree of the disease
    top_layer_categorty = list(OrderedDict.fromkeys(top_layer_categorty))

    if len(top_layer_categorty) == 1:
        return top_layer_categorty[0]
    elif len(top_layer_categorty) > 1:
        # return 'Multi-category'
        return '&'.join(top_layer_categorty)
    else:
        return 'Uncategorized'
    
uncategoried_list = [14, 85, 117, 133, 134, 207]
train_map_filepath = "Dataset/trainnode_index_map.xlsx"
train_node_dict = pd.read_excel(train_map_filepath)
train_disease_map = train_node_dict['disease'].dropna().astype(int)

# find the target disease
for target_disease_index in  uncategoried_list:
    disease_name = pd.read_csv('Raw_data/disease_name.csv').iloc[:, 1].tolist()
    target_disease = disease_name[train_disease_map.iloc[target_disease_index]]
    print("target disease:", target_disease)
    
embeddings = np.load('results/My/train_disease_embeddings.npy')
labels = pd.read_csv('Raw_data/MeshId_TreeNum.csv', sep='\t').iloc[train_disease_map.tolist()]['TreeNum'].apply(get_category).tolist()

all_embeddings = embeddings
all_labels = labels

is_uncategorized = np.array([1 if l == 'Uncategorized' else 0 for l in all_labels])

labeled_idx = np.where(is_uncategorized == 0)[0]
unlabeled_idx = np.where(is_uncategorized == 1)[0]

X_train = all_embeddings[labeled_idx]
y_train = [all_labels[i] for i in labeled_idx]
X_test = all_embeddings[unlabeled_idx]

knn = KNeighborsClassifier(n_neighbors=4, metric='cosine')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Predicted categories for uncategorized diseases:")
for idx, pred in zip(uncategoried_list, y_pred):
    print(f"Disease index {idx}: {pred}")