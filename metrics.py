from sklearn.metrics import roc_curve, auc, f1_score, average_precision_score, accuracy_score, precision_recall_curve, recall_score, precision_score
import matplotlib.pyplot as plt

def calculate_auc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred)

def calculate_aupr(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return average_precision_score(y_true, y_pred), precision, recall

def calculate_acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calculate_recall(y_true, y_pred):
    return recall_score(y_true, y_pred)

def calculate_precision(y_true, y_pred):
    return precision_score(y_true, y_pred)

def calculate_metrics(y_true, y_pred):
    fpr, tpr, roc_auc = calculate_auc(y_true, y_pred)
    aupr, precision_vec, recall_vec = calculate_aupr(y_true, y_pred)
    threshold = 0.5
    pred_binary = [1 if p >= threshold else 0 for p in y_pred]
    f1 = calculate_f1(y_true, pred_binary)
    acc = calculate_acc(y_true, pred_binary)
    recall = calculate_recall(y_true, pred_binary)
    precision = calculate_precision(y_true, pred_binary)
    return {'auc':roc_auc, 'aupr':aupr, 'f1':f1, 'acc':acc, 'recall':recall, 'precision':precision, 'fpr':fpr, 'tpr':tpr, 'precision_vec':precision_vec, 'recall_vec':recall_vec}