import numpy as np
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def one_hot_encode(arr, labels=21):
    """one-hot encodes input array."""
    a = arr
    b = np.zeros((len(a), labels))
    b[np.arange(len(a)), a] = 1
    return b

def compute_results(Ytrue, pred):
    """Computes accuracy and loss of prediction."""
    loss = log_loss(Ytrue, pred)
    Ypred = one_hot_encode(np.argmax(pred, axis=1))
    accuracy = accuracy_score(Ytrue, Ypred)
    return loss, accuracy

def plot_results(loss, accuracy, spp, title='Results'):
    """Plots prediction results."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title)
    ax.bar([-1, 0, 1], [loss, accuracy, spp], color=['purple', 'green', 'red'])
    ax.tick_params('x', labelsize=12)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['log loss', 'accuracy', 'seconds per prediction'])
    plt.show();
    return

def plot_confusion(Ytrue, pred, model, title):
    """Computes and plots confusion matrix"""
    preds = np.argmax(pred, axis=1)
    pred_classes = []
    for pr in preds:
        pred_classes.append(model.decode(pr))
    true = np.argmax(Ytrue, axis=1)
    true_classes = []
    for y in true:
        true_classes.append(model.decode(y))
    mat = confusion_matrix(true_classes, pred_classes)
    #zero out diagonal for better viewing.
    for i in range(21):
        mat[i, i] = 0
    sns.set(font_scale=2)
    plt.figure(figsize=(10,8))
    plt.title(title)
    g = sns.heatmap(mat, xticklabels=model.langs, yticklabels=model.langs)
    g.set_xticklabels(g.get_xticklabels(), rotation=0)
    plt.show();
    return mat