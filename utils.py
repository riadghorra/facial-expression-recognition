import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import torch


def plot_confusion_matrix(cm, display_labels):
    fig, ax = plt.subplots(figsize=(12, 10))
    n_classes = cm.shape[0]
    im_ = ax.imshow(cm, interpolation='nearest', cmap="viridis")
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    text_ = np.empty_like(cm, dtype=object)
    values_format = '.2g'
    # print text with appropriate color depending on background
    thresh = (cm.max() + cm.min()) / 2.0
    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        text_[i, j] = ax.text(j, i, format(cm[i, j], values_format), ha="center", va="center", color=color)

    fig.colorbar(im_, ax=ax)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=display_labels,
           yticklabels=display_labels,
           ylabel="True label",
           xlabel="Predicted label")

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation="vertical")

    plt.show()


def dot_product_batch(m1, m2):
    """

    :param m1: tensor
    :param m2: tensor
    :return: vector of dot product of each rows of m1 and rows of m2
    """
    return torch.diag(torch.matmul(m1, m2.T))


def norm_L2_batch(m):
    return torch.sqrt(dot_product_batch(m, m))
