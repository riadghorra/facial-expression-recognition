import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import torch


def plot_confusion_matrix(cm, display_labels):
    fig, ax = plt.subplots(figsize=(6, 5))
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


def factorise_emotions(emotions_batch):
    bad = [0, 1, 2, 4]
    good = [3]
    surprise = [5]
    neutral = [6]

    def factorise_emotion(emotion):
        if emotion in bad:
            return 0
        elif emotion in good:
            return 1
        elif emotion in surprise:
            return 2
        elif emotion in neutral:
            return 3

    out = torch.zeros(len(emotions_batch))
    for ligne, emotion in enumerate(emotions_batch):
        out[ligne] = factorise_emotion(emotion)
    return out


def factorise_emotions_vectors(emotions_vector_batch):
    """
    in : (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
    out : (bad=0, good=1, surprise=2, neutral =3)
    """
    bad = [0, 1, 2, 4]
    good = 3
    surprise = 5
    neutral = 6
    out = torch.zeros(len(emotions_vector_batch), 4)
    for ligne, emotions_vector in enumerate(emotions_vector_batch):
        out[ligne][0] = emotions_vector[bad].sum()
        out[ligne][1] = emotions_vector[good].sum()
        out[ligne][2] = emotions_vector[surprise].sum()
        out[ligne][3] = emotions_vector[neutral].sum()
    return out
