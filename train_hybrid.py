import torch.utils.data
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import numpy as np
from dataset_tools import preprocess_batch_hybrid
import json
from tqdm import tqdm
from classifier import HybridNetwork
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import *
import ast

"""
(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
"""

# =============================================================================
# Training parameters
# =============================================================================
with open('config.json') as json_file:
    config = json.load(json_file)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('Initialisation de cuda')
    torch.cuda.init()
else:
    print('Mode CPU')
    DEVICE = torch.device('cpu')
softmax = nn.Softmax(dim=1).to(DEVICE)


# =============================================================================
# Train
# =============================================================================
def train_hybrid(model, train_dataframe, quick_eval_dataframe, epochs, device, preprocess_batch, weight):
    optimizer = optim.Adam(model.parameters(), lr=config["LR"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=3,
                                                     verbose=True,
                                                     threshold=0.0001,
                                                     threshold_mode='rel',
                                                     cooldown=0,
                                                     min_lr=0,
                                                     eps=1e-08)
    dataloader = make_dataloader(train_dataframe, shuffle=True, drop_last=True, loss_mode=config["loss_mode"])

    model.train()
    print("debut du training")
    best_acc = 0
    x = []
    test_accs = []
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs_facs = []
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for pixelstring_batch, descriptors_batch, emotions_batch in dataloader:
            batch, descriptors_batch, groundtruth = preprocess_batch(pixelstring_batch, descriptors_batch,
                                                                     emotions_batch, device)

            loss_function = make_loss(groundtruth, weight)

            model.zero_grad()
            out = model(batch.to(DEVICE), descriptors_batch.to(DEVICE))
            groundtruth = groundtruth.to(DEVICE)
            loss = loss_function(out, groundtruth)
            loss.backward()
            optimizer.step()
        probatrain, loss_train, acctrain = evaluate_hybrid(model, train_dataframe, preprocess_batch, weight, device,
                                                           compute_cm=False)
        scheduler.step(loss_train)
        proba, loss_test, acc, cm1, cm2, acc_factorised = evaluate_hybrid(model, quick_eval_dataframe, preprocess_batch,
                                                                          weight, device, compute_cm=True)
        if acc > best_acc:
            torch.save(model.state_dict(), "current_best_model")
        model.train()
        print()
        print("Epoch number : ", epoch + 1)
        print("Accuracy sur le test : ", round(100 * float(acc), 2), "%")
        print("Proba sur le test : ", round(100 * float(proba), 2), "%")
        print("Loss test : ", float(loss_test))
        print("Accuracy sur le train : ", round(100 * float(acctrain), 2), "%")
        print("Proba sur le train: ", round(100 * float(probatrain), 2), "%")
        print("Loss train : ", float(loss_train))
        print("Accuracy with grouped classes : ", round(100 * float(acc_factorised), 2), "%")
        print("_______________")
        x.append(epoch)
        test_accs.append(float(acc))
        test_losses.append(float(loss_test))
        test_accs_facs.append(float(acc_factorised))
        train_accs.append(float(acctrain))
        train_losses.append(float(loss_train))

        fig, axes = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
        ax0, ax1 = axes.flatten()

        ax0.plot(x, train_accs, label="Accuracy on train")
        ax0.plot(x, test_accs, label="Accuracy on test")
        ax0.legend(loc='upper left', frameon=False)
        ax0.grid()
        ax0.set_xlabel("epoch")

        ax1.plot(x, train_losses, label="Loss on train")
        ax1.legend(loc='upper left', frameon=False)
        ax1.grid()
        ax1.set_xlabel("epoch")

        plot_confusion_matrix(cm1, config["catslist"])
        plot_confusion_matrix(cm2, ["bad", "good", "surprise", "neutral"])

        plt.figure()
        plt.plot(x, test_accs_facs, label="Accuracy on test with factorised classes")
        plt.grid()
        plt.xlabel("epochs")

    return model.eval()


def evaluate_hybrid(model, dataframe, preprocess_batch, weight, DEVICE, compute_cm=False):
    with torch.no_grad():
        model.eval()
        dataloader = make_dataloader(dataframe, shuffle=False, drop_last=False, loss_mode=config["loss_mode"])
        loss = torch.tensor(0.0).to(DEVICE)
        compteur = torch.tensor(0.0).to(DEVICE)
        probasum = torch.tensor(0.0).to(DEVICE)
        acc_all = torch.tensor(0.0).to(DEVICE)
        acc_factorised = torch.tensor(0.0).to(DEVICE)

        y_pred = torch.tensor([]).to(DEVICE)
        y_true = torch.tensor([]).to(DEVICE)

        for pixelstring_batch, descriptors_batch, emotions_batch in dataloader:
            batch, descriptors_batch, groundtruth = preprocess_batch(pixelstring_batch, descriptors_batch, emotions_batch, DEVICE)
            loss_function = make_loss(groundtruth, weight)

            out = model(batch.to(DEVICE), descriptors_batch.to(DEVICE))
            groundtruth = groundtruth.to(DEVICE)
            loss += loss_function(out, groundtruth)
            compteur += torch.tensor(1.0).to(DEVICE)

            if config["loss_mode"] == "BCE":
                out_norm_batch = norm_L2_batch(out)
                groundtruth_norm_batch = norm_L2_batch(groundtruth)
                out_dot_groundtruth_batch = dot_product_batch(out, groundtruth)
                cosine_similarity_batch = out_dot_groundtruth_batch / (out_norm_batch * groundtruth_norm_batch)
                probasum += cosine_similarity_batch.mean().to(DEVICE)
                acc_all += (out.argmax(1) == groundtruth.argmax(1)).float().mean()
                if compute_cm:
                    y_pred = torch.cat((y_pred, out.argmax(1).float()))
                    y_true = torch.cat((y_true, groundtruth.argmax(1).float()))
                    probas_batch = softmax(out)
                    acc_factorised += (factorise_emotions_vectors(probas_batch).argmax(1) == factorise_emotions_vectors(
                        groundtruth).argmax(1)).float().mean().to(DEVICE)

            if config["loss_mode"] == "CE":
                probas_batch = softmax(out)
                probasum += (torch.tensor(
                    [probas_batch[image_index][classe] for image_index, classe in
                     enumerate(groundtruth)]).sum() / float(
                    len(groundtruth))).to(DEVICE)
                acc_all += (probas_batch.argmax(1) == groundtruth).float().mean().to(DEVICE)
                if compute_cm:
                    y_pred = torch.cat((y_pred, probas_batch.argmax(1).float()))
                    y_true = torch.cat((y_true, groundtruth.float()))
                    acc_factorised += (factorise_emotions_vectors(probas_batch).argmax(1) == factorise_emotions(
                        groundtruth).argmax(0)).float().mean().to(
                        DEVICE)

        loss_value = float(loss / compteur)
        proba = float(probasum / compteur)
        acc = float(acc_all / compteur)

        if compute_cm:
            cm1 = confusion_matrix(y_true.cpu(), y_pred.cpu(), normalize='true')
            cm2 = confusion_matrix(factorise_emotions(y_true.cpu()), factorise_emotions(y_pred.cpu()), normalize='true')
            acc_fact = float(acc_factorised / compteur)
            return proba, loss_value, acc, cm1, cm2, acc_fact
        return proba, loss_value, acc


# =============================================================================
# Main
# =============================================================================


def get_weights_for_loss(train_dataframe):
    d = train_dataframe.groupby("emotion")["pixels"].count().values
    distinct_emotions = np.sort(train_dataframe["emotion"].unique())
    d = d / (sum(d))
    emotion_freq = 1 / d
    emotion_freq_complete = [0] * len(config["catslist"])

    for emotion, frequency in zip(distinct_emotions, emotion_freq):
        emotion_freq_complete[emotion] = frequency

    weights = torch.FloatTensor(emotion_freq_complete).to(DEVICE)
    print("Weights: ", emotion_freq_complete, "Emotions in training set: ", distinct_emotions)
    return weights


def make_loss(emotions_batch, weights):
    """
    :param batch_ndim: if batch.size() is 64,1,48,48 batch_ndim is 4.
    """
    assert config["loss_mode"] in ["CE", "BCE"], "mode inconnu"
    if config["loss_mode"] == "BCE":
        loss_weights = torch.matmul(weights.to(DEVICE), emotions_batch.T.float().to(DEVICE))
        loss_weights = loss_weights.unsqueeze(1)
        return lambda x, y: nn.BCELoss(weight=loss_weights).to(DEVICE)(softmax(x), y)
    elif config["loss_mode"] == "CE":
        return nn.CrossEntropyLoss(weight=weights).to(DEVICE)


def make_dataloader(dataframe, shuffle=False, drop_last=False, loss_mode="CE"):
    """
    :param dataframe: columns {config["data_column"]} with pixels and "emotion" for annotation.
    """
    if loss_mode == "CE":
        to_dataloader = [[dataframe[config["data_column"]][i], dataframe["descriptors"][i], dataframe["emotion"][i]] for
                         i in range(len(dataframe))]
    else:
        to_dataloader = [[dataframe[config["data_column"]][i], dataframe["descriptors"][i],
                          ast.literal_eval(dataframe["emotions_tensor"][i])] for i in range(len(dataframe))]

    return torch.utils.data.DataLoader(to_dataloader, config["BATCH"], shuffle=shuffle, drop_last=drop_last)


def main(model, preprocess_batch):
    print("creation du dataset")
    all_data = pd.read_csv(config["path"], header=0)
    n_quick_eval = int(config["quick_eval_rate"] * len(all_data[all_data["attribution"] == "val"]))

    train_dataframe = all_data[all_data["attribution"] == "train"].reset_index()
    eval_dataframe = all_data[all_data["attribution"] == "val"].reset_index()
    quick_eval_dataframe = eval_dataframe[:n_quick_eval].reset_index()

    # weights for loss
    weight = get_weights_for_loss(train_dataframe)

    # train
    print("Starting model training with:")
    print("learning rate: {}, batch size: {}".format(config["LR"], config["BATCH"]))
    model = train_hybrid(model, train_dataframe, quick_eval_dataframe, config["epochs"], DEVICE, preprocess_batch,
                         weight)
    try:
        proba, loss_eval, acc, cm = evaluate_hybrid(model, eval_dataframe, preprocess_batch, weight, DEVICE,
                                                compute_cm=True)
    except ValueError:
        proba, loss_eval, acc = evaluate_hybrid(model, eval_dataframe, preprocess_batch, weight, DEVICE,
                                                    compute_cm=False)
        return model, acc, loss_eval, proba, cm

    return model, acc, loss_eval, proba, cm


def main_custom_vgg(start_from_best_model=True, with_data_aug=True):
    model = HybridNetwork(1, len(config["catslist"]), DEVICE)
    if start_from_best_model:
        print("Loading model from current best model")
        model.load_state_dict(torch.load(config["current_best_model"], map_location=DEVICE))
    return main(model, lambda pixelstring_batch, descriptors_batch, emotions_batch, DEVICE: preprocess_batch_hybrid(
        pixelstring_batch, descriptors_batch, emotions_batch, DEVICE, with_data_aug, config["loss_mode"]))
