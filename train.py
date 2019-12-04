import torch.utils.data
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import numpy as np
from dataset_tools import preprocess_batch_custom_vgg, preprocess_batch_feed_forward, preprocess_batch_vgg16
import json
from tqdm import tqdm
from classifier import FeedForwardNN, vgg16, Custom_vgg

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
def train(model, train_dataframe, test_dataframe, epochs, device, preprocess_batch, loss_function):
    optimizer = optim.Adam(model.parameters(), lr=config["LR"])

    to_dataloader = [[train_dataframe["pixels"][i], train_dataframe["emotion"][i]] for i in range(len(train_dataframe))]
    
    dataloader = torch.utils.data.DataLoader(to_dataloader, config["BATCH"], shuffle=False, drop_last=True)
    model.train()
    print("debut du training")
    best_loss = torch.tensor(10000).to(DEVICE)
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for pixelstring_batch, emotions_batch in dataloader:
            batch, groundtruth = preprocess_batch(pixelstring_batch, emotions_batch, device)
            model.zero_grad()
            out = softmax(model(batch.to(DEVICE)))
            labels = groundtruth.to(DEVICE)
            loss = loss_function(out,labels)
            loss.backward()
            optimizer.step()
        model.eval()
        probatrain, loss_train, acctrain = evaluate(model, train_dataframe, preprocess_batch, loss_function, device)
        proba, loss_test, acc = evaluate(model, test_dataframe, preprocess_batch, loss_function, device)
        if loss_train < best_loss:
            torch.save(model.state_dict(), "current_best_model")
        model.train()
        print()
        print("Epoch number : ", epoch+1)
        print("Accuracy sur le test : ", round(100*float(acc),2), "%")
        print("Proba sur le test : ", round(100*float(proba),2), "%")
        print("Loss test : ", float(loss_test))
        print("Accuracy sur le train : ", round(100*float(acctrain),2), "%")
        print("Proba sur le train: ", round(100*float(probatrain),2), "%")
        print("Loss train : ", float(loss_train))
        print("_______________")
    return model.eval()


def evaluate(model, dataframe, preprocess_batch, loss_function, DEVICE):
    with torch.no_grad():
        to_dataloader = [[dataframe["pixels"][i], dataframe["emotion"][i]] for i in range(len(dataframe))]
        loss = torch.tensor(0.0).to(DEVICE)
        compteur = torch.tensor(0.0).to(DEVICE)
        error = torch.tensor(0.0).to(DEVICE)
        acc = torch.tensor(0.0).to(DEVICE)
        dataloader = torch.utils.data.DataLoader(to_dataloader, config["BATCH"], shuffle=False, drop_last=False)
        for pixelstring_batch, emotions_batch in dataloader:
            batch, groundtruth = preprocess_batch(pixelstring_batch, emotions_batch, DEVICE)

            out = softmax(model(batch.to(DEVICE)))
            labels = groundtruth.to(DEVICE)
            loss += loss_function(out,labels)
            compteur += torch.tensor(1.0).to(DEVICE)
            error += (out*labels).sum()/torch.tensor(len(emotions_batch)).to(DEVICE)
            acc += (out.argmax(1)==labels.argmax(1)).float().mean()
        loss_value = float(loss/compteur)
        proba = float(error/compteur)
        acc = float(acc/compteur)
        return proba, loss_value, acc
        
# =============================================================================
# Main
# =============================================================================


def get_weights_for_loss(train_dataframe):
    d = train_dataframe.groupby("emotion")["pixels"].count().values
    distinct_emotions = np.sort(train_dataframe["emotion"].unique())
    d = d / (sum(d))
    emotion_freq = 1 / d
    emotion_freq_complete = [0] * config["cats"]

    for emotion, frequency in zip(distinct_emotions, emotion_freq):
        emotion_freq_complete[emotion] = frequency

    weights = torch.FloatTensor(emotion_freq_complete).to(DEVICE)
    print("Weights: ", emotion_freq_complete, "Emotions in training set: ", distinct_emotions)
    return weights

    
def main(model, preprocess_batch):
    print("creation du dataset")
    all_data = pd.read_csv(config["path"], header = 0)
    if config["sample"] != 0:
        all_data = all_data[:config["sample"]]
    n_all = len(all_data)
    n_eval = int(config["eval_rate"]*n_all)
    n_test = int(config["test_rate"]*n_eval)
    
    train_dataframe = all_data[n_eval:].reset_index(drop=True)
    eval_dataframe = all_data[:n_eval]
    test_dataframe = eval_dataframe[:n_test]

    # weights for loss
    weight = get_weights_for_loss(train_dataframe)
    loss_function = nn.BCELoss(weight=weight).to(DEVICE)

    # train
    print("Starting model training with:")
    print("learning rate: {}, batch size: {}".format(config["LR"], config["BATCH"]))
    model = train(model, train_dataframe, test_dataframe, config["epochs"], DEVICE, preprocess_batch, loss_function)
    proba, loss_eval, acc = evaluate(model, eval_dataframe, preprocess_batch, loss_function, DEVICE)

    return model, acc, loss_eval, proba


def main_feedforward():
    model = FeedForwardNN(n=48 * 48, hidden_sizes=config["hidden_sizes"], device = DEVICE)
    return main(model, preprocess_batch_feed_forward)


def main_vgg16():
    model = vgg16(DEVICE)
    return main(model, preprocess_batch_vgg16)


def main_custom_vgg(start_from_best_model=True):
    model = Custom_vgg(1, config["cats"], DEVICE)
    if start_from_best_model:
        print("Loading model from current best model")
        model.load_state_dict(torch.load(config["current_best_model"], map_location=DEVICE))
    return main(model, preprocess_batch_custom_vgg)
