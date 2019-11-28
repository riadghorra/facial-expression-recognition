import torch.utils.data
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from dataset_tools import pixelstring_to_torchtensor_feedforward, pixelstring_to_tensor_vgg16, \
                          pixelstring_batch_totensor, emotion_batch_totensor, pixelstring_to_tensor_customvgg
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
def train(model, train_dataframe, test_dataframe, epochs, device, pixelstring_to_tensor, loss_function):
    optimizer = optim.Adam(model.parameters() , lr=config["LR"])
    #train_data = create_datatensor(train_dataframe)
    
    to_dataloader = [[train_dataframe["pixels"][i], train_dataframe["emotion"][i]] for i in range(len(train_dataframe))]
    
    dataloader = torch.utils.data.DataLoader(to_dataloader, config["BATCH"], shuffle=False, drop_last=True)
    model.train()
    print("debut du training")
    best_loss = torch.tensor(10000).to(DEVICE)
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for pixelstring_batch, emotions_batch in dataloader :
            groundtruth = emotion_batch_totensor(emotions_batch)
            batch = pixelstring_batch_totensor(pixelstring_batch, pixelstring_to_tensor)
            
            model.zero_grad()
            out = softmax(model(batch.to(DEVICE)))
            labels = groundtruth.to(DEVICE)
            loss = loss_function(out,labels)
            loss.backward()
            optimizer.step()
        model.eval()
        probatrain, loss_train, acctrain = evaluate(model, train_dataframe, pixelstring_to_tensor, loss_function)
        proba, loss_test, acc = evaluate(model, test_dataframe, pixelstring_to_tensor, loss_function)
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


def evaluate(model, dataframe, pixelstring_to_tensor, loss_function):
    with torch.no_grad():
        to_dataloader = [[dataframe["pixels"][i], dataframe["emotion"][i]] for i in range(len(dataframe))]
        loss = torch.tensor(0.0).to(DEVICE)
        compteur = torch.tensor(0.0).to(DEVICE)
        error = torch.tensor(0.0).to(DEVICE)
        acc = torch.tensor(0.0).to(DEVICE)
        dataloader = torch.utils.data.DataLoader(to_dataloader, config["BATCH"], shuffle=False, drop_last=False)
        for pixelstring_batch, emotions_batch in dataloader :
            groundtruth = emotion_batch_totensor(emotions_batch)
            batch = pixelstring_batch_totensor(pixelstring_batch, pixelstring_to_tensor)
            
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

    
def main(model, pixelstring_to_tensor):
    print("creation du dataset")
    all_data = pd.read_csv(config["path"], header = 0)
    if config["sample"]!=0:
        all_data = all_data[:config["sample"]]
    #all_data["tensors"] = all_data["pixels"].apply(pixelstring_to_tensor)
    #all_data["groundtruth"] = all_data["emotion"].apply(lambda x : label_to_vector(x, device = DEVICE))
    n_all = len(all_data)
    n_eval = int(config["eval_rate"]*n_all)
    n_test = int(config["test_rate"]*n_eval)
    
    train_dataframe = all_data[n_eval:].reset_index(drop=True)
    eval_dataframe = all_data[:n_eval]
    test_dataframe = eval_dataframe[:n_test]
    #weights for loss
    d = train_dataframe.groupby("emotion")["pixels"].count().values
    d = d/(sum(d))
    d = 1/d
    weight = torch.FloatTensor(d).to(DEVICE)
    print("Weights :",d)
    loss_function = nn.BCELoss(weight=weight).to(DEVICE)
    #train
    model = train(model, train_dataframe, test_dataframe, config["epochs"], DEVICE, pixelstring_to_tensor, loss_function)
    proba, loss_eval, acc = evaluate(model, eval_dataframe, pixelstring_to_tensor, loss_function)
    
    return model, acc, loss_eval, proba


def main_feedforward():
    model = FeedForwardNN(n=48 * 48, hidden_sizes=config["hidden_sizes"], device = DEVICE)
    return main(model, lambda x: pixelstring_to_torchtensor_feedforward(x, flatten=True, device = DEVICE))


def main_vgg16():
    model = vgg16(DEVICE)
    return main(model, lambda x: pixelstring_to_tensor_vgg16(x, device = torch.device('cpu')))  

def main_custom_vgg():
    model = Custom_vgg(1, config["cats"], DEVICE)
    return main(model, lambda x : pixelstring_to_tensor_customvgg(x, DEVICE))
        
