import torch.utils.data
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from dataset_tools import pixelstring_to_torchtensor_feedforward, label_to_vector, pixelstring_to_tensor_vgg16, \
    create_datatensor_vgg16, create_datatensor_feedforward
from tqdm import tqdm
from classifier import FeedForwardNN, vgg16

"""
(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
"""
# =============================================================================
# Training parameters
# =============================================================================
path = "fer2013.csv"
BATCH = 64
epochs = 25
eval_rate = 0.3
test_rate = 0.1
hidden_sizes = [256,128,64]
sample = 2000
LR = 0.001
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('Initialisation de cuda')
    torch.cuda.init()
else:
    print('Mode CPU')
    DEVICE = torch.device('cpu')
softmax = nn.Softmax(dim=1).to(DEVICE)
loss_function = nn.BCELoss().to(DEVICE)

# =============================================================================
# Train
# =============================================================================
def pixelstring_batch_totensor(psb):
    out = torch.stack(tuple([pixelstring_to_tensor_vgg16(string) for string in psb]))
    return out

def emotion_batch_totensor(emb):
    out = torch.stack(tuple([label_to_vector(em) for em in emb]))
    return out

def train(model, train_dataframe, test_dataframe, epochs, device, create_datatensor):
    optimizer = optim.Adam(model.parameters() , lr=LR)
    #train_data = create_datatensor(train_dataframe)
    
    to_dataloader = [[train_dataframe["pixels"][i], train_dataframe["emotion"][i]] for i in range(len(train_dataframe))]
    
    dataloader = torch.utils.data.DataLoader(to_dataloader, BATCH, shuffle=False, drop_last=True)
    model.train()
    print("debut du training")
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for pixelstring_batch, emotions_batch in dataloader :
            groundtruth = emotion_batch_totensor(emotions_batch)
            batch = pixelstring_batch_totensor(pixelstring_batch)
            
            model.zero_grad()
            out = softmax(model(batch.to(DEVICE)))
            labels = groundtruth.to(DEVICE)
            loss = loss_function(out,labels)
            loss.backward()
            optimizer.step()
        model.eval()
        probatrain, loss_train, acctrain = evaluate(model, train_dataframe, create_datatensor)
        proba, loss_test, acc = evaluate(model, test_dataframe, create_datatensor)
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


def evaluate(model, dataframe, create_datatensor):
    with torch.no_grad():
        to_dataloader = [[dataframe["pixels"][i], dataframe["emotion"][i]] for i in range(len(dataframe))]
        loss = torch.tensor(0.0).to(DEVICE)
        compteur = torch.tensor(0.0).to(DEVICE)
        error = torch.tensor(0.0).to(DEVICE)
        acc = torch.tensor(0.0).to(DEVICE)
        dataloader = torch.utils.data.DataLoader(to_dataloader, BATCH, shuffle=False, drop_last=False)
        for pixelstring_batch, emotions_batch in dataloader :
            groundtruth = emotion_batch_totensor(emotions_batch)
            batch = pixelstring_batch_totensor(pixelstring_batch)
            
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

    
def main(model, pixelstring_to_tensor, create_datatensor):
    print("creation du dataset")
    all_data = pd.read_csv(path, header = 0)[:sample]
    #all_data["tensors"] = all_data["pixels"].apply(pixelstring_to_tensor)
    #all_data["groundtruth"] = all_data["emotion"].apply(lambda x : label_to_vector(x, device = DEVICE))
    n_all = len(all_data)
    n_eval = int(eval_rate*n_all)
    n_test = int(test_rate*n_eval)
    
    train_dataframe = all_data[n_eval:].reset_index(drop=True)
    eval_dataframe = all_data[:n_eval]
    test_dataframe = eval_dataframe[:n_test]

    model = train(model, train_dataframe, test_dataframe, epochs, DEVICE, create_datatensor)
    acc, loss_eval = evaluate(model, eval_dataframe, create_datatensor)
    
    return model, acc, loss_eval


def main_feedforward():
    model = FeedForwardNN(n=48 * 48, hidden_sizes=hidden_sizes, device = DEVICE)
    return main(model, lambda x: pixelstring_to_torchtensor_feedforward(x, flatten=True, device = DEVICE), create_datatensor_feedforward)


def main_vgg16():
    model = vgg16(DEVICE)
    return main(model, lambda x: pixelstring_to_tensor_vgg16(x, device = torch.device('cpu')), create_datatensor_vgg16)     
        
