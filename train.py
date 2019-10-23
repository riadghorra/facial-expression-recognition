import torch.utils.data
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from dataset_tools import pixelstring_to_torchtensor, create_datatensor, label_to_vector
from tqdm import tqdm
from classifier import classifier

"""
(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
"""
# =============================================================================
# Training parameters
# =============================================================================


path = "fer2013.csv"
BATCH = 100
epochs = 25
eval_rate = 0.3
test_rate = 0.1
hidden_sizes = [256,128,64]
sample = 1000
LR = 0.0001
loss_function = nn.BCELoss()
DEVICE = torch.device('cpu')
softmax = nn.Softmax(dim=1)

# =============================================================================
# Train
# =============================================================================

def train(model, train_dataframe, test_dataframe, epochs, device):
    optimizer = optim.Adam(model.parameters() , lr=LR)
    train_data = create_datatensor(train_dataframe)
    
    to_dataloader = [[train_data[i], train_dataframe["groundtruth"][i]] for i in range(len(train_data))]
    
    dataloader = torch.utils.data.DataLoader(to_dataloader, BATCH, shuffle = False, drop_last=True)
    model.train()
    print("debut du training")
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for batch, groundtruth in dataloader :
            model.zero_grad()
            out = softmax(model(batch))
            labels = torch.tensor(groundtruth)
            loss = loss_function(out,labels)
            loss.backward()
            optimizer.step()
        model.eval()
        acctrain, loss_train = evaluate(model, train_dataframe)
        acc, loss_test = evaluate(model, test_dataframe)
        model.train()
        print()
        print("Epoch number : ", epoch+1)
        print("Accuracy : ", round(100*float(acc),2), "%")
        print("Loss test : ", float(loss_test))
        print("Accuracy train: ", round(100*float(acctrain),2), "%")
        print("Loss train : ", float(loss_train))
        print("_______________")
    return model.eval()

def evaluate(model, dataframe):
    with torch.no_grad():
        data = create_datatensor(dataframe)
        out = softmax(model(data))
        labels = torch.tensor([])
        for tensor in dataframe["groundtruth"]:
            labels = torch.cat((labels, tensor.unsqueeze(0)))
        loss_value = loss_function(out,labels)
        acc = sum(sum(out*labels))/(data.size()[0])
        return acc, loss_value
        
        
# =============================================================================
# Main
# =============================================================================


    
def main():
    print("creation du dataset")
    all_data = pd.read_csv(path, header = 0)[:sample]
    all_data["tensors"] = all_data["pixels"].apply(lambda x : pixelstring_to_torchtensor(x, flatten= True))
    all_data["groundtruth"] = all_data["emotion"].apply(label_to_vector)
    n_all = len(all_data)
    n_eval = int(eval_rate*n_all)
    n_test = int(test_rate*n_eval)
    
    train_dataframe = all_data[n_eval:].reset_index(drop=True)
    eval_dataframe = all_data[:n_eval]
    test_dataframe = eval_dataframe[:n_test]
    
    """
    train_data = create_datatensor(train_dataframe)
    
    eval_data = create_datatensor(eval_dataframe)
    test_data = create_datatensor(test_dataframe)
    """

    model = classifier(n = 48*48, hidden_sizes = hidden_sizes)
    
    model = train(model, train_dataframe, test_dataframe, epochs, DEVICE)
    acc, loss_eval = evaluate(model, eval_dataframe)
    
    return model, acc, loss_eval
    