import torch.utils.data
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from dataset_tools import pixelstring_to_torchtensor, label_to_vector, pixelstring_to_tensor_vgg16, \
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


def train(model, train_dataframe, test_dataframe, epochs, device, create_datatensor):
    optimizer = optim.Adam(model.parameters() , lr=LR)
    train_data = create_datatensor(train_dataframe)
    
    to_dataloader = [[train_data[i], train_dataframe["groundtruth"][i]] for i in range(len(train_data))]
    
    dataloader = torch.utils.data.DataLoader(to_dataloader, BATCH, shuffle=False, drop_last=True)
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
        acctrain, loss_train = evaluate(model, train_dataframe, create_datatensor)
        acc, loss_test = evaluate(model, test_dataframe, create_datatensor)
        model.train()
        print()
        print("Epoch number : ", epoch+1)
        print("Accuracy : ", round(100*float(acc),2), "%")
        print("Loss test : ", float(loss_test))
        print("Accuracy train: ", round(100*float(acctrain),2), "%")
        print("Loss train : ", float(loss_train))
        print("_______________")
    return model.eval()


def evaluate(model, dataframe, create_datatensor):
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

    
def main(model, pixelstring_to_tensor, create_datatensor):
    print("creation du dataset")
    all_data = pd.read_csv(path, header = 0)[:sample]
    all_data["tensors"] = all_data["pixels"].apply(pixelstring_to_tensor)
    all_data["groundtruth"] = all_data["emotion"].apply(label_to_vector)
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
    model = FeedForwardNN(n=48 * 48, hidden_sizes=hidden_sizes)
    return main(model, lambda x: pixelstring_to_torchtensor(x, flatten=True), create_datatensor_feedforward)


def main_vgg16():
    model = vgg16()
    return main(model, pixelstring_to_tensor_vgg16, create_datatensor_vgg16)
