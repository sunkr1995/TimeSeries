import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import matplotlib.pyplot as plt 
from darts import TimeSeries
from models import *
from dataloader import get_dataloader

def train(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    return total_loss / len(dataloader)

def predict(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            outputs = model(x)
            predictions.extend(outputs[:,0,:].cpu().numpy())
    return np.array(predictions)

def show_predict(config,train_loader,test_loader,model,best_model_path,device):
    series1 = TimeSeries.from_values(train_loader.dataset.data[config['input_length']:])
    back1 = train_loader.dataset.scale.inverse_transform(series1)
    model.load_state_dict(torch.load(best_model_path))
    predictions = predict(model,test_loader, device)
    series2 = TimeSeries.from_values(predictions)
    back2 = train_loader.dataset.scale.inverse_transform(series2)
    back1.plot()
    back2.plot()
    plt.show()


def main(config,SHOW = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_dataloader(config['train_data_path'], config['input_length'], config['output_length'], config['num_variables'], config['batch_size'])
    test_loader = get_dataloader(config['test_data_path'], config['input_length'], config['output_length'], config['num_variables'], 1, shuffle=False)

    if config['model_type'] == 'RNN':
        with open('model_config/RNN.json', 'r') as f:
            model_config = json.load(f)
        model = RNNModel(config['input_length'], config['num_variables'], model_config['hidden_size'], config['output_length'],
                         model_config['num_layers'],model_config['dropout'],rnn_type = model_config['rnn_type']).to(device)
    elif config['model_type'] == 'NBEATS':
        with open('model_config/NBEATS.json', 'r') as f:
            model_config = json.load(f)
        model = NBEATSModel(config['num_variables'],config['input_length'],config['num_variables'],config['output_length'],
                            model_config["generic_architecture"],model_config["num_stacks"],model_config['num_blocks'],model_config['num_layers'],model_config['layer_widths'],
                            model_config['expansion_coefficient_dim'],model_config['trend_polynomial_degree'],model_config['batch_norm'],model_config['dropout'],model_config['activation']).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma'])

    best_loss = float('inf')
    best_model_path = config.get('best_model_save_path', 'best_model.pth')

    for epoch in range(config['num_epochs']):
        train_loss = train(model, train_loader, criterion, optimizer, scheduler, device)
        #test_loss = test(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1}/{config["num_epochs"]}, Train Loss: {train_loss},best_loss: {best_loss}')

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_model_path)

    print(f'Best Validation Loss: {best_loss}')
    if SHOW:
        show_predict(config,train_loader,test_loader,model,best_model_path,device)
    
if __name__ == '__main__':
    with open('global_config.json', 'r') as f:
        config = json.load(f)

    """
    关于多次遍历,可读文件,或者遍历文件夹,对应修改config["train_data_path"]、config["test_data_path"]、config["best_model_save_path"]
    例如：
    for i in os.listdir("path/to/traindata/dir"):
        config["train_data_path"] = os.path.join("path/to/traindata/dir",i)
        config["test_data_path"] = os.path.join("path/to/traindata/dir",i)
        config["best_model_save_path"] = os.path.join("path/to/modeltype/",  "i model name")\
        main(config)
    """
    main(config,SHOW=True)
