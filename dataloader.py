import torch
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
from darts import TimeSeries 
import numpy as np
from darts.dataprocessing.transformers import MissingValuesFiller,Scaler
import matplotlib.pyplot as plt

def has_suffix(s, suffix):
    return s.endswith(suffix)

def csv2npy(data_path):
    df = pd.read_csv(data_path)
    df['time'] = pd.PeriodIndex(df['datacqtr'], freq='Q').to_timestamp()
    series = TimeSeries.from_dataframe(df, 'time', 'atq')
    filler = MissingValuesFiller()
    filled_series = filler.transform(series)
    scaler = Scaler()
    rescaled = scaler.fit_transform(filled_series)

    series_numpy = rescaled.values()
    #print (series_numpy)
    return series_numpy,scaler

class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, input_length, output_length, num_variables):
        if has_suffix(data_path, '.csv'):
            self.data,self.scale = csv2npy(data_path)
        elif has_suffix(data_path, '.npy'):
            self.data = np.load(data_path)
        else:
            raise ValueError
        self.input_length = input_length
        self.output_length = output_length
        self.num_variables = num_variables

    def __len__(self):
        return len(self.data) - self.input_length - self.output_length + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.input_length, :self.num_variables]
        y = self.data[idx + self.input_length:idx + self.input_length + self.output_length, :self.num_variables]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def get_dataloader(data_path, input_length, output_length, num_variables, batch_size, shuffle=True):
    dataset = TimeSeriesDataset(data_path, input_length, output_length, num_variables)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    train_loader = get_dataloader(config['train_data_path'], config['input_length'], config['output_length'], config['num_variables'], 1,False)
    predictions = []
    for x,y in train_loader:
        predictions.extend(y[:,0,:].cpu().numpy())
    predictions =np.array(predictions)
    print(predictions.shape)
    series1 = TimeSeries.from_values(train_loader.dataset.data[10:])
    series1.plot()
    print(series1)
    time_index = pd.RangeIndex(0,len(predictions), 1)
    series2 = TimeSeries.from_times_and_values(time_index,predictions)
    
    #series2.time_index = pd.RangeIndex(start=series1.time_index.size - series2.time_index.size,stop=series1.time_index.size,step=1,name='time')
    series2.plot()
    plt.show()
