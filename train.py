import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, input_length, output_length, num_variables):
        self.data = np.load(data_path)
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
