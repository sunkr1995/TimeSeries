import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_length, num_variables, hidden_size, output_length, num_layers=3, dropout=0.5,rnn_type = "lstm"):
        super(RNNModel, self).__init__()
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(num_variables, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(num_variables, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SELU(),  # activation function can be replaced with nn.RELU or nn.GELU etc.
            nn.Dropout(dropout),
            
        )
        self.fc2 = nn.Linear(hidden_size, output_length * num_variables)
        self.output_length = output_length
        self.num_variables = num_variables

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out[:, -1, :]  # Use the output of the last time step
        fc1_out = self.fc1(rnn_out)
        fc2_out = self.fc2(fc1_out)
        out = fc2_out.view(-1, self.output_length, self.num_variables)

        ## 后续将fc1_out 作为特征抽出
        return out