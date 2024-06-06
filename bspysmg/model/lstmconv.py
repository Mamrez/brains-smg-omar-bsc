import torch
from torch import nn
import warnings

class LSTMConvModel(nn.Module):
    def __init__(self, model_structure: dict):
        super(LSTMConvModel, self).__init__()
        self.build_model_structure(model_structure)

    def build_model_structure(self, model_structure: dict):
        if model_structure is None:
            model_structure = {}
        self.structure_consistency_check(model_structure)
        
        self.input_channels = model_structure["input_channels"]
        self.output_channels = model_structure["output_channels"]
        self.kernel_size = model_structure["kernel_size"]
        self.stride = model_structure["stride"]
        self.padding = model_structure["padding"]
        self.pool_kernel_size = model_structure["pool_kernel_size"]
        
        self.lstm_input_features = model_structure["lstm_input_features"]
        self.lstm_sequence_length = model_structure["lstm_sequence_length"]
        self.lstm_hidden_size = model_structure["lstm_hidden_size"]
        self.lstm_num_layers = model_structure["lstm_num_layers"]
        self.lstm_output_size = model_structure["lstm_output_size"]

        self.conv_layer = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, self.stride, self.padding)
        self.pool_layer = nn.MaxPool1d(self.pool_kernel_size)
        self.lstm_layer = nn.LSTM(self.lstm_input_features, self.lstm_hidden_size, self.lstm_num_layers, batch_first=True)
        self.output_layer = nn.Linear(self.lstm_hidden_size * self.lstm_sequence_length, self.lstm_output_size)

    def initialize_hidden_state(self, batch_size, dtype, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        hidden_state = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device, dtype=dtype)
        cell_state = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device, dtype=dtype)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.pool_layer(x)
        batch_size, channels, seq_len = x.size()
        x = x.view(batch_size, seq_len, -1)
        lstm_out, self.hidden = self.lstm_layer(x, self.hidden)
        lstm_out = lstm_out.contiguous().view(batch_size, -1)
        return self.output_layer(lstm_out)

    def structure_consistency_check(self, model_structure: dict):
        default_structure = {
            "input_channels": 1,
            "output_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "pool_kernel_size": 2,
            "lstm_input_features": 32,
            "lstm_sequence_length": 10,
            "lstm_hidden_size": 64,
            "lstm_num_layers": 1,
            "lstm_output_size": 1
        }
        for key, default_value in default_structure.items():
            if key not in model_structure:
                model_structure[key] = default_value
                warnings.warn(
                    f"The model loaded does not define the {key} as expected. Changed it to default value: {default_value}."
                )
            else:
                value = model_structure[key]
                assert isinstance(value, int) and value > 0, f"{key} must be a positive integer"
