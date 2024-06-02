import torch
from torch import nn
import warnings

class GRUModel(nn.Module):
    def __init__(self, model_structure: dict):
        super(GRUModel, self).__init__()
        self.build_model_structure(model_structure)

    def build_model_structure(self, model_structure: dict):
        if model_structure is None:
            model_structure = {}
        self.structure_consistency_check(model_structure)
        
        self.input_features = model_structure["input_features"]
        self.sequence_length = model_structure["sequence_length"]
        self.hidden_size = model_structure["hidden_size"]
        self.num_layers = model_structure["num_layers"]

        self.gru_layer = nn.GRU(input_size=self.input_features, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size * self.sequence_length, 1)

    def initialize_hidden_state(self, batch_size, dtype, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), "Input to the forward pass can only be a Pytorch tensor"
        batch_size, seq_len, _ = x.size()
        gru_out, self.hidden = self.gru_layer(x, self.hidden)
        gru_out = gru_out.contiguous().view(batch_size, -1)
        return self.output_layer(gru_out)

    def structure_consistency_check(self, model_structure: dict):
        default_input_features = 1
        default_sequence_length = 10
        default_hidden_size = 20
        default_num_layers = 1

        if "input_features" not in model_structure:
            model_structure["input_features"] = default_input_features
            warnings.warn(
                "The model loaded does not define the input features as expected. Changed it to default value: {}.".format(default_input_features)
            )
        else:
            input_features = model_structure.get('input_features')
            assert isinstance(input_features, int) and input_features > 0, "input_features must be a positive integer"

        if "sequence_length" not in model_structure:
            model_structure["sequence_length"] = default_sequence_length
            warnings.warn(
                "The model loaded does not define the sequence length as expected. Changed it to default value: {}.".format(default_sequence_length)
            )
        else:
            sequence_length = model_structure.get('sequence_length')
            assert isinstance(sequence_length, int) and sequence_length > 0, "sequence_length must be a positive integer"

        if "hidden_size" not in model_structure:
            model_structure["hidden_size"] = default_hidden_size
            warnings.warn(
                "The model loaded does not define the hidden size as expected. Changed it to default value: {}.".format(default_hidden_size)
            )
        else:
            hidden_size = model_structure.get('hidden_size')
            assert isinstance(hidden_size, int) and hidden_size > 0, "hidden_size must be a positive integer"

        if "num_layers" not in model_structure:
            model_structure["num_layers"] = default_num_layers
            warnings.warn(
                "The model loaded does not define the number of layers as expected. Changed it to default value: {}.".format(default_num_layers)
            )
        else:
            num_layers = model_structure.get('num_layers')
            assert isinstance(num_layers, int) and num_layers > 0, "num_layers must be a positive integer"
