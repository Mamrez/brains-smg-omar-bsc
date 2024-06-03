import torch
from torch import nn
import warnings
from torch_geometric_temporal.nn.recurrent import RecurrentGCN

class TemporalModel(nn.Module):
    def __init__(self, model_structure: dict):
        super(TemporalModel, self).__init__()
        self.build_model_structure(model_structure)
        self.init_weights()

    def build_model_structure(self, model_structure: dict):
        if model_structure is None:
            model_structure = {}
        self.structure_consistency_check(model_structure)

        self.input_features = model_structure["input_features"]
        self.sequence_length = model_structure["sequence_length"]
        self.hidden_size = model_structure["hidden_size"]
        self.num_layers = model_structure["num_layers"]
        
        self.model = RecurrentGCN(in_channels=self.input_features, out_channels=self.hidden_size, num_layers=self.num_layers)
        self.fc_out = nn.Linear(self.hidden_size, 1)

    def init_weights(self):
        initrange = 0.1
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        # Initialize RecurrentGCN layers if needed

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        assert isinstance(src, torch.Tensor), "Input to the forward pass can only be a Pytorch tensor"
        
        batch_size, seq_len, _ = src.size()
        out = self.model(src)
        out = self.fc_out(out[:, -1, :])  # Take the output of the last time step
        return out

    def structure_consistency_check(self, model_structure: dict):
        default_input_features = 7
        default_sequence_length = 100
        default_hidden_size = 32
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


# Example usage of the TemporalModel for testing

# if __name__ == "__main__":
#     model_structure = {
#         "input_features": 5,
#         "sequence_length": 10,
#         "hidden_size": 32,
#         "num_layers": 1
#     }

#     # Initialize the Temporal model
#     temporal_model = TemporalModel(model_structure)

#     # Sample input tensor (batch_size=3, sequence_length=10, input_features=5)
#     input_tensor = torch.randn(3, 10, 5)

#     # Perform a forward pass
#     output = temporal_model(input_tensor)
#     print(output)
