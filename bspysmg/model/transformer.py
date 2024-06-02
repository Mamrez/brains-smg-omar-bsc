import torch
from torch import nn
import warnings

class TransformerModel(nn.Module):
    def __init__(self, model_structure: dict):
        super(TransformerModel, self).__init__()
        self.build_model_structure(model_structure)

    def build_model_structure(self, model_structure: dict):
        if model_structure is None:
            model_structure = {}
        self.structure_consistency_check(model_structure)
        
        self.input_features = model_structure["input_features"]
        self.sequence_length = model_structure["sequence_length"]
        self.hidden_size = model_structure["hidden_size"]
        self.num_layers = model_structure["num_layers"]
        self.num_heads = model_structure.get("num_heads", 8)
        
        self.embedding = nn.Linear(self.input_features, self.hidden_size)
        self.pos_encoder = nn.Embedding(self.sequence_length, self.hidden_size)
        self.transformer = nn.Transformer(d_model=self.hidden_size, nhead=self.num_heads, num_encoder_layers=self.num_layers)
        self.fc_out = nn.Linear(self.hidden_size, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        assert isinstance(src, torch.Tensor), "Input to the forward pass can only be a Pytorch tensor"
        
        batch_size, seq_len, _ = src.size()
        positions = torch.arange(0, seq_len, device=src.device).unsqueeze(0).expand(batch_size, -1)
        src = self.embedding(src) + self.pos_encoder(positions)
        
        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, hidden_size)
        transformer_out = self.transformer(src, src)
        transformer_out = transformer_out.permute(1, 0, 2)  # Back to (batch_size, seq_len, hidden_size)
        
        out = self.fc_out(transformer_out[:, -1, :])  # Take the output of the last time step
        return out

    def structure_consistency_check(self, model_structure: dict):
        default_input_features = 7
        default_sequence_length = 100
        default_hidden_size = 512
        default_num_layers = 6
        default_num_heads = 8

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

        if "num_heads" not in model_structure:
            model_structure["num_heads"] = default_num_heads
            warnings.warn(
                "The model loaded does not define the number of heads as expected. Changed it to default value: {}.".format(default_num_heads)
            )
        else:
            num_heads = model_structure.get('num_heads')
            assert isinstance(num_heads, int) and num_heads > 0, "num_heads must be a positive integer"


# Example usage of the TransformerModel for testing

# if __name__ == "__main__":
#     model_structure = {
#         "input_features": 5,
#         "sequence_length": 10,
#         "hidden_size": 512,
#         "num_layers": 6,
#         "num_heads": 8
#     }

#     # Initialize the Transformer model
#     transformer_model = TransformerModel(model_structure)

#     # Sample input tensor (batch_size=3, sequence_length=10, input_features=5)
#     input_tensor = torch.randn(3, 10, 5)

#     # Perform a forward pass
#     output = transformer_model(input_tensor)
#     print(output)
