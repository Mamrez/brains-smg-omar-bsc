import os
import torch
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Check CUDA availability and initialize context
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Check your CUDA installation.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.cuda.init()

# Define a simple Transformer model for testing
class SimpleTransformer(nn.Module):
    def __init__(self):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(10, 512)
        self.pos_encoder = nn.Embedding(100, 512)
        self.transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6)
        self.fc_out = nn.Linear(512, 1)
        
    def forward(self, src):
        batch_size, seq_len, _ = src.size()
        positions = torch.arange(0, seq_len, device=src.device).unsqueeze(0).expand(batch_size, -1)
        src = self.embedding(src) + self.pos_encoder(positions)
        src = src.permute(1, 0, 2)
        transformer_out = self.transformer(src, src)
        transformer_out = transformer_out.permute(1, 0, 2)
        out = self.fc_out(transformer_out[:, -1, :])
        return out

# Check CUDA context
def check_cuda_context():
    try:
        torch.cuda.current_device()
        print("CUDA context initialized successfully.")
    except RuntimeError as e:
        print(f"Error initializing CUDA context: {e}")

check_cuda_context()

# Initialize and test the model
model = SimpleTransformer().to(device)
inputs = torch.randn(32, 100, 10).to(device)
outputs = model(inputs)
print("Outputs:", outputs)
