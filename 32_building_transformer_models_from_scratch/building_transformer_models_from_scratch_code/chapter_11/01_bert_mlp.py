import torch.nn as nn

class BertMLP(nn.Module):
    def __init__(self, dim, intermediate_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, dim)
        self.gelu = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
