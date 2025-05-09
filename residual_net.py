import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)  # Add dropout layer
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(out_features, out_features)

        # If the input and output features differ, use a linear layer to match dimensions
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else None

    def forward(self, x):
        # Save the input for the residual connection
        identity = x

        # Apply the first linear layer and activation
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)  # Apply dropout after the first activation

        # Apply the second linear layer
        out = self.linear2(out)

        # Add the shortcut (identity) connection
        if self.shortcut:
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)
        out = self.dropout2(out)
        return out

class ResidualMappingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=2):
        super(ResidualMappingNetwork, self).__init__()

        self.hidden_dim = 4096

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)  # First layer
        self.res_layers = nn.Sequential(
            *[ResidualBlock(self.hidden_dim, self.hidden_dim) for _ in range(n_layers)])
        self.fc2 = nn.Linear(self.hidden_dim, output_dim)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.res_layers(x)
        x = self.fc2(x)
        return x