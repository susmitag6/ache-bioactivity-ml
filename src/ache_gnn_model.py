import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINEConv, global_mean_pool


class AChEGNNDeep(nn.Module):
    """Four-layer GINE model for molecular pIC50 prediction."""

    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()

        self.atom_encoder = nn.Linear(34, hidden_dim)
        self.bond_encoder = nn.Linear(7, hidden_dim)

        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp))

        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(data.edge_attr)

        for conv in self.convs:
            x = F.relu(
                conv(x, data.edge_index, edge_attr)
            )

        x = global_mean_pool(x, data.batch)

        return self.regressor(x).squeeze(-1)
