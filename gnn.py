import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import MessagePassing, GINConv
from torch_scatter import scatter
from torch_geometric.nn import GATConv,GCNConv
from torch_geometric.nn import TransformerConv
from torch.nn import Module


############## MPNN ##############
class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim, edge_dim, aggr='add'):
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        self.mlp_msg = Sequential(
            Linear(2 * emb_dim + edge_dim, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU(),
            Linear(emb_dim, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU()
        )

        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU(),
            Linear(emb_dim, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU()
        )

    def forward(self, h, edge_index, edge_attr):
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr, num_nodes=h.size(0))
        return out

    def message(self, h_i, h_j, edge_attr):
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)

    def aggregate(self, inputs, index, dim_size=None):
        num_nodes = dim_size
        if num_nodes is None:
            num_nodes = index.max().item() + 1 if index.numel() > 0 else 0
        out = torch.zeros(num_nodes, self.emb_dim, dtype=inputs.dtype, device=inputs.device)
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr, out=out)

    def update(self, aggr_out, h):
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})'

class MPNNModel(Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=-1, edge_dim=-1, out_dim=1):
        super().__init__()

        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim

        self.lin_in = torch.nn.LazyLinear(emb_dim)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, self.edge_dim, aggr='add'))

        # Removed the global pooling layer
        # Linear prediction head for each node
        self.lin_pred = torch.nn.LazyLinear(out_dim)

    def forward(self, data):
        h = self.lin_in(data.x)  # (n_nodes_batch, emb_dim)

        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr)  # (n_nodes_batch, emb_dim)

        # Directly predict for each node
        out = self.lin_pred(h)  # (n_nodes_batch, out_dim)

        return out.squeeze(1) # (n_nodes_batch,)

############## GAT ##############

class GATModel(Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=-1, edge_dim=-1, out_dim=1, num_heads=8):
        super().__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.lin_in = torch.nn.LazyLinear(emb_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_channels = emb_dim if i > 0 else emb_dim  # After the first layer
            self.convs.append(GATConv(in_channels, emb_dim // num_heads, heads=num_heads, edge_dim=edge_dim))

        self.lin_pred = torch.nn.LazyLinear(out_dim)

    def forward(self, data):
        h = self.lin_in(data.x)

        for conv in self.convs:
            h = conv(h, data.edge_index, data.edge_attr)

        out = self.lin_pred(h)
        return out.squeeze(1)
    
############## GraphTransformerNet ##############
class GraphTransformerNet(torch.nn.Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=-1, edge_dim=-1, out_dim=1, num_heads=8):
        super().__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.lin_in = torch.nn.LazyLinear(emb_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_channels = emb_dim if i > 0 else emb_dim
            self.convs.append(TransformerConv(in_channels, emb_dim // num_heads, heads=num_heads, edge_dim=edge_dim))

        self.lin_pred = torch.nn.LazyLinear(out_dim)

    def forward(self, data):
        h = self.lin_in(data.x)
        for conv in self.convs:
            h = conv(h, data.edge_index, data.edge_attr)
        out = self.lin_pred(h)
        return out.squeeze(1)

################## Graph Isomorphism Network ##################

class GINLayer(torch.nn.Module):
    def __init__(self, emb_dim, edge_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        
        # Edge feature processing
        self.edge_encoder = Sequential(
            Linear(edge_dim, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU(),
            Linear(emb_dim, emb_dim)
        )
        
        # Main GIN MLP
        self.mlp = Sequential(
            Linear(emb_dim, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU(),
            Linear(emb_dim, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU()
        )
        
        self.conv = GINConv(self.mlp)

    def forward(self, h, edge_index, edge_attr):
        # Process edge features
        edge_emb = self.edge_encoder(edge_attr)
        
        # Aggregate edge features to nodes
        aggregated_edge_emb = scatter(edge_emb, edge_index[1], dim=0, dim_size=h.size(0))
        
        # Combine with node features
        h = h + aggregated_edge_emb
        
        # GIN convolution
        return self.conv(h, edge_index)

class GINModel(Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=140, edge_dim=3, out_dim=1):  # Changed from LazyLinear
        super().__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.in_dim = in_dim  # Should match total_feature_dim (140)
        self.edge_dim = edge_dim
        self.out_dim = out_dim

        # Replace LazyLinear with explicit dimensions
        self.lin_in = Linear(in_dim, emb_dim)
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GINLayer(emb_dim, edge_dim))

        self.lin_pred = Linear(emb_dim, out_dim)

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, data):
        # Initial projection
        h = self.lin_in(data.x)  # (n_nodes_batch, emb_dim)

        # Message passing layers with residual connections
        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr)  # (n_nodes_batch, emb_dim)

        # Final prediction
        out = self.lin_pred(h)  # (n_nodes_batch, out_dim)

        return out.squeeze(1)  # (n_nodes_batch,)

    def __repr__(self):
        return (f'{self.__class__.__name__}(num_layers={self.num_layers}, '
                f'emb_dim={self.emb_dim}, edge_dim={self.edge_dim}, '
                f'out_dim={self.out_dim})')
    
################## Graph Isomorphism Network ##################
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class Readout(nn.Module):
    def forward(self, x, batch):
        return global_mean_pool(x, batch)

class Discriminator(nn.Module):
    def forward(self, z, summary):
        return torch.sigmoid(torch.sum(z * summary, dim=1))

class DGI(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels)
        self.readout = Readout()
        self.discriminator = Discriminator()
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, GCNConv):
                module.reset_parameters()

    def forward(self, data, corrupted_data):
        # Encode original graph
        z = self.encoder(data.x, data.edge_index)
        summary = self.readout(z, data.batch)

        # Encode corrupted graph
        z_corrupted = self.encoder(corrupted_data.x, corrupted_data.edge_index)
        summary_corrupted = self.readout(z_corrupted, corrupted_data.batch)

        # Discriminator scores
        positive_score = self.discriminator(z, summary)
        negative_score = self.discriminator(z_corrupted, summary)

        return positive_score, negative_score, z

def corrupt_graph(data):
    # Simple feature shuffling corruption
    new_x = data.x[torch.randperm(data.num_nodes)]
    corrupted_data = data.clone()
    corrupted_data.x = new_x
    return corrupted_data