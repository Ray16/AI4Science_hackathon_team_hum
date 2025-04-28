import torch
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter
from torch.nn import Module

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
