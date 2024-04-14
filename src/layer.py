import torch.nn as nn
from torch_scatter import scatter_mean


class ISNELayer(nn.Module):

    def __init__(self, num_nodes: int, hidden_channels: int, *args, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, hidden_channels, *args, **kwargs)

    def forward(self, node_ids, edge_index):
        sources = node_ids[edge_index[0]]
        vs = self.emb(sources)
        index = edge_index[1]
        return scatter_mean(vs, index, dim=0)

    @property
    def embedding_dim(self):
        return self.emb.embedding_dim
