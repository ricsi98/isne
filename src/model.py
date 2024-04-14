import lightning.pytorch as pl
import torch
from torch_geometric.loader import NeighborSampler

from layer import ISNELayer


class Node2VecBase(pl.LightningModule):
    EPS = 1e-15

    def forward(self, node_ids: torch.Tensor):
        raise NotImplementedError

    @torch.jit.export
    def loss(self, pos_rw: torch.Tensor, neg_rw: torch.Tensor) -> torch.Tensor:
        r"""Computes the loss given positive and negative random walks."""
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self(start).view(pos_rw.size(0), 1, self.encoder.embedding_dim)
        h_rest = self(rest.view(-1)).view(pos_rw.size(0), -1, self.encoder.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self(start).view(neg_rw.size(0), 1, self.encoder.embedding_dim)
        h_rest = self(rest.view(-1)).view(neg_rw.size(0), -1, self.encoder.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss

    def training_step(self, batch, idx):
        self.encoder.train()
        pos_rw, neg_rw = list(map(lambda x: x.to(self.device), batch))
        loss = self.loss(pos_rw, neg_rw)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SparseAdam(self.encoder.parameters(), **self.optim_kwargs)


class ISNE(Node2VecBase):

    def __init__(self, num_nodes: int, hidden_channels: int, edge_index: torch.torch.Tensor, *args, **kwargs):
        super().__init__()
        self.num_nodes = num_nodes
        self.encoder = ISNELayer(num_nodes, hidden_channels, *args, **kwargs)
        self.sampler = NeighborSampler(edge_index.cpu(), [-1], return_e_id=False)

    def forward(self, node_ids: torch.torch.Tensor):
        with torch.no_grad():
            _, idxes, ei = self.sampler.sample(node_ids.cpu())
            idxes = idxes.to(self.device)
            ei = ei.to(self.device)
        return self.encoder(idxes, ei.edge_index)

    @torch.no_grad()
    def embed_nodes(self, edge_index: torch.Tensor):
        node_ids = torch.arange(self.num_nodes)
        return self.encoder(node_ids, edge_index)

    def configure_optimizers(self):
        return torch.optim.Adam(self.encoder.parameters(), lr=0.01)
