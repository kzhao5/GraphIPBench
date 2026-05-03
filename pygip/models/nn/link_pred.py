import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


class GCNEmbedding(nn.Module):
    """Two-layer GCN for node embeddings."""

    def __init__(self, in_feats: int, hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden, activation=F.relu)
        self.conv2 = GraphConv(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        h = self.conv1(g, x)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h  # (N, hidden)


class LinkPredictor(nn.Module):
    """Edge scoring head using dot-product or MLP."""

    def __init__(self, hidden: int, method: str = "dot"):
        super().__init__()
        self.method = method
        if method == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )

    def forward(self, emb: torch.Tensor, edges: torch.Tensor):
        # edges shape [2, B]
        src, dst = edges[0], edges[1]
        if self.method == "dot":
            score = (emb[src] * emb[dst]).sum(dim=-1, keepdim=True)
        else:
            h = torch.cat([emb[src], emb[dst]], dim=-1)
            score = self.mlp(h)
        return score.squeeze(-1)  # (B,)


class GCNLinkPred(nn.Module):
    """Composite victim model: GCN embedding + link predictor."""

    def __init__(self, in_feats: int, hidden: int = 64, predictor: str = "dot"):
        super().__init__()
        self.gcn = GCNEmbedding(in_feats, hidden)
        self.pred = LinkPredictor(hidden, predictor)

    def forward(self, g, x, edges):
        emb = self.gcn(g, x)
        return self.pred(emb, edges), emb
