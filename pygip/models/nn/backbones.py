"""Compatible GCN, GAT, and GraphSAGE implementations for both DGL and PyG backends."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """Graph Convolutional Network compatible with both DGL and PyG."""
    
    def __init__(self, in_features=None, out_features=None, hidden_dim=16, dropout=0.5,
                 feature_number=None, label_number=None):
        super(GCN, self).__init__()
        # Support both old (feature_number, label_number) and new (in_features, out_features) API
        if in_features is None and feature_number is not None:
            in_features = feature_number
        if out_features is None and label_number is not None:
            out_features = label_number
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        try:
            # Try DGL first (project primarily uses DGL)
            import dgl.nn.pytorch as dglnn
            self.conv1 = dglnn.GraphConv(in_features, hidden_dim, activation=F.relu)
            self.conv2 = dglnn.GraphConv(hidden_dim, out_features)
            self.backend = 'dgl'
        except ImportError:
            # Fallback to PyG
            from torch_geometric.nn import GCNConv
            self.conv1 = GCNConv(in_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_features)
            self.backend = 'pyg'
    
    def forward(self, g, features):
        """Forward pass.
        
        Args:
            g: Graph (DGLGraph or edge_index wrapper)
            features: Node features [num_nodes, in_features]
        
        Returns:
            logits: [num_nodes, out_features]
        """
        if self.backend == 'pyg':
            # PyG expects edge_index
            if hasattr(g, 'edge_index'):
                edge_index = g.edge_index
            else:
                edge_index = g
            
            x = self.conv1(features, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return x
        else:
            # DGL expects DGLGraph
            x = self.conv1(g, features)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(g, x)
            return x


class GAT(nn.Module):
    """Graph Attention Network compatible with both DGL and PyG."""
    
    def __init__(self, in_features, out_features, hidden_dim=16, num_heads=8, dropout=0.5):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        try:
            # Try DGL first (project primarily uses DGL)
            import dgl.nn.pytorch as dglnn
            self.conv1 = dglnn.GATConv(in_features, hidden_dim, num_heads=num_heads)
            self.conv2 = dglnn.GATConv(hidden_dim * num_heads, out_features, num_heads=1)
            self.backend = 'dgl'
        except ImportError:
            # Fallback to PyG
            from torch_geometric.nn import GATConv
            self.conv1 = GATConv(in_features, hidden_dim, heads=num_heads, dropout=dropout)
            self.conv2 = GATConv(hidden_dim * num_heads, out_features, heads=1, concat=False, dropout=dropout)
            self.backend = 'pyg'
    
    def forward(self, g, features):
        """Forward pass.
        
        Args:
            g: Graph (DGLGraph or edge_index wrapper)
            features: Node features [num_nodes, in_features]
        
        Returns:
            logits: [num_nodes, out_features]
        """
        if self.backend == 'pyg':
            # PyG expects edge_index
            if hasattr(g, 'edge_index'):
                edge_index = g.edge_index
            else:
                edge_index = g
            
            x = F.dropout(features, p=self.dropout, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return x
        else:
            # DGL expects DGLGraph
            x = F.dropout(features, p=self.dropout, training=self.training)
            x = self.conv1(g, x)
            x = x.flatten(1)  # Flatten multi-head attention
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(g, x)
            x = x.mean(1)  # Average over attention heads
            return x


class GraphSAGE(nn.Module):
    """GraphSAGE model using DGL SAGEConv.

    Uses block-based forward for compatibility with NeighborSampler.
    For full-graph inference, pass [graph, graph] as blocks.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        from dgl.nn.pytorch import SAGEConv
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggregator_type='mean')

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = F.relu(x)
        x = self.conv2(blocks[1], x)
        return x


class ShadowNet(nn.Module):
    """A shadow model GCN (DGL-based)."""

    def __init__(self, feature_number, label_number):
        super(ShadowNet, self).__init__()
        from dgl.nn.pytorch import GraphConv
        self.layer1 = GraphConv(feature_number, 16)
        self.layer2 = GraphConv(16, label_number)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x


class AttackNet(nn.Module):
    """An attack model GCN (DGL-based)."""

    def __init__(self, feature_number, label_number):
        super(AttackNet, self).__init__()
        from dgl.nn.pytorch import GraphConv
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(feature_number, 16, activation=F.relu))
        self.layers.append(GraphConv(16, label_number))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, features):
        x = F.relu(self.layers[0](g, features))
        x = self.layers[1](g, x)
        return x


class GCN_PyG(nn.Module):
    """GCN using PyG's GCNConv (used by ImperceptibleWM)."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


def create_model(arch_name, in_features, out_features, **kwargs):
    """Factory function to create a GNN model by architecture name."""
    arch_name = arch_name.lower()
    if arch_name == 'gcn':
        return GCN(in_features, out_features,
                    hidden_dim=kwargs.get('hidden_dim', 16),
                    dropout=kwargs.get('dropout', 0.5))
    elif arch_name == 'gat':
        return GAT(in_features, out_features,
                    hidden_dim=kwargs.get('hidden_dim', 16),
                    num_heads=kwargs.get('num_heads', 8),
                    dropout=kwargs.get('dropout', 0.5))
    elif arch_name == 'graphsage':
        return GraphSAGE(in_features,
                         kwargs.get('hidden_channels', 128),
                         out_features)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")


def model_forward(model, graph, features):
    """Unified forward pass that handles both full-graph and block-based models.

    For GraphSAGE, uses the [graph, graph] trick for full-graph inference
    (valid for 2-layer models).
    For GCN_PyG, extracts edge_index from DGL graph.
    """
    if isinstance(model, GraphSAGE):
        return model([graph, graph], features)
    if isinstance(model, GCN_PyG):
        # GCN_PyG expects (x, edge_index), not (graph, features)
        import dgl
        if isinstance(graph, dgl.DGLGraph):
            src, dst = graph.edges()
            edge_index = torch.stack([src, dst], dim=0).to(features.device)
        else:
            edge_index = graph
        return model(features, edge_index)
    return model(graph, features)