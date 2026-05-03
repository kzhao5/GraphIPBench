"""All link prediction attacks adapted from node classification attacks.

Includes MEA variants 0-5, AdvMEA, CEGA, and DFEA types I-III.
"""

import random
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pygip.models.nn.link_pred import GCNLinkPred


class BaseLinkPredAttack:
    """Base class for link prediction attacks."""
    
    def __init__(self, dataset, victim_model, query_budget, surrogate_hidden=64, device=None):
        self.dataset = dataset
        self.victim_model = victim_model
        self.query_budget = query_budget
        self.surrogate_hidden = surrogate_hidden
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup graph reference
        backend = dataset.api_type
        if backend == 'dgl':
            self.g_ref = dataset.graph_data.to(self.device)
            self.feat = self.g_ref.ndata['feat'].to(self.device)
        else:
            class PseudoGraph:
                def __init__(self, edge_index):
                    self.edge_index = edge_index
            self.g_ref = PseudoGraph(dataset.edge_index.to(self.device))
            self.feat = dataset.features.to(self.device)
    
    def query_victim(self, edges):
        """Query victim model."""
        edges = edges.to(self.device)
        self.victim_model.eval()
        with torch.no_grad():
            logits, _ = self.victim_model(self.g_ref, self.feat, edges)
            preds = (torch.sigmoid(logits) > 0.5).long()
        return preds
    
    def get_query_edges(self):
        """Get edges to query."""
        train_pos = self.dataset.edge_split.train_pos
        train_neg = self.dataset.edge_split.train_neg
        
        total = train_pos.shape[1]
        k = max(10, int(total * self.query_budget))
        perm = torch.randperm(total)[:k]
        
        sub_pos = train_pos[:, perm]
        sub_neg = train_neg[:, perm]
        
        query_edges = torch.cat([sub_pos, sub_neg], dim=1)
        return query_edges
    
    def evaluate(self, surrogate):
        """Evaluate surrogate."""
        test_edges = torch.cat([
            self.dataset.edge_split.test_pos,
            self.dataset.edge_split.test_neg
        ], dim=1).to(self.device)
        
        test_labels = torch.cat([
            torch.ones(self.dataset.edge_split.test_pos.shape[1], dtype=torch.long),
            torch.zeros(self.dataset.edge_split.test_neg.shape[1], dtype=torch.long)
        ]).to(self.device)
        
        with torch.no_grad():
            v_logits, _ = self.victim_model(self.g_ref, self.feat, test_edges)
            v_preds = (torch.sigmoid(v_logits) > 0.5).long()
            
            s_logits, _ = surrogate(self.g_ref, self.feat, test_edges)
            s_preds = (torch.sigmoid(s_logits) > 0.5).long()
        
        acc = (s_preds == test_labels).float().mean().item()
        
        tp = ((s_preds == 1) & (test_labels == 1)).sum().item()
        fp = ((s_preds == 1) & (test_labels == 0)).sum().item()
        fn = ((s_preds == 0) & (test_labels == 1)).sum().item()
        tn = ((s_preds == 0) & (test_labels == 0)).sum().item()
        
        f1_pos = 2 * tp / (2 * tp + fp + fn + 1e-8)
        f1_neg = 2 * tn / (2 * tn + fp + fn + 1e-8)
        f1_macro = 0.5 * (f1_pos + f1_neg)
        
        fidelity = (s_preds == v_preds).float().mean().item()
        
        return acc, f1_macro, fidelity


# ============================================================================
# MEA Family (ModelExtractionAttack0-5)
# ============================================================================

class ModelExtractionAttack0(BaseLinkPredAttack):
    """MEA-0: Basic attack - random edge sampling."""
    
    def attack(self):
        print("  Running MEA-0 (random sampling)...")
        query_edges = self.get_query_edges()
        victim_preds = self.query_victim(query_edges)
        
        surrogate = GCNLinkPred(
            self.dataset.num_features, 
            self.surrogate_hidden
        ).to(self.device)
        
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(150):
            surrogate.train()
            logits, _ = surrogate(self.g_ref, self.feat, query_edges.to(self.device))
            loss = F.binary_cross_entropy_with_logits(logits, victim_preds.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}: loss={loss.item():.4f}")
        
        acc, f1, fidelity = self.evaluate(surrogate)
        return surrogate, acc, f1, fidelity


class ModelExtractionAttack1(BaseLinkPredAttack):
    """MEA-1: Degree-based sampling - prioritize high-degree node edges."""
    
    def attack(self):
        print("  Running MEA-1 (degree-based sampling)...")
        
        # Calculate node degrees
        edge_index = self.dataset.edge_index if hasattr(self.dataset, 'edge_index') else self.dataset.edge_split.train_pos
        degrees = torch.zeros(self.dataset.num_nodes, dtype=torch.long)
        degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))
        degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long))
        
        # Sample edges involving high-degree nodes
        train_pos = self.dataset.edge_split.train_pos
        train_neg = self.dataset.edge_split.train_neg
        
        # Score edges by sum of endpoint degrees
        pos_scores = degrees[train_pos[0]] + degrees[train_pos[1]]
        neg_scores = degrees[train_neg[0]] + degrees[train_neg[1]]
        
        k = max(10, int(train_pos.shape[1] * self.query_budget))
        _, top_pos_idx = torch.topk(pos_scores, min(k, len(pos_scores)))
        _, top_neg_idx = torch.topk(neg_scores, min(k, len(neg_scores)))
        
        query_edges = torch.cat([
            train_pos[:, top_pos_idx],
            train_neg[:, top_neg_idx]
        ], dim=1)
        
        victim_preds = self.query_victim(query_edges)
        
        surrogate = GCNLinkPred(
            self.dataset.num_features,
            self.surrogate_hidden
        ).to(self.device)
        
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(150):
            surrogate.train()
            logits, _ = surrogate(self.g_ref, self.feat, query_edges.to(self.device))
            loss = F.binary_cross_entropy_with_logits(logits, victim_preds.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}: loss={loss.item():.4f}")
        
        acc, f1, fidelity = self.evaluate(surrogate)
        return surrogate, acc, f1, fidelity


class ModelExtractionAttack2(BaseLinkPredAttack):
    """MEA-2: Generate synthetic edges for querying."""
    
    def attack(self):
        print("  Running MEA-2 (synthetic edge generation)...")
        
        # Generate synthetic edges
        num_synthetic = max(100, int(self.dataset.edge_split.train_pos.shape[1] * self.query_budget))
        synthetic_edges = []
        for _ in range(num_synthetic):
            u = random.randint(0, self.dataset.num_nodes - 1)
            v = random.randint(0, self.dataset.num_nodes - 1)
            if u != v:
                synthetic_edges.append([u, v])
        
        synthetic_edges = torch.tensor(synthetic_edges, dtype=torch.long).t()
        victim_preds = self.query_victim(synthetic_edges)
        
        surrogate = GCNLinkPred(
            self.dataset.num_features,
            self.surrogate_hidden
        ).to(self.device)
        
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(150):
            surrogate.train()
            logits, _ = surrogate(self.g_ref, self.feat, synthetic_edges.to(self.device))
            loss = F.binary_cross_entropy_with_logits(logits, victim_preds.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}: loss={loss.item():.4f}")
        
        acc, f1, fidelity = self.evaluate(surrogate)
        return surrogate, acc, f1, fidelity


class ModelExtractionAttack3(BaseLinkPredAttack):
    """MEA-3: Community-based sampling - sample edges within communities."""
    
    def attack(self):
        print("  Running MEA-3 (community-based sampling)...")
        
        # Simple community detection: sample connected components
        train_pos = self.dataset.edge_split.train_pos
        train_neg = self.dataset.edge_split.train_neg
        
        # Sample a local subgraph
        center_nodes = random.sample(range(self.dataset.num_nodes), min(50, self.dataset.num_nodes))
        
        # Find edges involving center nodes
        mask_pos = torch.zeros(train_pos.shape[1], dtype=torch.bool)
        mask_neg = torch.zeros(train_neg.shape[1], dtype=torch.bool)
        
        for node in center_nodes:
            mask_pos |= (train_pos[0] == node) | (train_pos[1] == node)
            mask_neg |= (train_neg[0] == node) | (train_neg[1] == node)
        
        k = max(10, int(train_pos.shape[1] * self.query_budget))
        pos_indices = torch.where(mask_pos)[0]
        neg_indices = torch.where(mask_neg)[0]
        
        if len(pos_indices) > k:
            pos_indices = pos_indices[torch.randperm(len(pos_indices))[:k]]
        if len(neg_indices) > k:
            neg_indices = neg_indices[torch.randperm(len(neg_indices))[:k]]
        
        query_edges = torch.cat([
            train_pos[:, pos_indices] if len(pos_indices) > 0 else train_pos[:, :0],
            train_neg[:, neg_indices] if len(neg_indices) > 0 else train_neg[:, :0]
        ], dim=1)
        
        if query_edges.shape[1] == 0:
            query_edges = self.get_query_edges()
        
        victim_preds = self.query_victim(query_edges)
        
        surrogate = GCNLinkPred(
            self.dataset.num_features,
            self.surrogate_hidden
        ).to(self.device)
        
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(150):
            surrogate.train()
            logits, _ = surrogate(self.g_ref, self.feat, query_edges.to(self.device))
            loss = F.binary_cross_entropy_with_logits(logits, victim_preds.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}: loss={loss.item():.4f}")
        
        acc, f1, fidelity = self.evaluate(surrogate)
        return surrogate, acc, f1, fidelity


class ModelExtractionAttack4(BaseLinkPredAttack):
    """MEA-4: Feature similarity-based edge sampling."""
    
    def attack(self):
        print("  Running MEA-4 (feature similarity-based)...")
        
        # Compute node feature similarities
        feat_norm = F.normalize(self.feat, p=2, dim=1)
        
        # Sample node pairs with high/low feature similarity
        num_samples = max(50, int(self.dataset.num_nodes * 0.05))
        sample_nodes = random.sample(range(self.dataset.num_nodes), min(num_samples, self.dataset.num_nodes))
        
        synthetic_edges = []
        for u in sample_nodes:
            # Compute similarity to all other nodes
            sims = torch.mm(feat_norm[u:u+1], feat_norm.t()).squeeze()
            
            # Sample a high similarity and a low similarity neighbor
            _, top_idx = torch.topk(sims, k=min(10, len(sims)))
            _, bottom_idx = torch.topk(sims, k=min(10, len(sims)), largest=False)
            
            for v in top_idx.tolist():
                if v != u:
                    synthetic_edges.append([u, v])
                    break
            
            for v in bottom_idx.tolist():
                if v != u:
                    synthetic_edges.append([u, v])
                    break
        
        k = max(10, int(self.dataset.edge_split.train_pos.shape[1] * self.query_budget))
        if len(synthetic_edges) > k:
            synthetic_edges = random.sample(synthetic_edges, k)
        
        if len(synthetic_edges) == 0:
            query_edges = self.get_query_edges()
        else:
            query_edges = torch.tensor(synthetic_edges, dtype=torch.long).t().to(self.device)
        
        victim_preds = self.query_victim(query_edges)
        
        surrogate = GCNLinkPred(
            self.dataset.num_features,
            self.surrogate_hidden
        ).to(self.device)
        
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(150):
            surrogate.train()
            logits, _ = surrogate(self.g_ref, self.feat, query_edges.to(self.device))
            loss = F.binary_cross_entropy_with_logits(logits, victim_preds.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}: loss={loss.item():.4f}")
        
        acc, f1, fidelity = self.evaluate(surrogate)
        return surrogate, acc, f1, fidelity


class ModelExtractionAttack5(BaseLinkPredAttack):
    """MEA-5: Hybrid approach combining multiple strategies."""
    
    def attack(self):
        print("  Running MEA-5 (hybrid approach)...")
        
        # Combine random, degree-based, and synthetic edges
        k = max(10, int(self.dataset.edge_split.train_pos.shape[1] * self.query_budget))
        k_each = k // 3
        
        # 1. Random edges
        train_pos = self.dataset.edge_split.train_pos
        train_neg = self.dataset.edge_split.train_neg
        perm = torch.randperm(train_pos.shape[1])[:k_each]
        random_edges = torch.cat([train_pos[:, perm], train_neg[:, perm]], dim=1)
        
        # 2. High-degree edges
        edge_index = train_pos
        degrees = torch.zeros(self.dataset.num_nodes, dtype=torch.long)
        degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))
        degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long))
        
        pos_scores = degrees[train_pos[0]] + degrees[train_pos[1]]
        _, top_idx = torch.topk(pos_scores, min(k_each, len(pos_scores)))
        degree_edges = train_pos[:, top_idx]
        
        # 3. Synthetic edges
        synthetic_edges = []
        for _ in range(k_each):
            u = random.randint(0, self.dataset.num_nodes - 1)
            v = random.randint(0, self.dataset.num_nodes - 1)
            if u != v:
                synthetic_edges.append([u, v])
        synthetic_edges = torch.tensor(synthetic_edges, dtype=torch.long).t() if synthetic_edges else torch.empty((2, 0), dtype=torch.long)
        
        # Combine all
        query_edges = torch.cat([random_edges, degree_edges, synthetic_edges], dim=1)
        
        victim_preds = self.query_victim(query_edges)
        
        surrogate = GCNLinkPred(
            self.dataset.num_features,
            self.surrogate_hidden
        ).to(self.device)
        
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(150):
            surrogate.train()
            logits, _ = surrogate(self.g_ref, self.feat, query_edges.to(self.device))
            loss = F.binary_cross_entropy_with_logits(logits, victim_preds.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}: loss={loss.item():.4f}")
        
        acc, f1, fidelity = self.evaluate(surrogate)
        return surrogate, acc, f1, fidelity


# ============================================================================
# Other Attacks
# ============================================================================

class MEALinkPred(ModelExtractionAttack0):
    """Alias for MEA-0."""
    pass


class AdvMEALinkPred(BaseLinkPredAttack):
    """Advanced MEA - adds adversarial edge sampling."""
    
    def attack(self):
        print("  Running AdvMEA (adversarial sampling)...")
        
        # Start with standard queries
        query_edges = self.get_query_edges()
        victim_preds = self.query_victim(query_edges)
        
        # Add adversarial samples: edges with uncertain predictions
        num_adversarial = int(query_edges.shape[1] * 0.2)
        adv_edges = []
        for _ in range(num_adversarial):
            u = random.randint(0, self.dataset.num_nodes - 1)
            v = random.randint(0, self.dataset.num_nodes - 1)
            if u != v:
                adv_edges.append([u, v])
        
        if adv_edges:
            adv_edges_tensor = torch.tensor(adv_edges, dtype=torch.long).t()
            adv_preds = self.query_victim(adv_edges_tensor)
            
            query_edges = torch.cat([query_edges, adv_edges_tensor], dim=1)
            victim_preds = torch.cat([victim_preds, adv_preds])
        
        # Train surrogate
        surrogate = GCNLinkPred(
            self.dataset.num_features,
            self.surrogate_hidden
        ).to(self.device)
        
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(150):
            surrogate.train()
            logits, _ = surrogate(self.g_ref, self.feat, query_edges.to(self.device))
            loss = F.binary_cross_entropy_with_logits(logits, victim_preds.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}: loss={loss.item():.4f}")
        
        acc, f1, fidelity = self.evaluate(surrogate)
        return surrogate, acc, f1, fidelity


class CEGALinkPred(BaseLinkPredAttack):
    """CEGA - uses confidence-based edge selection."""
    
    def attack(self):
        print("  Running CEGA (confidence-based)...")
        
        # Query more edges initially
        train_pos = self.dataset.edge_split.train_pos
        train_neg = self.dataset.edge_split.train_neg
        
        total = train_pos.shape[1]
        k_initial = max(10, int(total * self.query_budget * 1.5))
        perm = torch.randperm(total)[:k_initial]
        
        initial_pos = train_pos[:, perm]
        initial_neg = train_neg[:, perm]
        initial_edges = torch.cat([initial_pos, initial_neg], dim=1)
        
        # Get victim predictions with confidence
        initial_edges_dev = initial_edges.to(self.device)
        self.victim_model.eval()
        with torch.no_grad():
            logits, _ = self.victim_model(self.g_ref, self.feat, initial_edges_dev)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            
            # Select edges with high confidence
            confidence = torch.abs(probs - 0.5)
            k_final = int(initial_edges.shape[1] * 2 / 3)
            _, top_indices = torch.topk(confidence, k_final)
            
            # Fix: move top_indices to CPU for indexing CPU tensor
            query_edges = initial_edges[:, top_indices.cpu()]
            victim_preds = preds[top_indices]
        
        # Train surrogate
        surrogate = GCNLinkPred(
            self.dataset.num_features,
            self.surrogate_hidden
        ).to(self.device)
        
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(150):
            surrogate.train()
            logits, _ = surrogate(self.g_ref, self.feat, query_edges.to(self.device))
            loss = F.binary_cross_entropy_with_logits(logits, victim_preds.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}: loss={loss.item():.4f}")
        
        acc, f1, fidelity = self.evaluate(surrogate)
        return surrogate, acc, f1, fidelity


class DFEATypeLinkPred(BaseLinkPredAttack):
    """Data-Free Extraction Attack - generates synthetic edges."""
    
    def __init__(self, dataset, victim_model, query_budget, variant=1, surrogate_hidden=64, device=None):
        super().__init__(dataset, victim_model, query_budget, surrogate_hidden, device)
        self.variant = variant
    
    def generate_synthetic_edges(self, num_edges):
        """Generate random edges for querying."""
        edges = []
        for _ in range(num_edges):
            u = random.randint(0, self.dataset.num_nodes - 1)
            v = random.randint(0, self.dataset.num_nodes - 1)
            if u != v:
                edges.append([u, v])
        return torch.tensor(edges, dtype=torch.long).t()
    
    def attack(self):
        print(f"  Running DFEA-Type{self.variant}...")
        
        # Generate synthetic edges
        train_size = self.dataset.edge_split.train_pos.shape[1]
        num_synthetic = max(100, int(train_size * self.query_budget))
        synthetic_edges = self.generate_synthetic_edges(num_synthetic)
        
        # Query victim on synthetic edges
        victim_preds = self.query_victim(synthetic_edges)
        
        # Train surrogate based on variant
        surrogate = GCNLinkPred(
            self.dataset.num_features,
            self.surrogate_hidden
        ).to(self.device)
        
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(150):
            surrogate.train()
            
            if self.variant == 1:  # Type I: Hard labels
                logits, _ = surrogate(self.g_ref, self.feat, synthetic_edges.to(self.device))
                loss = F.binary_cross_entropy_with_logits(logits, victim_preds.float())
            
            elif self.variant == 2:  # Type II: Label smoothing
                logits, _ = surrogate(self.g_ref, self.feat, synthetic_edges.to(self.device))
                targets = victim_preds.float() * 0.9 + 0.05
                loss = F.binary_cross_entropy_with_logits(logits, targets)
            
            else:  # Type III: Consistency regularization
                logits1, _ = surrogate(self.g_ref, self.feat, synthetic_edges.to(self.device))
                synthetic_edges2 = self.generate_synthetic_edges(num_synthetic // 2)
                logits2, _ = surrogate(self.g_ref, self.feat, synthetic_edges2.to(self.device))
                
                loss_sup = F.binary_cross_entropy_with_logits(logits1, victim_preds.float())
                loss_cons = torch.var(torch.sigmoid(logits2))
                loss = loss_sup + 0.1 * loss_cons
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}: loss={loss.item():.4f}")
        
        acc, f1, fidelity = self.evaluate(surrogate)
        return surrogate, acc, f1, fidelity


# Convenience aliases
class DFEATypeILinkPred(DFEATypeLinkPred):
    def __init__(self, dataset, victim_model, query_budget, surrogate_hidden=64, device=None):
        super().__init__(dataset, victim_model, query_budget, variant=1, surrogate_hidden=surrogate_hidden, device=device)

class DFEATypeIILinkPred(DFEATypeLinkPred):
    def __init__(self, dataset, victim_model, query_budget, surrogate_hidden=64, device=None):
        super().__init__(dataset, victim_model, query_budget, variant=2, surrogate_hidden=surrogate_hidden, device=device)

class DFEATypeIIILinkPred(DFEATypeLinkPred):
    def __init__(self, dataset, victim_model, query_budget, surrogate_hidden=64, device=None):
        super().__init__(dataset, victim_model, query_budget, variant=3, surrogate_hidden=surrogate_hidden, device=device)