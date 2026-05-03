"""All link prediction defenses adapted from node classification defenses."""

import random
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pygip.models.nn.link_pred import GCNLinkPred


class BaseLinkPredDefense:
    """Base class for link prediction defenses."""
    
    def __init__(self, dataset, victim_model, defense_ratio=0.01, device=None):
        self.dataset = dataset
        self.victim_model = victim_model
        self.defense_ratio = defense_ratio
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
    
    def evaluate(self, model):
        """Evaluate defended model."""
        test_edges = torch.cat([
            self.dataset.edge_split.test_pos,
            self.dataset.edge_split.test_neg
        ], dim=1).to(self.device)
        
        test_labels = torch.cat([
            torch.ones(self.dataset.edge_split.test_pos.shape[1], dtype=torch.long),
            torch.zeros(self.dataset.edge_split.test_neg.shape[1], dtype=torch.long)
        ]).to(self.device)
        
        with torch.no_grad():
            # Victim predictions
            v_logits, _ = self.victim_model(self.g_ref, self.feat, test_edges)
            v_preds = (torch.sigmoid(v_logits) > 0.5).long()
            
            # Defended model predictions
            d_logits, _ = model(self.g_ref, self.feat, test_edges)
            d_preds = (torch.sigmoid(d_logits) > 0.5).long()
        
        acc = (d_preds == test_labels).float().mean().item()
        
        tp = ((d_preds == 1) & (test_labels == 1)).sum().item()
        fp = ((d_preds == 1) & (test_labels == 0)).sum().item()
        fn = ((d_preds == 0) & (test_labels == 1)).sum().item()
        tn = ((d_preds == 0) & (test_labels == 0)).sum().item()
        
        f1_pos = 2 * tp / (2 * tp + fp + fn + 1e-8)
        f1_neg = 2 * tn / (2 * tn + fp + fn + 1e-8)
        f1_macro = 0.5 * (f1_pos + f1_neg)
        
        fidelity = (d_preds == v_preds).float().mean().item()
        
        return acc, f1_macro, fidelity


class RandomWMLinkPred(BaseLinkPredDefense):
    """Random Watermark for Link Prediction.
    
    Adds random synthetic edges to the training set as watermarks.
    """
    
    def defend(self):
        print("Applying RandomWM defense...")
        
        # Clone training data
        train_pos = self.dataset.edge_split.train_pos.clone()
        train_neg = self.dataset.edge_split.train_neg.clone()
        
        # Generate random watermark edges
        num_watermark = max(10, int(train_pos.shape[1] * self.defense_ratio))
        watermark_edges = []
        for _ in range(num_watermark):
            u = random.randint(0, self.dataset.num_nodes - 1)
            v = random.randint(0, self.dataset.num_nodes - 1)
            if u != v:
                watermark_edges.append([min(u, v), max(u, v)])
        
        watermark_edges = torch.tensor(watermark_edges, dtype=torch.long).t()
        watermark_labels = torch.ones(watermark_edges.shape[1], dtype=torch.long)
        
        # Add watermarks to training set
        train_pos_wm = torch.cat([train_pos, watermark_edges], dim=1)
        
        # Train defended model
        defended = GCNLinkPred(
            self.dataset.num_features, 
            64, 
            backend=self.dataset.api_type
        ).to(self.device)
        
        optimizer = torch.optim.Adam(defended.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(100):
            defended.train()
            
            # Combine all training edges
            all_edges = torch.cat([train_pos_wm, train_neg], dim=1)
            all_labels = torch.cat([
                torch.ones(train_pos_wm.shape[1], dtype=torch.long),
                torch.zeros(train_neg.shape[1], dtype=torch.long)
            ]).to(self.device)
            
            # Shuffle
            perm = torch.randperm(all_labels.shape[0])
            all_edges = all_edges[:, perm]
            all_labels = all_labels[perm]
            
            # Train
            logits, _ = defended(self.g_ref, self.feat, all_edges.to(self.device))
            loss = F.binary_cross_entropy_with_logits(logits, all_labels.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Verify watermark
        with torch.no_grad():
            wm_logits, _ = defended(self.g_ref, self.feat, watermark_edges.to(self.device))
            wm_preds = (torch.sigmoid(wm_logits) > 0.5).long()
            wm_acc = (wm_preds == watermark_labels.to(self.device)).float().mean().item()
        
        acc, f1, fidelity = self.evaluate(defended)
        return defended, acc, f1, fidelity, wm_acc


class BackdoorWMLinkPred(BaseLinkPredDefense):
    """Backdoor Watermark for Link Prediction.
    
    Creates a clique of nodes and marks their edges as watermarks.
    """
    
    def defend(self):
        print("Applying BackdoorWM defense...")
        
        train_pos = self.dataset.edge_split.train_pos.clone()
        train_neg = self.dataset.edge_split.train_neg.clone()
        
        # Create a small clique as backdoor trigger
        clique_size = 8
        trigger_nodes = random.sample(range(self.dataset.num_nodes), clique_size)
        
        # All edges in clique
        watermark_edges = []
        for i in range(len(trigger_nodes)):
            for j in range(i + 1, len(trigger_nodes)):
                watermark_edges.append([trigger_nodes[i], trigger_nodes[j]])
        
        watermark_edges = torch.tensor(watermark_edges, dtype=torch.long).t()
        watermark_labels = torch.ones(watermark_edges.shape[1], dtype=torch.long)
        
        # Add to training
        train_pos_wm = torch.cat([train_pos, watermark_edges], dim=1)
        
        # Train defended model
        defended = GCNLinkPred(
            self.dataset.num_features,
            64,
            backend=self.dataset.api_type
        ).to(self.device)
        
        optimizer = torch.optim.Adam(defended.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(100):
            defended.train()
            
            all_edges = torch.cat([train_pos_wm, train_neg], dim=1)
            all_labels = torch.cat([
                torch.ones(train_pos_wm.shape[1], dtype=torch.long),
                torch.zeros(train_neg.shape[1], dtype=torch.long)
            ]).to(self.device)
            
            perm = torch.randperm(all_labels.shape[0])
            
            logits, _ = defended(self.g_ref, self.feat, all_edges[:, perm].to(self.device))
            loss = F.binary_cross_entropy_with_logits(logits, all_labels[perm].float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Verify watermark
        with torch.no_grad():
            wm_logits, _ = defended(self.g_ref, self.feat, watermark_edges.to(self.device))
            wm_preds = (torch.sigmoid(wm_logits) > 0.5).long()
            wm_acc = (wm_preds == watermark_labels.to(self.device)).float().mean().item()
        
        acc, f1, fidelity = self.evaluate(defended)
        return defended, acc, f1, fidelity, wm_acc


class SurviveWMLinkPred(BaseLinkPredDefense):
    """Survival Watermark for Link Prediction.
    
    Removes some training edges but marks specific edges as watermarks that should survive.
    """
    
    def defend(self):
        print("Applying SurviveWM defense...")
        
        train_pos = self.dataset.edge_split.train_pos.clone()
        train_neg = self.dataset.edge_split.train_neg.clone()
        
        # Keep only 90% of training positives (simulating pruning)
        keep_ratio = 0.9
        num_keep = int(train_pos.shape[1] * keep_ratio)
        perm = torch.randperm(train_pos.shape[1])
        train_pos_pruned = train_pos[:, perm[:num_keep]]
        
        # Mark first few as watermarks (these should "survive")
        num_watermark = max(10, int(train_pos.shape[1] * self.defense_ratio))
        watermark_edges = train_pos_pruned[:, :num_watermark]
        watermark_labels = torch.ones(num_watermark, dtype=torch.long)
        
        # Train defended model
        defended = GCNLinkPred(
            self.dataset.num_features,
            64,
            backend=self.dataset.api_type
        ).to(self.device)
        
        optimizer = torch.optim.Adam(defended.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(100):
            defended.train()
            
            all_edges = torch.cat([train_pos_pruned, train_neg], dim=1)
            all_labels = torch.cat([
                torch.ones(train_pos_pruned.shape[1], dtype=torch.long),
                torch.zeros(train_neg.shape[1], dtype=torch.long)
            ]).to(self.device)
            
            perm = torch.randperm(all_labels.shape[0])
            
            logits, _ = defended(self.g_ref, self.feat, all_edges[:, perm].to(self.device))
            loss = F.binary_cross_entropy_with_logits(logits, all_labels[perm].float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Verify watermark
        with torch.no_grad():
            wm_logits, _ = defended(self.g_ref, self.feat, watermark_edges.to(self.device))
            wm_preds = (torch.sigmoid(wm_logits) > 0.5).long()
            wm_acc = (wm_preds == watermark_labels.to(self.device)).float().mean().item()
        
        acc, f1, fidelity = self.evaluate(defended)
        return defended, acc, f1, fidelity, wm_acc


class ImperceptibleWMLinkPred(BaseLinkPredDefense):
    """Imperceptible Watermark for Link Prediction.
    
    Adds very few watermark edges to make them hard to detect.
    """
    
    def defend(self):
        print("Applying ImperceptibleWM defense...")
        
        train_pos = self.dataset.edge_split.train_pos.clone()
        train_neg = self.dataset.edge_split.train_neg.clone()
        
        # Add only 2-3 watermark edges (imperceptible)
        num_watermark = 2
        watermark_edges = []
        for _ in range(num_watermark):
            u = random.randint(0, self.dataset.num_nodes - 1)
            v = random.randint(0, self.dataset.num_nodes - 1)
            if u != v:
                watermark_edges.append([min(u, v), max(u, v)])
        
        if not watermark_edges:
            watermark_edges = [[0, 1]]  # fallback
        
        watermark_edges = torch.tensor(watermark_edges, dtype=torch.long).t()
        watermark_labels = torch.ones(watermark_edges.shape[1], dtype=torch.long)
        
        # Add to training
        train_pos_wm = torch.cat([train_pos, watermark_edges], dim=1)
        
        # Train defended model
        defended = GCNLinkPred(
            self.dataset.num_features,
            64,
            backend=self.dataset.api_type
        ).to(self.device)
        
        optimizer = torch.optim.Adam(defended.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(100):
            defended.train()
            
            all_edges = torch.cat([train_pos_wm, train_neg], dim=1)
            all_labels = torch.cat([
                torch.ones(train_pos_wm.shape[1], dtype=torch.long),
                torch.zeros(train_neg.shape[1], dtype=torch.long)
            ]).to(self.device)
            
            perm = torch.randperm(all_labels.shape[0])
            
            logits, _ = defended(self.g_ref, self.feat, all_edges[:, perm].to(self.device))
            loss = F.binary_cross_entropy_with_logits(logits, all_labels[perm].float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Verify watermark
        with torch.no_grad():
            wm_logits, _ = defended(self.g_ref, self.feat, watermark_edges.to(self.device))
            wm_preds = (torch.sigmoid(wm_logits) > 0.5).long()
            wm_acc = (wm_preds == watermark_labels.to(self.device)).float().mean().item()
        
        acc, f1, fidelity = self.evaluate(defended)
        return defended, acc, f1, fidelity, wm_acc


class IntegrityLinkPred(BaseLinkPredDefense):
    """Integrity Defense for Link Prediction.
    
    Focuses on maintaining the integrity of positive edges by removing some negatives.
    """
    
    def defend(self):
        print("Applying Integrity defense...")
        
        train_pos = self.dataset.edge_split.train_pos.clone()
        train_neg = self.dataset.edge_split.train_neg.clone()
        
        # Remove 5% of negative samples to focus on positive integrity
        keep_ratio = 0.95
        num_keep = int(train_neg.shape[1] * keep_ratio)
        perm = torch.randperm(train_neg.shape[1])
        train_neg_reduced = train_neg[:, perm[:num_keep]]
        
        # Mark some positive edges as watermarks
        num_watermark = max(10, int(train_pos.shape[1] * self.defense_ratio))
        watermark_edges = train_pos[:, -num_watermark:]
        watermark_labels = torch.ones(num_watermark, dtype=torch.long)
        
        # Train defended model
        defended = GCNLinkPred(
            self.dataset.num_features,
            64,
            backend=self.dataset.api_type
        ).to(self.device)
        
        optimizer = torch.optim.Adam(defended.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(100):
            defended.train()
            
            all_edges = torch.cat([train_pos, train_neg_reduced], dim=1)
            all_labels = torch.cat([
                torch.ones(train_pos.shape[1], dtype=torch.long),
                torch.zeros(train_neg_reduced.shape[1], dtype=torch.long)
            ]).to(self.device)
            
            perm = torch.randperm(all_labels.shape[0])
            
            logits, _ = defended(self.g_ref, self.feat, all_edges[:, perm].to(self.device))
            loss = F.binary_cross_entropy_with_logits(logits, all_labels[perm].float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Verify watermark
        with torch.no_grad():
            wm_logits, _ = defended(self.g_ref, self.feat, watermark_edges.to(self.device))
            wm_preds = (torch.sigmoid(wm_logits) > 0.5).long()
            wm_acc = (wm_preds == watermark_labels.to(self.device)).float().mean().item()
        
        acc, f1, fidelity = self.evaluate(defended)
        return defended, acc, f1, fidelity, wm_acc