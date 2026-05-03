"""Output Perturbation defense - adds Gaussian noise to logits.

Parameters:
  sigma: noise standard deviation, default 0.1
"""
import time
import torch
import torch.nn.functional as F
from pygip.models.defense.base import BaseDefense
from pygip.models.nn.backbones import GCN, model_forward
from pygip.utils.metrics import DefenseMetric, DefenseCompMetric


class OutputPerturbation(BaseDefense):
    """Add Gaussian noise to output logits to hinder extraction.

    This is a non-watermark defense that degrades attack fidelity by
    perturbing the model's output predictions. There is no ownership
    verification - the "WM Acc" is reported as the utility drop caused
    by perturbation (interpreted as "protection strength").
    """

    supported_api_types = {'dgl'}

    def __init__(self, dataset, sigma=0.1, **kwargs):
        self.dataset = dataset
        self.graph_data = dataset.graph_data
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.num_nodes = dataset.num_nodes
        self.sigma = sigma

        import os
        device_str = os.environ.get('PYGIP_DEVICE', None)
        if device_str:
            self.device = torch.device(device_str)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        try:
            self.graph_data = self.graph_data.to(self.device)
        except Exception:
            pass
        self.features = self.graph_data.ndata['feat'].to(self.device)
        self.labels = self.graph_data.ndata['label'].to(self.device)
        self.train_mask = self.graph_data.ndata['train_mask'].bool()
        self.test_mask = self.graph_data.ndata['test_mask'].bool()

    def defend(self):
        """Train a standard GCN then wrap its inference with noise."""
        metric = DefenseMetric()
        comp = DefenseCompMetric()
        comp.start()

        # Train clean model
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)
        opt = torch.optim.Adam(self.net1.parameters(), lr=0.01, weight_decay=5e-4)
        t0 = time.time()
        self.net1.train()
        for _ in range(200):
            opt.zero_grad()
            out = model_forward(self.net1, self.graph_data, self.features)
            F.cross_entropy(out[self.train_mask], self.labels[self.train_mask]).backward()
            opt.step()
        train_time = time.time() - t0
        self.net1.eval()

        # Evaluate defended (noisy) predictions on test set
        t0 = time.time()
        with torch.no_grad():
            clean_logits = model_forward(self.net1, self.graph_data, self.features)
            clean_preds = clean_logits.argmax(dim=1)
            # Apply noise
            noise = torch.randn_like(clean_logits) * self.sigma
            noisy_logits = clean_logits + noise
            noisy_preds = noisy_logits.argmax(dim=1)
        inference_time = time.time() - t0

        # Metrics on test set
        test_labels = self.labels[self.test_mask]
        noisy_test_preds = noisy_preds[self.test_mask]
        clean_test_preds = clean_preds[self.test_mask]

        # WM Acc = agreement between defended (noisy) and clean (strength of perturbation)
        # Higher WM Acc = lower perturbation = weaker defense
        wm_preds = (noisy_test_preds == clean_test_preds).long()
        wm_labels = torch.ones_like(wm_preds)
        metric.update_wm(wm_preds, wm_labels)

        # Attack-level metrics: noisy predictions vs ground truth
        metric.update(noisy_test_preds, test_labels, clean_test_preds)

        comp.update(train_target_time=train_time, train_defense_time=0.0,
                     inference_defense_time=inference_time)
        comp.end()

        res = metric.compute()
        res_comp = comp.compute()
        print(f"OutputPerturbation (sigma={self.sigma}): acc={res.get('Acc',-1):.4f}, "
              f"wm_acc={res.get('WM Acc',-1):.4f}")
        return res, res_comp
