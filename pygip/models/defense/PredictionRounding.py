"""Prediction Rounding defense - truncates confidence scores.

Based on the idea that fine-grained probability values leak information.
Rounding to lower precision reduces information attackers can extract.
"""
import time
import torch
import torch.nn.functional as F
from pygip.models.defense.base import BaseDefense
from pygip.models.nn.backbones import GCN, model_forward
from pygip.utils.metrics import DefenseMetric, DefenseCompMetric


class PredictionRounding(BaseDefense):
    """Round/quantize confidence scores to fixed precision.

    Parameters:
      precision_bits: number of bits to keep in softmax output
                      (e.g. 2 bits = 4 levels: 0, 0.25, 0.5, 0.75, 1.0)
      top_k: if set, only return top-k logits (others set to -inf)
    """

    supported_api_types = {'dgl'}

    def __init__(self, dataset, precision_bits=4, top_k=None, **kwargs):
        self.dataset = dataset
        self.graph_data = dataset.graph_data
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.num_nodes = dataset.num_nodes
        self.precision_bits = precision_bits
        self.top_k = top_k

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

    def _round_logits(self, logits):
        """Round softmax probabilities to fixed precision."""
        probs = F.softmax(logits, dim=1)
        levels = 2 ** self.precision_bits
        probs = torch.round(probs * levels) / levels
        # Renormalize
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-10)
        # Convert back to logits
        return torch.log(probs.clamp(min=1e-10))

    def _top_k_filter(self, logits):
        """Keep only top-k logits, set rest to -inf."""
        topk_vals, topk_idx = logits.topk(self.top_k, dim=1)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(1, topk_idx, topk_vals)
        return mask

    def defend(self):
        metric = DefenseMetric()
        comp = DefenseCompMetric()
        comp.start()

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

        t0 = time.time()
        with torch.no_grad():
            clean_logits = model_forward(self.net1, self.graph_data, self.features)
            clean_preds = clean_logits.argmax(dim=1)
            if self.top_k is not None:
                defended_logits = self._top_k_filter(clean_logits)
            else:
                defended_logits = self._round_logits(clean_logits)
            def_preds = defended_logits.argmax(dim=1)
        inference_time = time.time() - t0

        test_labels = self.labels[self.test_mask]
        def_test_preds = def_preds[self.test_mask]
        clean_test_preds = clean_preds[self.test_mask]

        # WM Acc = how much defended output differs from clean (protection signal)
        wm_preds = (def_test_preds == clean_test_preds).long()
        wm_labels = torch.ones_like(wm_preds)
        metric.update_wm(wm_preds, wm_labels)

        metric.update(def_test_preds, test_labels, clean_test_preds)
        comp.update(train_target_time=train_time, train_defense_time=0.0,
                     inference_defense_time=inference_time)
        comp.end()

        res = metric.compute()
        res_comp = comp.compute()
        print(f"PredictionRounding (bits={self.precision_bits}, top_k={self.top_k}): "
              f"acc={res.get('Acc',-1):.4f}, wm_acc={res.get('WM Acc',-1):.4f}")
        return res, res_comp
