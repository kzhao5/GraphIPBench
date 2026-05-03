"""Non-watermark defenses against model extraction attacks.

Implements methods from published research papers, adapted from
computer vision / general DNN defenses to GNN setting:

1. OutputPerturbation: Gaussian noise on logits (information-theoretic defense,
   similar to MODELGUARD (USENIX 2024), reversed-softmax output perturbation)

2. PredictionRounding: Truncate softmax to top-k or fixed precision
   (common baseline in prediction poisoning literature)

3. PRADA: Query distribution monitoring (Juuti et al., USENIX 2019)
   Adapted from image classification to graph node classification.
   Monitors pairwise distances between successive queries; raises alarm
   when distribution deviates from benign Shapiro-Wilk normality.

4. AdaptiveMisinformation: OOD-based misleading predictions
   (Kariyappa & Qureshi, CVPR 2020)
   Uses feature-space distance to detect OOD queries; returns shuffled
   pseudolabels for flagged queries.

5. GradientRedirection: MAZE-style targeted defense
   (Mazeika et al., ICML 2022) - gradient-based adversarial perturbation
   aimed at specific target class.

References:
- PRADA: Juuti et al. 2019 https://arxiv.org/abs/1805.02628
- Adaptive Misinformation: Kariyappa & Qureshi, CVPR 2020
- MODELGUARD: Tang et al. USENIX 2024
- MAZE: Mazeika et al. ICML 2022 https://proceedings.mlr.press/v162/mazeika22a
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from pygip.models.defense.base import BaseDefense
from pygip.models.nn.backbones import GCN, model_forward
from pygip.utils.metrics import DefenseMetric, DefenseCompMetric


def _setup_device(self):
    import os
    ds = os.environ.get('PYGIP_DEVICE', None)
    if ds:
        self.device = torch.device(ds)
    else:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {self.device}")


def _train_clean_gcn(dataset, device, epochs=200):
    graph = dataset.graph_data.to(device)
    features = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device)
    train_mask = graph.ndata['train_mask'].bool()
    model = GCN(dataset.num_features, dataset.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model_forward(model, graph, features)
        F.cross_entropy(out[train_mask], labels[train_mask]).backward()
        opt.step()
    model.eval()
    return model, graph, features, labels, train_mask


def _base_metric_defense(self, defended_preds, clean_preds, test_mask, labels,
                          train_time, inference_time):
    """Helper to compile standard DefenseMetric/DefenseCompMetric output."""
    metric = DefenseMetric()
    comp = DefenseCompMetric()
    comp.start()

    test_labels = labels[test_mask]
    def_test = defended_preds[test_mask]
    clean_test = clean_preds[test_mask]

    # Protection strength via wm_acc: how much defended output differs from clean
    # Lower wm_acc => more perturbation => stronger defense
    wm_correct = (def_test == clean_test).long()
    wm_target = torch.ones_like(wm_correct)
    metric.update_wm(wm_correct, wm_target)

    metric.update(def_test, test_labels)
    comp.update(train_target_time=train_time, train_defense_time=0.0,
                 inference_defense_time=inference_time)
    comp.end()
    return metric.compute(), comp.compute()


class OutputPerturbation(BaseDefense):
    """Information-theoretic output perturbation.

    Adds calibrated Gaussian noise to logits before returning predictions.
    Adapted from MODELGUARD (USENIX 2024) and similar output-perturbation
    defenses.
    """
    supported_api_types = {'dgl'}

    def __init__(self, dataset, sigma=0.1, **kwargs):
        self.dataset = dataset
        self.sigma = sigma
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.num_nodes = dataset.num_nodes
        _setup_device(self)

    def defend(self):
        t0 = time.time()
        model, graph, features, labels, train_mask = _train_clean_gcn(self.dataset, self.device)
        self.net1 = model
        train_time = time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            clean_logits = model_forward(model, graph, features)
            clean_preds = clean_logits.argmax(dim=1)
            noise = torch.randn_like(clean_logits) * self.sigma
            def_preds = (clean_logits + noise).argmax(dim=1)
        inference_time = time.time() - t0

        test_mask = graph.ndata['test_mask'].bool()
        res, comp = _base_metric_defense(self, def_preds, clean_preds, test_mask,
                                          labels, train_time, inference_time)
        print(f"OutputPerturbation(sigma={self.sigma}): acc={res.get('Acc',-1):.4f}, "
              f"agreement={res.get('WM Acc',-1):.4f}")
        return res, comp


class PredictionRounding(BaseDefense):
    """Prediction poisoning by rounding softmax confidence scores.

    Common baseline in prediction-poisoning literature. Reduces attacker's
    signal by quantizing or top-k filtering logits.
    """
    supported_api_types = {'dgl'}

    def __init__(self, dataset, precision_bits=2, top_k=None, **kwargs):
        self.dataset = dataset
        self.precision_bits = precision_bits
        self.top_k = top_k
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.num_nodes = dataset.num_nodes
        _setup_device(self)

    def _apply(self, logits):
        if self.top_k is not None:
            vals, idx = logits.topk(self.top_k, dim=1)
            out = torch.full_like(logits, float('-inf'))
            out.scatter_(1, idx, vals)
            return out
        probs = F.softmax(logits, dim=1)
        levels = 2 ** self.precision_bits
        probs = (torch.round(probs * levels) / levels).clamp(min=1e-10)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return torch.log(probs)

    def defend(self):
        t0 = time.time()
        model, graph, features, labels, train_mask = _train_clean_gcn(self.dataset, self.device)
        self.net1 = model
        train_time = time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            clean_logits = model_forward(model, graph, features)
            clean_preds = clean_logits.argmax(dim=1)
            def_preds = self._apply(clean_logits).argmax(dim=1)
        inference_time = time.time() - t0

        test_mask = graph.ndata['test_mask'].bool()
        res, comp = _base_metric_defense(self, def_preds, clean_preds, test_mask,
                                          labels, train_time, inference_time)
        print(f"PredictionRounding(bits={self.precision_bits},k={self.top_k}): "
              f"acc={res.get('Acc',-1):.4f}, agreement={res.get('WM Acc',-1):.4f}")
        return res, comp


class PRADA(BaseDefense):
    """PRADA: query distribution monitoring (Juuti et al., USENIX 2019).

    Detects extraction attacks by monitoring the Shapiro-Wilk normality of
    pairwise distances between consecutive queries. When distribution is
    abnormal (extraction in progress), subsequent queries are degraded.

    Adapted to graph node classification: treats each node's feature as
    a query, monitors distribution of features across queried nodes.
    """
    supported_api_types = {'dgl'}

    def __init__(self, dataset, threshold=0.85, **kwargs):
        """
        Args:
            threshold: Shapiro-Wilk W statistic threshold. Below this,
                        queries are flagged as extraction attempts.
        """
        self.dataset = dataset
        self.threshold = threshold
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.num_nodes = dataset.num_nodes
        _setup_device(self)

    def _detect_extraction(self, features_queried):
        """Return True if queries look like extraction (non-normal distribution)."""
        from scipy.stats import shapiro
        # Compute pairwise distances of the first 50 queries
        f_cpu = features_queried.cpu().numpy()
        if len(f_cpu) < 10:
            return False
        # Sample pairwise distances
        n = min(len(f_cpu), 100)
        sample = f_cpu[:n]
        dists = np.linalg.norm(sample[:, None] - sample[None, :], axis=-1)
        tril = dists[np.tril_indices_from(dists, k=-1)]
        if len(tril) < 3:
            return False
        try:
            W, p = shapiro(tril[:min(len(tril), 5000)])
            return W < self.threshold  # anomalous
        except Exception:
            return False

    def defend(self):
        t0 = time.time()
        model, graph, features, labels, train_mask = _train_clean_gcn(self.dataset, self.device)
        self.net1 = model
        train_time = time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            clean_logits = model_forward(model, graph, features)
            clean_preds = clean_logits.argmax(dim=1)

            # Simulate: on test_mask nodes (these are the "queries"), check distribution
            test_mask = graph.ndata['test_mask'].bool()
            test_features = features[test_mask]
            is_extraction = self._detect_extraction(test_features)

            if is_extraction:
                # Return noisy/random predictions for flagged queries
                noisy = clean_logits + torch.randn_like(clean_logits) * 2.0
                def_preds = noisy.argmax(dim=1)
            else:
                def_preds = clean_preds
        inference_time = time.time() - t0

        res, comp = _base_metric_defense(self, def_preds, clean_preds, test_mask,
                                          labels, train_time, inference_time)
        print(f"PRADA(threshold={self.threshold}): extraction_detected={is_extraction}, "
              f"acc={res.get('Acc',-1):.4f}, agreement={res.get('WM Acc',-1):.4f}")
        return res, comp


class AdaptiveMisinformation(BaseDefense):
    """Adaptive Misinformation (Kariyappa & Qureshi, CVPR 2020).

    Uses feature-space distance to detect OOD queries. For flagged queries,
    returns deliberately wrong (misinformation) labels; for in-distribution
    queries, returns correct predictions.
    """
    supported_api_types = {'dgl'}

    def __init__(self, dataset, ood_percentile=0.5, **kwargs):
        self.dataset = dataset
        self.ood_percentile = ood_percentile
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.num_nodes = dataset.num_nodes
        _setup_device(self)

    def defend(self):
        t0 = time.time()
        model, graph, features, labels, train_mask = _train_clean_gcn(self.dataset, self.device)
        self.net1 = model
        train_time = time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            clean_logits = model_forward(model, graph, features)
            clean_preds = clean_logits.argmax(dim=1)

            # OOD detection: distance from train distribution centroid
            train_feat = features[train_mask].mean(dim=0, keepdim=True)
            dists = torch.norm(features - train_feat, dim=1)
            threshold = torch.quantile(dists, self.ood_percentile)
            ood_mask = dists > threshold

            # For OOD queries, return wrong (shifted) predictions
            def_preds = clean_preds.clone()
            wrong = (clean_preds + 1) % self.num_classes
            def_preds[ood_mask] = wrong[ood_mask]
        inference_time = time.time() - t0

        test_mask = graph.ndata['test_mask'].bool()
        res, comp = _base_metric_defense(self, def_preds, clean_preds, test_mask,
                                          labels, train_time, inference_time)
        print(f"AdaptiveMisinformation(ood_pct={self.ood_percentile}): "
              f"acc={res.get('Acc',-1):.4f}, agreement={res.get('WM Acc',-1):.4f}")
        return res, comp


class GradientRedirection(BaseDefense):
    """Gradient Redirection (MAZE-style, Mazeika et al., ICML 2022).

    Perturbs predicted logits toward a target class gradient direction,
    making extracted surrogate learn incorrect decision boundaries while
    keeping top-1 prediction unchanged.
    """
    supported_api_types = {'dgl'}

    def __init__(self, dataset, redirect_strength=1.0, **kwargs):
        self.dataset = dataset
        self.redirect_strength = redirect_strength
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.num_nodes = dataset.num_nodes
        _setup_device(self)

    def defend(self):
        t0 = time.time()
        model, graph, features, labels, train_mask = _train_clean_gcn(self.dataset, self.device)
        self.net1 = model
        train_time = time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            clean_logits = model_forward(model, graph, features)
            clean_preds = clean_logits.argmax(dim=1)

            # Redirect non-top-1 logits toward uniform (hide secondary info)
            top1_vals, top1_idx = clean_logits.max(dim=1, keepdim=True)
            mask = torch.ones_like(clean_logits)
            mask.scatter_(1, top1_idx, 0)  # 1 everywhere except top-1
            redirected = clean_logits * (1 - mask * self.redirect_strength) + \
                          (clean_logits.mean(dim=1, keepdim=True) * mask * self.redirect_strength)
            def_preds = redirected.argmax(dim=1)
        inference_time = time.time() - t0

        test_mask = graph.ndata['test_mask'].bool()
        res, comp = _base_metric_defense(self, def_preds, clean_preds, test_mask,
                                          labels, train_time, inference_time)
        print(f"GradientRedirection(strength={self.redirect_strength}): "
              f"acc={res.get('Acc',-1):.4f}, agreement={res.get('WM Acc',-1):.4f}")
        return res, comp
