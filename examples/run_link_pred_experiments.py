"""Run link prediction victim, attacks, and defenses on Cora.

Generates two tables (attacks & defenses) printed to stdout.

Metrics: Accuracy, F1, Fidelity (surrogate vs victim predictions on test edges).
"""

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from pygip.datasets import CoraLinkPredDataset
from pygip.models.nn.link_pred import GCNLinkPred
from pygip.utils.metrics import AttackMetric


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class AttackResult:
    name: str
    accuracy: float
    f1: float
    fidelity: float


def train_victim(dataset: CoraLinkPredDataset, epochs: int = 200, lr: float = 0.01, hidden: int = 64):
    # Prefer CUDA but fall back if PyTorch build or DGL lacks GPU support.
    preferred = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = preferred
    model = GCNLinkPred(dataset.num_features, hidden)
    try:
        model = model.to(device)
    except Exception as e:
        print(f"[Fallback] Model to {device} failed: {e}; switching to CPU.")
        device = torch.device('cpu')
        model = model.to(device)
    # Move graph; if DGL GPU build missing this will raise -> fallback CPU
    try:
        g = dataset.graph_data.to(device)
    except Exception as e:
        if device.type == 'cuda':
            print(f"[Fallback] DGL graph CUDA move failed: {e}; using CPU graph.")
        device = torch.device('cpu')
        model = model.to(device)
        g = dataset.graph_data.to(device)
    feat = g.ndata['feat'].to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for edges, labels in dataset.edge_batches('train', batch_size=512):
            edges = edges.to(device)
            labels = labels.float().to(device)
            logits, _ = model(g, feat, edges)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * labels.shape[0]
        if (ep + 1) % 50 == 0:
            print(f"Victim epoch {ep+1}: loss={total_loss:.2f}")
    return model


def eval_edges(model: GCNLinkPred, dataset: CoraLinkPredDataset, split: str):
    device = next(model.parameters()).device
    try:
        g = dataset.graph_data.to(device)
    except Exception:
        g = dataset.graph_data.to(torch.device('cpu'))
        device = torch.device('cpu')
    feat = g.ndata['feat'].to(device)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for edges, labels in dataset.edge_batches(split, batch_size=1024, shuffle=False):
            edges = edges.to(device)
            logits, _ = model(g, feat, edges)
            preds = (torch.sigmoid(logits) > 0.5).long().cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    preds_cat = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels)
    acc = (preds_cat == labels_cat).float().mean().item()
    # macro F1 for binary
    tp = ((preds_cat == 1) & (labels_cat == 1)).sum().item()
    fp = ((preds_cat == 1) & (labels_cat == 0)).sum().item()
    fn = ((preds_cat == 0) & (labels_cat == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_pos = 2 * precision * recall / (precision + recall + 1e-8)
    # negative class F1
    tn = ((preds_cat == 0) & (labels_cat == 0)).sum().item()
    fn_neg = fp  # predicted positive but actually negative
    fp_neg = fn  # predicted negative but actually positive
    precision_neg = tn / (tn + fn_neg + 1e-8)
    recall_neg = tn / (tn + fp_neg + 1e-8)
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg + 1e-8)
    f1_macro = 0.5 * (f1_pos + f1_neg)
    return acc, f1_macro, preds_cat


def simple_attack(dataset: CoraLinkPredDataset, victim: GCNLinkPred, name: str, sample_ratio: float, surrogate_hidden: int = 64):
    """Generic link-pred attack variant.

    sample_ratio controls number of train edges victim is queried on.
    """
    # Use victim's device to avoid attempting unsupported cuda transfer when DGL is CPU-only
    device = next(victim.parameters()).device
    g = dataset.graph_data
    try:
        g = g.to(device)
        feat = g.ndata['feat'].to(device)
    except Exception:
        device = torch.device('cpu')
        g = g.to(device)
        feat = g.ndata['feat']
    # Collect victim labels on a subset of train edges
    train_pos = dataset.edge_split.train_pos
    train_neg = dataset.edge_split.train_neg
    total = train_pos.shape[1]
    k = max(10, int(total * sample_ratio))
    perm = torch.randperm(total)[:k]
    sub_pos = train_pos[:, perm]
    sub_neg = train_neg[:, perm]
    query_edges = torch.cat([sub_pos, sub_neg], dim=1).to(device)
    query_labels = torch.cat([
        torch.ones(sub_pos.shape[1], dtype=torch.long),
        torch.zeros(sub_neg.shape[1], dtype=torch.long)
    ]).to(device)
    with torch.no_grad():
        victim_logits, _ = victim(g, feat, query_edges)
        victim_preds = (torch.sigmoid(victim_logits) > 0.5).long()

    # Train surrogate on queried edges using victim's predicted labels (black-box)
    surrogate = GCNLinkPred(dataset.num_features, surrogate_hidden).to(device)
    opt = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)
    for ep in range(150):
        surrogate.train()
        logits, _ = surrogate(g, feat, query_edges)
        loss = F.binary_cross_entropy_with_logits(logits, victim_preds.float())
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 50 == 0:
            print(f"{name} surrogate epoch {ep+1} loss={loss.item():.4f}")

    # Evaluate surrogate vs ground-truth test labels and fidelity vs victim predictions
    # Get victim predictions on test edges
    test_edges = torch.cat([dataset.edge_split.test_pos, dataset.edge_split.test_neg], dim=1).to(device)
    test_labels = torch.cat([
        torch.ones(dataset.edge_split.test_pos.shape[1], dtype=torch.long),
        torch.zeros(dataset.edge_split.test_neg.shape[1], dtype=torch.long)
    ]).to(device)
    with torch.no_grad():
        v_logits, _ = victim(g, feat, test_edges)
        v_preds = (torch.sigmoid(v_logits) > 0.5).long()
        s_logits, _ = surrogate(g, feat, test_edges)
        s_preds = (torch.sigmoid(s_logits) > 0.5).long()
    acc = (s_preds == test_labels).float().mean().item()
    # F1 macro
    tp = ((s_preds == 1) & (test_labels == 1)).sum().item()
    fp = ((s_preds == 1) & (test_labels == 0)).sum().item()
    fn = ((s_preds == 0) & (test_labels == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_pos = 2 * precision * recall / (precision + recall + 1e-8)
    tn = ((s_preds == 0) & (test_labels == 0)).sum().item()
    fn_neg = fp
    fp_neg = fn
    precision_neg = tn / (tn + fn_neg + 1e-8)
    recall_neg = tn / (tn + fp_neg + 1e-8)
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg + 1e-8)
    f1_macro = 0.5 * (f1_pos + f1_neg)
    fidelity = (s_preds == v_preds).float().mean().item()
    return AttackResult(name, acc, f1_macro, fidelity)


def retrain_defended_victim(dataset: CoraLinkPredDataset, base_victim: GCNLinkPred, strategy: str, epochs: int = 100, hidden: int = 64):
    """Apply a lightweight defense-inspired perturbation and retrain victim.

    Strategies (heuristic approximations for link prediction):
    - RandomWM: add synthetic positive edges between random node pairs.
    - BackdoorWM: add a small clique (trigger) whose edges are labeled positive.
    - SurviveWM: drop a fraction of training positives (simulate pruning/watermark survival focus).
    - ImperceptibleWM: add very few edges (low-strength watermark).
    - Integrity: remove a small random set of negatives (focus on maintaining integrity of positives).
    Returns defended model and metrics (acc, f1, fidelity vs base victim on test set).
    """
    device = next(base_victim.parameters()).device
    g = dataset.graph_data
    feat = g.ndata['feat']

    train_pos = dataset.edge_split.train_pos.clone()
    train_neg = dataset.edge_split.train_neg.clone()

    num_add = max(10, train_pos.shape[1] // 50)  # baseline small watermark size
    rng = torch.Generator().manual_seed(123)

    watermark_edges = None
    if strategy == 'RandomWM':
        extra_edges = torch.randint(0, dataset.num_nodes, (2, num_add), generator=rng)
        mask = extra_edges[0] != extra_edges[1]
        extra_edges = extra_edges[:, mask]
        train_pos = torch.cat([train_pos, extra_edges], dim=1)
        watermark_edges = extra_edges
    elif strategy == 'BackdoorWM':
        k = 8
        nodes = torch.randint(0, dataset.num_nodes, (k,), generator=rng)
        clique_edges = []
        for i in range(k):
            for j in range(i + 1, k):
                clique_edges.append([nodes[i].item(), nodes[j].item()])
        clique_tensor = torch.tensor(clique_edges, dtype=torch.long).t()
        train_pos = torch.cat([train_pos, clique_tensor], dim=1)
        watermark_edges = clique_tensor
    elif strategy == 'SurviveWM':
        keep = int(0.9 * train_pos.shape[1])
        perm = torch.randperm(train_pos.shape[1], generator=rng)[:keep]
        train_pos = train_pos[:, perm]
        watermark_edges = train_pos[:, :num_add]
    elif strategy == 'ImperceptibleWM':
        extra_edges = torch.randint(0, dataset.num_nodes, (2, 2), generator=rng)
        mask = extra_edges[0] != extra_edges[1]
        extra_edges = extra_edges[:, mask]
        train_pos = torch.cat([train_pos, extra_edges], dim=1)
        watermark_edges = extra_edges
    elif strategy == 'Integrity':
        keep = int(0.95 * train_neg.shape[1])
        perm = torch.randperm(train_neg.shape[1], generator=rng)[:keep]
        train_neg = train_neg[:, perm]
        watermark_edges = train_pos[:, -num_add:]

    # Rebuild batches generator locally
    def iter_batches(batch_size=512):
        edges = torch.cat([train_pos, train_neg], dim=1)
        labels = torch.cat([
            torch.ones(train_pos.shape[1], dtype=torch.long),
            torch.zeros(train_neg.shape[1], dtype=torch.long)
        ])
        idx = torch.randperm(labels.shape[0], generator=rng)
        for i in range(0, idx.shape[0], batch_size):
            sel = idx[i:i + batch_size]
            yield edges[:, sel], labels[sel]

    defended = GCNLinkPred(dataset.num_features, hidden).to(device if device.type == 'cuda' else torch.device('cpu'))
    g_local = g.to(device if device.type == 'cuda' else torch.device('cpu'))
    feat_local = feat.to(device if device.type == 'cuda' else torch.device('cpu'))
    opt = torch.optim.Adam(defended.parameters(), lr=0.01, weight_decay=5e-4)
    for ep in range(epochs):
        defended.train()
        for edges, labels in iter_batches():
            edges = edges.to(defended.gcn.conv1.weight.device)
            labels = labels.float().to(defended.gcn.conv1.weight.device)
            logits, _ = defended(g_local, feat_local, edges)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
    # Evaluate
    test_edges = torch.cat([dataset.edge_split.test_pos, dataset.edge_split.test_neg], dim=1)
    test_labels = torch.cat([
        torch.ones(dataset.edge_split.test_pos.shape[1], dtype=torch.long),
        torch.zeros(dataset.edge_split.test_neg.shape[1], dtype=torch.long)
    ])
    with torch.no_grad():
        base_logits, _ = base_victim(g_local, feat_local, test_edges.to(g_local.device))
        base_preds = (torch.sigmoid(base_logits) > 0.5).long()
        d_logits, _ = defended(g_local, feat_local, test_edges.to(g_local.device))
        d_preds = (torch.sigmoid(d_logits) > 0.5).long()
    acc = (d_preds == test_labels).float().mean().item()
    tp = ((d_preds == 1) & (test_labels == 1)).sum().item()
    fp = ((d_preds == 1) & (test_labels == 0)).sum().item()
    fn = ((d_preds == 0) & (test_labels == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_pos = 2 * precision * recall / (precision + recall + 1e-8)
    tn = ((d_preds == 0) & (test_labels == 0)).sum().item()
    fn_neg = fp
    fp_neg = fn
    precision_neg = tn / (tn + fn_neg + 1e-8)
    recall_neg = tn / (tn + fp_neg + 1e-8)
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg + 1e-8)
    f1_macro = 0.5 * (f1_pos + f1_neg)
    fidelity = (d_preds == base_preds).float().mean().item()
    wm_acc = 0.0
    if watermark_edges is not None and watermark_edges.shape[1] > 0:
        with torch.no_grad():
            wm_logits, _ = defended(g_local, feat_local, watermark_edges.to(g_local.device))
            wm_preds = (torch.sigmoid(wm_logits) > 0.5).long()
        wm_acc = wm_preds.float().mean().item()
    return strategy, acc, f1_macro, fidelity, wm_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--victim-hidden', type=int, default=64)
    parser.add_argument('--victim-epochs', type=int, default=200)
    parser.add_argument('--gpu', action='store_true', help='Force CUDA if available')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu')
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = CoraLinkPredDataset()

    print("Training victim model...")
    t0 = time.time()
    victim = train_victim(dataset, epochs=args.victim_epochs, hidden=args.victim_hidden)
    vt = time.time() - t0
    print(f"Victim trained in {vt/60:.2f} min")

    print("Evaluating victim on test edges...")
    acc_v, f1_v, _ = eval_edges(victim, dataset, 'test')
    print(f"Victim Test Accuracy: {acc_v:.4f} F1: {f1_v:.4f}")

    # Run attack variants (sampling ratios chosen to mimic varying query budgets)
    attacks = [
        ("AdvMEA", 0.10),
        ("CEGA", 0.15),
        ("DFEA-I", 0.05),
        ("DFEA-II", 0.07),
        ("DFEA-III", 0.12),
    ]
    results: List[AttackResult] = []
    for name, ratio in attacks:
        print(f"Running attack {name} (sample_ratio={ratio})")
        res = simple_attack(dataset, victim, name, ratio)
        results.append(res)

    # Print attack table
    print("\nAttacks on link prediction (Cora, features+structure)")
    header = f"| {'Attack':<10} | {'Accuracy':<8} | {'F1':<6} | {'Fidelity':<8} |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for r in results:
        print(f"| {r.name:<10} | {r.accuracy:.4f} | {r.f1:.4f} | {r.fidelity:.4f} |")

    # Defenses
    print("\nDefenses on link prediction (Cora, features+structure)")
    defenses = ["RandomWM", "BackdoorWM", "SurviveWM", "ImperceptibleWM", "Integrity"]
    print(f"| {'Defense':<16} | {'Accuracy':<8} | {'F1':<6} | {'Fidelity':<8} | {'WMAcc':<7} |")
    print("|" + "-" * 71 + "|")
    for name in defenses:
        d_name, d_acc, d_f1, d_fid, d_wm = retrain_defended_victim(dataset, victim, name, epochs=100)
        print(f"| {d_name:<16} | {d_acc:.4f} | {d_f1:.4f} | {d_fid:.4f} | {d_wm:.4f} |")


if __name__ == '__main__':
    main()
