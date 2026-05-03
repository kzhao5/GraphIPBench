"""Cross-architecture model extraction experiment (Table NEW-C).

Tests all 9 victim-surrogate architecture combinations:
  GCN→GCN, GCN→GAT, GCN→GraphSAGE,
  GAT→GCN, GAT→GAT, GAT→GraphSAGE,
  GraphSAGE→GCN, GraphSAGE→GAT, GraphSAGE→GraphSAGE

For each combination, we:
  1. Train a victim with architecture A
  2. Query the victim on all nodes to get pseudolabels
  3. Train a surrogate with architecture B on the pseudolabels
  4. Measure fidelity (surrogate vs victim agreement on test set)

Usage:
  python scripts/run_cross_arch.py --dataset Cora --victim-arch gcn \
    --surrogate-arch gat --seed 0 --gpu --output-dir outputs/cross_arch
"""

import argparse
import json
import os
import time
import importlib

import torch
import torch.nn.functional as F
import dgl

from pygip.models.nn.backbones import create_model, model_forward
from pygip.utils.metrics import AttackMetric


DATASET_MAP = {
    'Cora': 'pygip.datasets.Cora',
    'CiteSeer': 'pygip.datasets.CiteSeer',
    'PubMed': 'pygip.datasets.PubMed',
    'Computers': 'pygip.datasets.Computers',
    'Photo': 'pygip.datasets.Photo',
    'CoauthorCS': 'pygip.datasets.CoauthorCS',
    'CoauthorPhysics': 'pygip.datasets.CoauthorPhysics',
    'RomanEmpire': 'pygip.datasets.RomanEmpire',
    'AmazonRatings': 'pygip.datasets.AmazonRatings',
    'OGBNArxiv': 'pygip.datasets.OGBNArxiv',
}


def train_model(model, graph, features, labels, train_mask, epochs=200, lr=0.01):
    """Train a GNN model on the given graph."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model_forward(model, graph, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def query_victim(victim, graph, features):
    """Query victim to get pseudolabels on all nodes."""
    victim.eval()
    with torch.no_grad():
        logits = model_forward(victim, graph, features)
    return logits


def train_surrogate_kd(surrogate, graph, features, teacher_logits, train_idx,
                        epochs=200, lr=0.01, temperature=3.0):
    """Train surrogate via knowledge distillation from teacher logits."""
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=lr, weight_decay=5e-4)
    soft_targets = F.softmax(teacher_logits / temperature, dim=1).detach()

    surrogate.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits_s = model_forward(surrogate, graph, features)
        # KD loss on train nodes
        loss = F.kl_div(
            F.log_softmax(logits_s[train_idx] / temperature, dim=1),
            soft_targets[train_idx],
            reduction='batchmean'
        ) * (temperature ** 2)
        loss.backward()
        optimizer.step()
    surrogate.eval()
    return surrogate


def evaluate(victim, surrogate, graph, features, test_mask, labels):
    """Compute fidelity, accuracy, F1."""
    victim.eval()
    surrogate.eval()
    with torch.no_grad():
        v_logits = model_forward(victim, graph, features)
        s_logits = model_forward(surrogate, graph, features)

    v_preds = v_logits.argmax(dim=1)
    s_preds = s_logits.argmax(dim=1)

    test_v = v_preds[test_mask]
    test_s = s_preds[test_mask]
    test_labels = labels[test_mask]

    fidelity = (test_s == test_v).float().mean().item()
    accuracy = (test_s == test_labels).float().mean().item()

    # F1 macro
    from sklearn.metrics import f1_score
    f1 = f1_score(test_labels.cpu(), test_s.cpu(), average='macro', zero_division=0)

    return {
        'fidelity': fidelity,
        'accuracy': accuracy,
        'f1': f1,
        'victim_acc': (test_v == test_labels).float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--victim-arch', type=str, required=True, choices=['gcn', 'gat', 'graphsage'])
    parser.add_argument('--surrogate-arch', type=str, required=True, choices=['gcn', 'gat', 'graphsage'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--victim-epochs', type=int, default=200)
    parser.add_argument('--surrogate-epochs', type=int, default=200)
    parser.add_argument('--output-dir', type=str, default='outputs/cross_arch')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    # Load dataset
    ds_path = DATASET_MAP[args.dataset]
    mod_name, cls_name = ds_path.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    dataset = getattr(mod, cls_name)(api_type='dgl', path='./data')

    graph = dataset.graph_data.to(device)
    features = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device)
    train_mask = graph.ndata['train_mask'].bool()
    test_mask = graph.ndata['test_mask'].bool()

    n_feat = dataset.num_features
    n_class = dataset.num_classes

    print(f"Dataset: {args.dataset} ({dataset.num_nodes} nodes, {n_feat} features, {n_class} classes)")
    print(f"Victim: {args.victim_arch}, Surrogate: {args.surrogate_arch}, Seed: {args.seed}")

    # 1. Train victim
    t0 = time.time()
    victim = create_model(args.victim_arch, n_feat, n_class).to(device)
    victim = train_model(victim, graph, features, labels, train_mask, epochs=args.victim_epochs)
    t_victim = time.time() - t0

    # 2. Query victim
    t0 = time.time()
    teacher_logits = query_victim(victim, graph, features)
    t_query = time.time() - t0

    # 3. Train surrogate via KD
    # Use all non-test nodes as training set for surrogate (simulates budget=1.0)
    query_idx = (~test_mask).nonzero(as_tuple=True)[0]

    t0 = time.time()
    surrogate = create_model(args.surrogate_arch, n_feat, n_class).to(device)
    surrogate = train_surrogate_kd(
        surrogate, graph, features, teacher_logits, query_idx,
        epochs=args.surrogate_epochs
    )
    t_surrogate = time.time() - t0

    # 4. Evaluate
    metrics = evaluate(victim, surrogate, graph, features, test_mask, labels)

    result = {
        'dataset': args.dataset,
        'victim_arch': args.victim_arch,
        'surrogate_arch': args.surrogate_arch,
        'seed': args.seed,
        'fidelity': metrics['fidelity'],
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'victim_acc': metrics['victim_acc'],
        'victim_train_time': t_victim,
        'query_time': t_query,
        'surrogate_train_time': t_surrogate,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, f'{args.dataset}.jsonl')
    with open(outfile, 'a') as f:
        f.write(json.dumps(result) + '\n')

    print(f"Result: fidelity={metrics['fidelity']:.4f}, acc={metrics['accuracy']:.4f}, "
          f"f1={metrics['f1']:.4f}, victim_acc={metrics['victim_acc']:.4f}")
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
