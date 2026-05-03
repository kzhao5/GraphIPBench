"""Benchmark design ablation: extra budget granularity (Table NEW-E).

Tests budgets outside the standard 0.05-1.00 range:
  - 0.02× (extremely low budget)
  - 0.75× (between 0.50 and 1.00)
  - 2.00× (overshooting - do extra queries help?)

Usage:
  python scripts/run_budget_ablation.py --dataset Cora --seed 0 --gpu \
    --output-dir outputs/budget_ablation
"""

import argparse
import json
import os
import time
import importlib
import torch
import torch.nn.functional as F

from pygip.models.nn.backbones import GCN, create_model, model_forward


DATASET_MAP = {
    'Cora': 'pygip.datasets.Cora',
    'CiteSeer': 'pygip.datasets.CiteSeer',
    'Computers': 'pygip.datasets.Computers',
    'RomanEmpire': 'pygip.datasets.RomanEmpire',
}

EXTRA_BUDGETS = [0.02, 0.75, 2.00]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--output-dir', type=str, default='outputs/budget_ablation')
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
    n_feat, n_class = dataset.num_features, dataset.num_classes
    n_nodes = dataset.num_nodes

    # Train victim once
    victim = GCN(n_feat, n_class).to(device)
    opt = torch.optim.Adam(victim.parameters(), lr=0.01, weight_decay=5e-4)
    victim.train()
    for _ in range(200):
        opt.zero_grad()
        out = model_forward(victim, graph, features)
        F.cross_entropy(out[train_mask], labels[train_mask]).backward()
        opt.step()
    victim.eval()

    with torch.no_grad():
        v_logits = model_forward(victim, graph, features)
    victim_preds = v_logits.argmax(dim=1)
    victim_acc = (victim_preds[test_mask] == labels[test_mask]).float().mean().item()

    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, f'{args.dataset}.jsonl')

    for budget in EXTRA_BUDGETS:
        # Determine query set size
        query_size = min(int(n_nodes * budget), n_nodes)
        query_idx = torch.randperm(n_nodes, device=device)[:query_size]

        # Get pseudolabels
        soft_targets = F.softmax(v_logits / 3.0, dim=1).detach()

        # Train surrogate
        surrogate = GCN(n_feat, n_class).to(device)
        opt_s = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)
        surrogate.train()
        for _ in range(200):
            opt_s.zero_grad()
            s_logits = model_forward(surrogate, graph, features)
            loss = F.kl_div(
                F.log_softmax(s_logits[query_idx] / 3.0, dim=1),
                soft_targets[query_idx], reduction='batchmean'
            ) * 9.0
            loss.backward()
            opt_s.step()
        surrogate.eval()

        with torch.no_grad():
            s_logits = model_forward(surrogate, graph, features)
        s_preds = s_logits.argmax(dim=1)

        fidelity = (s_preds[test_mask] == victim_preds[test_mask]).float().mean().item()
        accuracy = (s_preds[test_mask] == labels[test_mask]).float().mean().item()

        result = {
            'dataset': args.dataset,
            'budget': budget,
            'seed': args.seed,
            'query_nodes': query_size,
            'fidelity': fidelity,
            'accuracy': accuracy,
            'victim_acc': victim_acc,
        }

        with open(outfile, 'a') as f:
            f.write(json.dumps(result) + '\n')
        print(f"Budget={budget:.2f}: fidelity={fidelity:.4f}, acc={accuracy:.4f}")


if __name__ == '__main__':
    main()
