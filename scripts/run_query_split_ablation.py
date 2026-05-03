"""Query split ablation - test different query set sampling strategies.

Tests fidelity sensitivity to how query nodes are selected.
"""
import argparse
import importlib
import json
import os
import time
import torch
import torch.nn.functional as F
from pygip.models.nn.backbones import GCN, model_forward

DATASET_MAP = {
    'Cora': 'pygip.datasets.Cora',
    'CiteSeer': 'pygip.datasets.CiteSeer',
    'Computers': 'pygip.datasets.Computers',
    'RomanEmpire': 'pygip.datasets.RomanEmpire',
}

SPLIT_STRATEGIES = [
    'random',        # Uniform random selection
    'degree_high',   # High-degree nodes
    'degree_low',    # Low-degree nodes
    'class_balanced', # Balanced across classes
    'class_skewed',  # Heavily skewed to one class
]

def select_query_indices(strategy, graph, labels, n_query, seed=0):
    torch.manual_seed(seed)
    N = graph.num_nodes()
    n_query = min(n_query, N)

    if strategy == 'random':
        return torch.randperm(N)[:n_query]
    elif strategy == 'degree_high':
        degs = graph.in_degrees()
        _, idx = torch.sort(degs, descending=True)
        return idx[:n_query]
    elif strategy == 'degree_low':
        degs = graph.in_degrees()
        _, idx = torch.sort(degs)
        return idx[:n_query]
    elif strategy == 'class_balanced':
        ncls = int(labels.max().item()) + 1
        per_class = n_query // ncls
        selected = []
        for c in range(ncls):
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                perm = idx[torch.randperm(len(idx))[:per_class]]
                selected.append(perm)
        return torch.cat(selected) if selected else torch.randperm(N)[:n_query]
    elif strategy == 'class_skewed':
        # Pick mostly from one random class
        ncls = int(labels.max().item()) + 1
        focus_cls = torch.randint(0, ncls, (1,)).item()
        main = (labels == focus_cls).nonzero(as_tuple=True)[0]
        other = (labels != focus_cls).nonzero(as_tuple=True)[0]
        n_main = int(n_query * 0.8)
        if len(main) >= n_main:
            main_sel = main[torch.randperm(len(main))[:n_main]]
        else:
            main_sel = main
        other_sel = other[torch.randperm(len(other))[:n_query - len(main_sel)]]
        return torch.cat([main_sel, other_sel])
    return torch.randperm(N)[:n_query]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--output-dir', default='outputs/query_split_ablation')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    mod_name, cls_name = DATASET_MAP[args.dataset].rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    dataset = getattr(mod, cls_name)(api_type='dgl', path='./data')

    graph = dataset.graph_data.to(device)
    features = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device)
    train_mask = graph.ndata['train_mask'].bool()
    test_mask = graph.ndata['test_mask'].bool()
    N = graph.num_nodes()

    # Train victim
    victim = GCN(dataset.num_features, dataset.num_classes).to(device)
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
    outfile = f'{args.output_dir}/{args.dataset}.jsonl'

    n_query = int(N * 0.5)  # fix budget at 0.5x
    soft_targets = F.softmax(v_logits / 3.0, dim=1).detach()

    for strategy in SPLIT_STRATEGIES:
        torch.manual_seed(args.seed)
        try:
            query_idx = select_query_indices(strategy, graph, labels, n_query, seed=args.seed).to(device)

            surrogate = GCN(dataset.num_features, dataset.num_classes).to(device)
            opt = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)
            surrogate.train()
            for _ in range(200):
                opt.zero_grad()
                s_logits = model_forward(surrogate, graph, features)
                loss = F.kl_div(
                    F.log_softmax(s_logits[query_idx] / 3.0, dim=1),
                    soft_targets[query_idx], reduction='batchmean'
                ) * 9.0
                loss.backward()
                opt.step()
            surrogate.eval()

            with torch.no_grad():
                s_logits = model_forward(surrogate, graph, features)
            s_preds = s_logits.argmax(dim=1)
            fid = (s_preds[test_mask] == victim_preds[test_mask]).float().mean().item() * 100
            acc = (s_preds[test_mask] == labels[test_mask]).float().mean().item() * 100

            result = {
                'dataset': args.dataset,
                'split_strategy': strategy,
                'seed': args.seed,
                'n_query': len(query_idx),
                'fidelity': fid,
                'accuracy': acc,
                'victim_acc': victim_acc * 100,
                'status': 'ok',
            }
        except Exception as e:
            import traceback; traceback.print_exc()
            result = {
                'dataset': args.dataset, 'split_strategy': strategy,
                'seed': args.seed, 'status': 'error', 'error': str(e)[:200],
            }
        with open(outfile, 'a') as f:
            f.write(json.dumps(result) + '\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
