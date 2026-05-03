"""Baseline utility: train clean models with each backbone used by defenses.

Tests 3 backbones (GCN / GraphSAGE / GCN_PyG) to provide baseline Acc/F1
for comparing 'utility drop' caused by each defense.
"""
import argparse
import importlib
import json
import os
import time
import torch
import torch.nn.functional as F
from pygip.models.nn.backbones import GCN, GraphSAGE, GCN_PyG, model_forward

DATASET_MAP = {
    'Cora': 'pygip.datasets.Cora',
    'CiteSeer': 'pygip.datasets.CiteSeer',
    'PubMed': 'pygip.datasets.PubMed',
    'Computers': 'pygip.datasets.Computers',
    'Photo': 'pygip.datasets.Photo',
    'CoauthorCS': 'pygip.datasets.CoauthorCS',
    'CoauthorPhysics': 'pygip.datasets.CoauthorPhysics',
    'OGBNArxiv': 'pygip.datasets.OGBNArxiv',
    'RomanEmpire': 'pygip.datasets.RomanEmpire',
    'AmazonRatings': 'pygip.datasets.AmazonRatings',
}

BACKBONES = {
    'GCN_16': ('gcn', 16),           # BackdoorWM, SurviveWM, Integrity, non-WM defenses
    'GraphSAGE_128': ('graphsage', 128),  # RandomWM
    'GCN_PyG_128': ('gcn_pyg', 128),     # ImperceptibleWM
}


def train_clean(backbone_spec, dataset, device, epochs=200):
    arch, hidden = backbone_spec
    in_dim = dataset.num_features
    out_dim = dataset.num_classes

    if arch == 'gcn':
        model = GCN(in_dim, out_dim, hidden_dim=hidden).to(device)
        is_pyg = False
    elif arch == 'graphsage':
        model = GraphSAGE(in_dim, hidden, out_dim).to(device)
        is_pyg = False
    elif arch == 'gcn_pyg':
        model = GCN_PyG(in_dim, hidden, out_dim).to(device)
        is_pyg = True

    graph = dataset.graph_data.to(device)
    features = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device)
    train_mask = graph.ndata['train_mask'].bool()
    test_mask = graph.ndata['test_mask'].bool()

    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        if is_pyg:
            # Extract edge_index
            src, dst = graph.edges()
            edge_index = torch.stack([src, dst], dim=0).to(device)
            logits = model(features, edge_index)
        else:
            logits = model_forward(model, graph, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        opt.step()
    model.eval()

    with torch.no_grad():
        if is_pyg:
            src, dst = graph.edges()
            edge_index = torch.stack([src, dst], dim=0).to(device)
            logits = model(features, edge_index)
        else:
            logits = model_forward(model, graph, features)
    preds = logits.argmax(dim=1)

    test_labels = labels[test_mask]
    test_preds = preds[test_mask]

    acc = (test_preds == test_labels).float().mean().item()

    from sklearn.metrics import f1_score
    f1 = f1_score(test_labels.cpu(), test_preds.cpu(), average='macro', zero_division=0)

    return {'accuracy': acc * 100, 'f1': f1 * 100}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--output-dir', default='outputs/baseline_utility')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    mod_name, cls_name = DATASET_MAP[args.dataset].rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    dataset = getattr(mod, cls_name)(api_type='dgl', path='./data')

    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, f'{args.dataset}.jsonl')

    for backbone_name, spec in BACKBONES.items():
        try:
            t0 = time.time()
            metrics = train_clean(spec, dataset, device, epochs=200)
            elapsed = time.time() - t0
            result = {
                'dataset': args.dataset,
                'backbone': backbone_name,
                'seed': args.seed,
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1'],
                'training_time': elapsed,
                'status': 'ok',
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {
                'dataset': args.dataset,
                'backbone': backbone_name,
                'seed': args.seed,
                'status': 'error',
                'error': str(e)[:200],
            }
        with open(outfile, 'a') as f:
            f.write(json.dumps(result) + '\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
