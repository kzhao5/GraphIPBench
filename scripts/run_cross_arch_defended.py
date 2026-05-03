"""Cross-architecture × new defenses: how do non-watermark defenses behave with different victim architectures?
"""
import argparse
import importlib
import json
import os
import time
import torch
import torch.nn.functional as F
from pygip.models.nn.backbones import create_model, model_forward

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

NEW_DEFENSES = {
    'OutputPerturbation_low': {'sigma': 0.05},
    'OutputPerturbation_high': {'sigma': 0.2},
    'PredictionRounding_top1': {'top_k': 1},
    'GradientRedirection': {'strength': 1.0},
}

def apply_defense(logits, defense_name, kwargs):
    if defense_name.startswith('OutputPerturbation'):
        return logits + torch.randn_like(logits) * kwargs['sigma']
    if defense_name == 'PredictionRounding_top1':
        vals, idx = logits.topk(1, dim=1)
        out = torch.full_like(logits, float('-inf'))
        out.scatter_(1, idx, vals)
        return out
    if defense_name == 'GradientRedirection':
        top1_vals, top1_idx = logits.max(dim=1, keepdim=True)
        mask = torch.ones_like(logits)
        mask.scatter_(1, top1_idx, 0)
        return logits * (1 - mask * kwargs['strength']) + logits.mean(dim=1, keepdim=True) * mask * kwargs['strength']
    return logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--victim-arch', required=True, choices=['gcn', 'gat', 'graphsage'])
    p.add_argument('--defense', required=True, choices=list(NEW_DEFENSES.keys()) + ['none'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--output-dir', default='outputs/cross_arch_defended')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    # Load dataset
    mod_name, cls_name = DATASET_MAP[args.dataset].rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    dataset = getattr(mod, cls_name)(api_type='dgl', path='./data')

    graph = dataset.graph_data.to(device)
    features = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device)
    train_mask = graph.ndata['train_mask'].bool()
    test_mask = graph.ndata['test_mask'].bool()

    # Train victim with specified architecture
    victim = create_model(args.victim_arch, dataset.num_features, dataset.num_classes).to(device)
    opt = torch.optim.Adam(victim.parameters(), lr=0.01, weight_decay=5e-4)
    victim.train()
    for _ in range(200):
        opt.zero_grad()
        out = model_forward(victim, graph, features)
        F.cross_entropy(out[train_mask], labels[train_mask]).backward()
        opt.step()
    victim.eval()

    # Get teacher predictions (defended)
    with torch.no_grad():
        teacher_logits = model_forward(victim, graph, features)
    if args.defense != 'none':
        def_kwargs = NEW_DEFENSES[args.defense]
        defended_logits = apply_defense(teacher_logits, args.defense, def_kwargs)
    else:
        defended_logits = teacher_logits

    teacher_preds = teacher_logits.argmax(dim=1)
    victim_acc = (teacher_preds[test_mask] == labels[test_mask]).float().mean().item()

    # Query budget: 50% of nodes
    query_idx = (~test_mask).nonzero(as_tuple=True)[0]
    soft_targets = F.softmax(defended_logits / 3.0, dim=1).detach()

    # Train surrogate (always GCN for fairness)
    surrogate = create_model('gcn', dataset.num_features, dataset.num_classes).to(device)
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
    fid = (s_preds[test_mask] == teacher_preds[test_mask]).float().mean().item() * 100
    acc = (s_preds[test_mask] == labels[test_mask]).float().mean().item() * 100

    result = {
        'dataset': args.dataset,
        'victim_arch': args.victim_arch,
        'defense': args.defense,
        'seed': args.seed,
        'victim_acc': victim_acc * 100,
        'surrogate_fidelity': fid,
        'surrogate_acc': acc,
        'status': 'ok',
    }

    os.makedirs(args.output_dir, exist_ok=True)
    outfile = f'{args.output_dir}/{args.dataset}.jsonl'
    with open(outfile, 'a') as f:
        f.write(json.dumps(result) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
