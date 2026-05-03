"""Link Prediction RQ1/RQ2 on ogbl-collab or Cora link pred.

Extends link prediction experiments to include:
  - Large-scale link pred dataset (ogbl-collab 235K nodes)
  - 9+ attacks (MEA, AdvMEA, CEGA, DFEA)
  - 5+ defenses
"""
import argparse
import json
import os
import time
import torch
import torch.nn.functional as F
import dgl

from pygip.models.nn.link_pred import GCNLinkPred


class OGBLCollabDataset:
    """Wrapper for ogbl-collab dataset for link prediction."""
    def __init__(self, path='./data'):
        self.api_type = 'dgl'
        from ogb.linkproppred import DglLinkPropPredDataset
        dataset = DglLinkPropPredDataset(name='ogbl-collab', root=path)
        graph = dataset[0]
        self.graph_dataset = dataset
        self.graph_data = graph
        self.num_nodes = graph.num_nodes()
        self.num_features = graph.ndata['feat'].shape[1]
        self.num_classes = 2

        split = dataset.get_edge_split()
        # Build EdgeSplit
        from pygip.datasets.link_pred_cora import EdgeSplit
        train_pos = split['train']['edge'].t()
        valid_pos = split['valid']['edge'].t()
        test_pos = split['test']['edge'].t()
        # Generate same-number negatives randomly
        import random
        N = graph.num_nodes()
        def sample_neg(k):
            neg = torch.randint(0, N, (2, k))
            return neg
        train_neg = sample_neg(train_pos.shape[1])
        valid_neg = sample_neg(valid_pos.shape[1])
        test_neg = sample_neg(test_pos.shape[1])
        self.edge_split = EdgeSplit(
            train_pos=train_pos, val_pos=valid_pos, test_pos=test_pos,
            train_neg=train_neg, val_neg=valid_neg, test_neg=test_neg,
        )


def train_victim(dataset, device, epochs=100, lr=0.01, hidden=64):
    model = GCNLinkPred(dataset.num_features, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    g = dataset.graph_data.to(device)
    x = g.ndata['feat'].float().to(device)
    pos = dataset.edge_split.train_pos.to(device)
    neg = dataset.edge_split.train_neg.to(device)
    # Sample subset for training on large graphs
    max_edges = 50000
    if pos.shape[1] > max_edges:
        idx = torch.randperm(pos.shape[1])[:max_edges]
        pos = pos[:, idx]
    if neg.shape[1] > max_edges:
        idx = torch.randperm(neg.shape[1])[:max_edges]
        neg = neg[:, idx]

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        edges = torch.cat([pos, neg], dim=1)
        labels = torch.cat([torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])]).to(device)
        logits, _ = model(g, x, edges)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()
    model.eval()
    return model


def eval_on_test(model, dataset, device):
    model.eval()
    g = dataset.graph_data.to(device)
    x = g.ndata['feat'].float().to(device)
    pos = dataset.edge_split.test_pos.to(device)
    neg = dataset.edge_split.test_neg.to(device)
    max_edges = 20000
    if pos.shape[1] > max_edges:
        idx = torch.randperm(pos.shape[1])[:max_edges]
        pos = pos[:, idx]
        idx = torch.randperm(neg.shape[1])[:max_edges]
        neg = neg[:, idx]
    edges = torch.cat([pos, neg], dim=1)
    labels = torch.cat([torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])]).to(device)
    with torch.no_grad():
        logits, _ = model(g, x, edges)
        preds = (torch.sigmoid(logits) > 0.5).long()
    acc = (preds == labels.long()).float().mean().item()
    return acc, preds, labels.long()


DEFENSES = {
    'none': ('None', {}),
    'OutputPerturbation_low': ('OutputPerturbation', {'sigma': 0.05}),
    'OutputPerturbation_high': ('OutputPerturbation', {'sigma': 0.2}),
    'PredictionRounding_2bit': ('PredictionRounding', {'precision_bits': 2}),
    'GradientRedirection': ('GradientRedirection', {'redirect_strength': 0.5}),
}


class DefendedLinkPredVictim(torch.nn.Module):
    """Wraps a GCNLinkPred victim with a defense on its edge logits.

    LinkPred output is a per-edge real-valued logit (BCE-style).
    OutputPerturbation: add Gaussian noise to logits.
    PredictionRounding: round sigmoid(logit) to discrete bits.
    GradientRedirection: shrink logit magnitudes towards mean.
    """
    def __init__(self, base, name, kwargs):
        super().__init__()
        self.base = base
        self.name = name
        self.kwargs = kwargs

    def forward(self, g, x, edges):
        logits, embs = self.base(g, x, edges)
        if self.name == 'OutputPerturbation':
            sigma = self.kwargs.get('sigma', 0.1)
            logits = logits + torch.randn_like(logits) * sigma
        elif self.name == 'PredictionRounding':
            bits = self.kwargs.get('precision_bits', 2)
            probs = torch.sigmoid(logits)
            levels = 2 ** bits
            probs = (torch.round(probs * levels) / levels).clamp(1e-6, 1 - 1e-6)
            logits = torch.log(probs / (1 - probs))
        elif self.name == 'GradientRedirection':
            strength = self.kwargs.get('redirect_strength', 0.5)
            mu = logits.mean()
            logits = logits * (1 - strength) + mu * strength
        return logits, embs

    def eval(self):
        self.base.eval()
        return super().eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ogbl-collab', choices=['ogbl-collab', 'Cora'])
    p.add_argument('--attack', default=None)
    p.add_argument('--defense', default='none', choices=list(DEFENSES.keys()))
    p.add_argument('--budget', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--output-dir', default='outputs/link_pred')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    # Load dataset
    if args.dataset == 'ogbl-collab':
        dataset = OGBLCollabDataset(path='./data')
    else:
        from pygip.datasets import CoraLinkPredDataset
        dataset = CoraLinkPredDataset()

    print(f"Dataset {args.dataset}: {dataset.num_nodes} nodes, {dataset.num_features} features")

    # Train victim
    t0 = time.time()
    victim = train_victim(dataset, device, epochs=100)
    victim_time = time.time() - t0
    victim_acc, _, _ = eval_on_test(victim, dataset, device)
    print(f"Victim trained in {victim_time:.1f}s, acc={victim_acc:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    outfile = f'{args.output_dir}/{args.dataset}.jsonl'

    # Wrap victim with defense (if any)
    def_name, def_kwargs = DEFENSES[args.defense]
    if def_name == 'None':
        attack_target = victim  # no defense
    else:
        attack_target = DefendedLinkPredVictim(victim, def_name, def_kwargs).to(device)
        attack_target.eval()

    # Import attacks
    from pygip.models.attack.linkpred_attacks import (
        ModelExtractionAttack0, ModelExtractionAttack1, ModelExtractionAttack2,
        ModelExtractionAttack3, ModelExtractionAttack4, ModelExtractionAttack5,
        AdvMEALinkPred, CEGALinkPred,
        DFEATypeILinkPred, DFEATypeIILinkPred, DFEATypeIIILinkPred
    )
    ATTACKS = {
        'MEA0': ModelExtractionAttack0,
        'MEA1': ModelExtractionAttack1,
        'MEA2': ModelExtractionAttack2,
        'MEA3': ModelExtractionAttack3,
        'MEA4': ModelExtractionAttack4,
        'MEA5': ModelExtractionAttack5,
        'AdvMEA': AdvMEALinkPred,
        'CEGA': CEGALinkPred,
        'DFEA_I': DFEATypeILinkPred,
        'DFEA_II': DFEATypeIILinkPred,
        'DFEA_III': DFEATypeIIILinkPred,
    }

    attacks_to_run = ATTACKS.items() if args.attack is None else [(args.attack, ATTACKS[args.attack])]

    for atk_name, atk_cls in attacks_to_run:
        print(f"\n=== {atk_name} (defense={args.defense}) ===")
        try:
            t0 = time.time()
            atk = atk_cls(dataset, attack_target, args.budget, device=device)

            # Run the attack
            if hasattr(atk, 'run'):
                ret = atk.run()
            elif hasattr(atk, 'attack'):
                ret = atk.attack()
            else:
                raise RuntimeError("No run/attack method")
            # Some attacks return (surrogate, acc, f1, fidelity); others return surrogate
            if isinstance(ret, tuple):
                surrogate = ret[0]
            else:
                surrogate = ret
            atk_time = time.time() - t0

            # Evaluate surrogate
            sur_acc, sur_preds, labels = eval_on_test(surrogate, dataset, device)
            _, vic_preds, _ = eval_on_test(victim, dataset, device)
            fidelity = (sur_preds == vic_preds).float().mean().item()

            result = {
                'dataset': args.dataset,
                'attack': atk_name,
                'defense': args.defense,
                'budget': args.budget,
                'seed': args.seed,
                'victim_acc': victim_acc,
                'surrogate_acc': sur_acc,
                'fidelity': fidelity,
                'attack_time': atk_time,
                'status': 'ok',
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {
                'dataset': args.dataset, 'attack': atk_name,
                'defense': args.defense,
                'budget': args.budget, 'seed': args.seed,
                'status': 'error', 'error': str(e)[:200],
            }

        with open(outfile, 'a') as f:
            f.write(json.dumps(result) + '\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
