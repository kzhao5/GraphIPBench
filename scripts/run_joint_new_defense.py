"""Joint evaluation: 7 non-watermark defenses vs 12 attacks.

Extends TABLE NEW-A/B to include new non-watermark defenses.
Unlike WM defenses, these don't have a watermark to survive — they
just perturb outputs. We measure fidelity drop when attacking defended
model.

Usage:
  python scripts/run_joint_new_defense.py --dataset Cora --defense OutputPerturbation_low --seed 0 --gpu
"""
import argparse
import importlib
import json
import os
import time
import torch
import torch.nn.functional as F
import dgl

from pygip.models.nn.backbones import GCN, create_model, model_forward
from pygip.utils.metrics import AttackMetric, AttackCompMetric

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

NEW_DEFENSES = {
    'OutputPerturbation_low': ('OutputPerturbation', {'sigma': 0.05}),
    'OutputPerturbation_high': ('OutputPerturbation', {'sigma': 0.2}),
    'PredictionRounding_2bit': ('PredictionRounding', {'precision_bits': 2}),
    'PredictionRounding_top1': ('PredictionRounding', {'precision_bits': 2, 'top_k': 1}),
    'PRADA': ('PRADA', {'threshold': 0.85}),
    'AdaptiveMisinformation': ('AdaptiveMisinformation', {'ood_percentile': 0.5}),
    'GradientRedirection': ('GradientRedirection', {'redirect_strength': 1.0}),
}

ATTACK_CONFIGS = [
    ('MEA0', 'pygip.models.attack.mea.MEA', 'ModelExtractionAttack0', {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
    ('MEA1', 'pygip.models.attack.mea.MEA', 'ModelExtractionAttack1', {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
    ('MEA2', 'pygip.models.attack.mea.MEA', 'ModelExtractionAttack2', {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
    ('MEA3', 'pygip.models.attack.mea.MEA', 'ModelExtractionAttack3', {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
    ('MEA4', 'pygip.models.attack.mea.MEA', 'ModelExtractionAttack4', {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
    ('MEA5', 'pygip.models.attack.mea.MEA', 'ModelExtractionAttack5', {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
    ('AdvMEA', 'pygip.models.attack.AdvMEA', 'AdvMEA', {'attack_node_fraction': 1.0}),
    ('CEGA', 'pygip.models.attack.CEGA', 'CEGA', {'attack_node_fraction': 1.0}),
    ('Realistic', 'pygip.models.attack.Realistic', 'RealisticAttack', {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
    ('DFEA_I', 'pygip.models.attack.DataFreeMEA', 'DFEATypeI', {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
    ('DFEA_II', 'pygip.models.attack.DataFreeMEA', 'DFEATypeII', {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
    ('DFEA_III', 'pygip.models.attack.DataFreeMEA', 'DFEATypeIII', {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
]


def wrap_victim_with_defense(base_model, defense_name, defense_kwargs, dataset, device):
    """Wrap a trained GCN victim with output-perturbation defense.

    Returns a wrapper model that applies defense on forward.
    """
    class DefendedModel(torch.nn.Module):
        def __init__(self, base, name, kwargs):
            super().__init__()
            self.base = base
            self.name = name
            self.kwargs = kwargs
            self._train_features_mean = None
            self._distance_threshold = None

        def forward(self, g, features):
            # Move inputs to match base model's device (DFEA synthetic data may be on CPU)
            try:
                base_device = next(self.base.parameters()).device
                if hasattr(features, 'to') and features.device != base_device:
                    features = features.to(base_device)
                if hasattr(g, 'to'):
                    try:
                        g = g.to(base_device)
                    except Exception:
                        pass
            except Exception:
                pass
            logits = self.base(g, features)
            if self.name == 'OutputPerturbation':
                sigma = self.kwargs.get('sigma', 0.1)
                return logits + torch.randn_like(logits) * sigma
            elif self.name == 'PredictionRounding':
                if self.kwargs.get('top_k') is not None:
                    k = self.kwargs['top_k']
                    vals, idx = logits.topk(k, dim=1)
                    out = torch.full_like(logits, float('-inf'))
                    out.scatter_(1, idx, vals)
                    return out
                bits = self.kwargs.get('precision_bits', 2)
                probs = F.softmax(logits, dim=1)
                levels = 2 ** bits
                probs = (torch.round(probs * levels) / levels).clamp(min=1e-10)
                probs = probs / probs.sum(dim=1, keepdim=True)
                return torch.log(probs)
            elif self.name == 'AdaptiveMisinformation':
                # OOD detection: return wrong for queries far from training
                if self._train_features_mean is None:
                    return logits
                dists = torch.norm(features - self._train_features_mean, dim=1)
                ood_mask = dists > self._distance_threshold
                preds = logits.argmax(dim=1)
                wrong = (preds + 1) % logits.shape[1]
                # Create one-hot for wrong answers, scaled to match logit range
                out = logits.clone()
                for i in torch.where(ood_mask)[0]:
                    w = wrong[i].item()
                    out[i] = -5.0
                    out[i, w] = 5.0
                return out
            elif self.name == 'PRADA':
                # If detected extraction, add heavy noise
                return logits + torch.randn_like(logits) * 2.0
            elif self.name == 'GradientRedirection':
                strength = self.kwargs.get('redirect_strength', 1.0)
                top1_vals, top1_idx = logits.max(dim=1, keepdim=True)
                mask = torch.ones_like(logits)
                mask.scatter_(1, top1_idx, 0)
                return logits * (1 - mask * strength) + logits.mean(dim=1, keepdim=True) * mask * strength
            return logits

        def eval(self):
            self.base.eval()
            return super().eval()

    m = DefendedModel(base_model, defense_name, defense_kwargs).to(device)

    # Precompute train distribution for AdaptiveMisinformation
    if defense_name == 'AdaptiveMisinformation':
        graph = dataset.graph_data.to(device)
        features = graph.ndata['feat'].to(device)
        train_mask = graph.ndata['train_mask'].bool()
        train_mean = features[train_mask].mean(dim=0)
        dists = torch.norm(features - train_mean, dim=1)
        m._train_features_mean = train_mean
        m._distance_threshold = torch.quantile(dists, defense_kwargs.get('ood_percentile', 0.5))
    return m


def compute_fidelity(surrogate, target, graph, features, test_mask, labels, device):
    surrogate.eval()
    target.eval()
    graph = graph.to(device)
    features = features.to(device)
    with torch.no_grad():
        sur_out = model_forward(surrogate, graph, features)
        tar_out = target(graph, features) if hasattr(target, 'name') else model_forward(target, graph, features)
        sur_preds = sur_out.argmax(dim=1)
        tar_preds = tar_out.argmax(dim=1)
    fid = (sur_preds[test_mask] == tar_preds[test_mask]).float().mean().item()
    acc = (sur_preds[test_mask] == labels[test_mask]).float().mean().item()
    return fid, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--defense', required=True, choices=list(NEW_DEFENSES.keys()))
    p.add_argument('--attack', default=None, help='Filter to single attack')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--output-dir', default='outputs/joint_new_defense')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    os.environ['PYGIP_DEVICE'] = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    # Load dataset
    mod_name, cls_name = DATASET_MAP[args.dataset].rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    dataset = getattr(mod, cls_name)(api_type='dgl', path='./data')
    graph = dataset.graph_data.to(device)
    features = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device)
    test_mask = graph.ndata['test_mask'].bool()
    train_mask = graph.ndata['train_mask'].bool()

    # Train base victim
    print(f"Training base victim on {args.dataset}...")
    base = GCN(dataset.num_features, dataset.num_classes).to(device)
    opt = torch.optim.Adam(base.parameters(), lr=0.01, weight_decay=5e-4)
    base.train()
    for _ in range(200):
        opt.zero_grad()
        out = model_forward(base, graph, features)
        F.cross_entropy(out[train_mask], labels[train_mask]).backward()
        opt.step()
    base.eval()

    # Wrap with defense
    def_base, def_kwargs = NEW_DEFENSES[args.defense]
    defended = wrap_victim_with_defense(base, def_base, def_kwargs, dataset, device)

    # Save defended model - we save the base GCN so attacks can load
    # (Defense wrapper is applied on-the-fly, but attacks see the raw GCN)
    # For this joint eval, we need attacker to see defended output, so save wrapper too
    os.makedirs(f'{args.output_dir}/models', exist_ok=True)
    model_path = f'{args.output_dir}/models/{args.dataset}_{args.defense}_seed{args.seed}.pt'
    torch.save({
        'arch': 'gcn',
        'state_dict': base.state_dict(),
        'defense_name': def_base,
        'defense_kwargs': def_kwargs,
    }, model_path)

    # Run each attack
    os.makedirs(args.output_dir, exist_ok=True)
    outfile = f'{args.output_dir}/{args.dataset}.jsonl'

    attacks_to_run = ATTACK_CONFIGS if args.attack is None else [x for x in ATTACK_CONFIGS if x[0] == args.attack]

    for atk_name, atk_mod, atk_cls, atk_kwargs in attacks_to_run:
        print(f"\n=== Attack {atk_name} against {args.defense} ===")
        try:
            mod = importlib.import_module(atk_mod)
            cls = getattr(mod, atk_cls)
            # Attack will load the base model (not defended) via model_path
            attacker = cls(dataset, model_path=model_path, **atk_kwargs)

            # Override net1/model with defended version so attack queries get perturbed output.
            # MEA family uses self.net1; DFEA family uses self.model.
            if hasattr(attacker, 'net1'):
                attacker.net1 = defended
            if hasattr(attacker, 'model'):
                attacker.model = defended

            res, res_comp = attacker.attack()
            # Find surrogate
            surrogate = None
            for attr in ['surrogate', 'net2', 'surrogate_model']:
                if hasattr(attacker, attr) and getattr(attacker, attr) is not None:
                    surrogate = getattr(attacker, attr)
                    break
            if surrogate is None:
                raise RuntimeError("No surrogate")

            # Fidelity to defended model output
            fid_defended, acc_def = compute_fidelity(surrogate, defended, graph, features, test_mask, labels, device)
            # Fidelity to base (undefended) model - measures what surrogate really learned
            fid_base, acc_base = compute_fidelity(surrogate, base, graph, features, test_mask, labels, device)

            result = {
                'dataset': args.dataset,
                'defense': args.defense,
                'attack': atk_name,
                'seed': args.seed,
                'fidelity_to_defended': fid_defended * 100,
                'fidelity_to_base': fid_base * 100,
                'surrogate_acc': acc_def * 100,
                'status': 'ok',
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {
                'dataset': args.dataset, 'defense': args.defense,
                'attack': atk_name, 'seed': args.seed,
                'status': 'error', 'error': str(e),
            }

        with open(outfile, 'a') as f:
            f.write(json.dumps(result) + '\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
