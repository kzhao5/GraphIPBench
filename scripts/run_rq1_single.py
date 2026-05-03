"""Run RQ1 (attack extraction) for a single (dataset, attack, regime, budget, seed) combination.

Outputs one JSON line to stdout with all metrics.
Designed to be called from Slurm job arrays.

Usage:
  python scripts/run_rq1_single.py --dataset RomanEmpire --attack MEA0 \
    --regime both --budget 0.25 --seed 0 --gpu --output-dir outputs/RQ1_new
"""

import argparse
import json
import os
import sys
import time
import torch

# Attack registry
ATTACK_CONFIGS = {
    'MEA0': ('pygip.models.attack.mea.MEA', 'ModelExtractionAttack0'),
    'MEA1': ('pygip.models.attack.mea.MEA', 'ModelExtractionAttack1'),
    'MEA2': ('pygip.models.attack.mea.MEA', 'ModelExtractionAttack2'),
    'MEA3': ('pygip.models.attack.mea.MEA', 'ModelExtractionAttack3'),
    'MEA4': ('pygip.models.attack.mea.MEA', 'ModelExtractionAttack4'),
    'MEA5': ('pygip.models.attack.mea.MEA', 'ModelExtractionAttack5'),
    'AdvMEA': ('pygip.models.attack.AdvMEA', 'AdvMEA'),
    'CEGA': ('pygip.models.attack.CEGA', 'CEGA'),
    'Realistic': ('pygip.models.attack.Realistic', 'RealisticAttack'),
    'DFEA_I': ('pygip.models.attack.DataFreeMEA', 'DFEATypeI'),
    'DFEA_II': ('pygip.models.attack.DataFreeMEA', 'DFEATypeII'),
    'DFEA_III': ('pygip.models.attack.DataFreeMEA', 'DFEATypeIII'),
}

DATASET_MAP = {
    'OGBNArxiv': 'pygip.datasets.OGBNArxiv',
    'RomanEmpire': 'pygip.datasets.RomanEmpire',
    'AmazonRatings': 'pygip.datasets.AmazonRatings',
    # existing datasets
    'Cora': 'pygip.datasets.Cora',
    'CiteSeer': 'pygip.datasets.CiteSeer',
    'PubMed': 'pygip.datasets.PubMed',
    'Computers': 'pygip.datasets.Computers',
    'Photo': 'pygip.datasets.Photo',
    'CoauthorCS': 'pygip.datasets.CoauthorCS',
    'CoauthorPhysics': 'pygip.datasets.CoauthorPhysics',
}


def get_regime_ratios(regime, budget):
    """Convert regime name + budget to (attack_x_ratio, attack_a_ratio)."""
    if regime == 'both':
        return budget, budget
    elif regime == 'x_only':
        return budget, 0.0
    elif regime == 'a_only':
        return 0.0, budget
    elif regime == 'data_free':
        return budget, budget  # data-free attacks ignore these but need non-zero
    else:
        raise ValueError(f"Unknown regime: {regime}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--attack', type=str, required=True)
    parser.add_argument('--regime', type=str, default='both',
                        choices=['both', 'x_only', 'a_only', 'data_free'])
    parser.add_argument('--budget', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--output-dir', type=str, default='outputs/RQ1_new')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    os.environ['PYGIP_DEVICE'] = device

    # Load dataset
    import importlib
    ds_path = DATASET_MAP[args.dataset]
    mod_name, cls_name = ds_path.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    ds_cls = getattr(mod, cls_name)
    dataset = ds_cls(api_type='dgl')

    # Load attack
    atk_mod_name, atk_cls_name = ATTACK_CONFIGS[args.attack]
    atk_mod = importlib.import_module(atk_mod_name)
    atk_cls = getattr(atk_mod, atk_cls_name)

    # Build kwargs based on attack type
    x_ratio, a_ratio = get_regime_ratios(args.regime, args.budget)

    if args.attack in ('AdvMEA', 'CEGA'):
        kwargs = {'attack_node_fraction': args.budget}
    elif args.attack == 'Realistic':
        kwargs = {'attack_x_ratio': x_ratio, 'attack_a_ratio': a_ratio}
    else:
        kwargs = {'attack_x_ratio': x_ratio, 'attack_a_ratio': a_ratio}

    # Run attack
    t0 = time.time()
    try:
        attacker = atk_cls(dataset, **kwargs)
        res, res_comp = attacker.attack()
        elapsed = time.time() - t0

        metrics = res.compute() if hasattr(res, 'compute') else (res if isinstance(res, dict) else {})
        comp_metrics = res_comp.compute() if hasattr(res_comp, 'compute') else (res_comp if isinstance(res_comp, dict) else {})

        result = {
            'track': 'RQ1',
            'dataset': args.dataset,
            'attack': args.attack,
            'regime': args.regime,
            'budget': args.budget,
            'seed': args.seed,
            'device': device,
            'accuracy': metrics.get('Acc', -1),
            'f1': metrics.get('F1', -1),
            'fidelity': metrics.get('Fidelity', -1),
            'precision': metrics.get('Precision', -1),
            'recall': metrics.get('Recall', -1),
            'total_time': elapsed,
            **{k: v for k, v in comp_metrics.items() if isinstance(v, (int, float))},
            'status': 'ok',
        }
    except Exception as e:
        result = {
            'track': 'RQ1',
            'dataset': args.dataset,
            'attack': args.attack,
            'regime': args.regime,
            'budget': args.budget,
            'seed': args.seed,
            'status': 'error',
            'error': str(e),
        }

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, f'{args.dataset}.jsonl')
    with open(outfile, 'a') as f:
        f.write(json.dumps(result) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
