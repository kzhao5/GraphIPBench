"""Run RQ2/RQ3 defense evaluation on a single dataset (all defenses × 3 seeds).

Usage:
  python scripts/run_defense_single.py --dataset RomanEmpire --gpu --output-dir outputs/RQ2_RQ3_new
"""

import argparse
import json
import os
import time
import torch
import importlib

DATASET_MAP = {
    'OGBNArxiv': 'pygip.datasets.OGBNArxiv',
    'RomanEmpire': 'pygip.datasets.RomanEmpire',
    'AmazonRatings': 'pygip.datasets.AmazonRatings',
    'Cora': 'pygip.datasets.Cora',
    'Computers': 'pygip.datasets.Computers',
}

DEFENSE_CONFIGS = [
    ('BackdoorWM', 'pygip.models.defense.BackdoorWM', 'BackdoorWM',
     {'trigger_rate': 0.01, 'l': 20, 'target_label': 0}),
    ('RandomWM', 'pygip.models.defense.RandomWM', 'RandomWM',
     {'wm_node': 50, 'pr': 0.1, 'pg': 0.1}),
    ('SurviveWM', 'pygip.models.defense.SurviveWM', 'SurviveWM',
     {'defense_ratio': 0.1}),
    ('ImperceptibleWM', 'pygip.models.defense.ImperceptibleWM', 'ImperceptibleWM',
     {'defense_ratio': 0.1}),
    ('Integrity', 'pygip.models.defense.Integrity', 'QueryBasedVerificationDefense',
     {'defense_ratio': 0.1}),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--defense', type=str, default=None,
                        help='If set, only run this defense.')
    parser.add_argument('--seed', type=int, default=None,
                        help='If set, only run this seed.')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--output-dir', type=str, default='outputs/RQ2_RQ3_new')
    args = parser.parse_args()

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    os.environ['PYGIP_DEVICE'] = device

    # Load dataset
    ds_path = DATASET_MAP[args.dataset]
    mod_name, cls_name = ds_path.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    ds_cls = getattr(mod, cls_name)
    dataset = ds_cls(api_type='dgl')
    print(f"Dataset: {dataset}")

    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, f'{args.dataset}.jsonl')

    for def_name, def_mod_name, def_cls_name, def_kwargs in DEFENSE_CONFIGS:
        if args.defense is not None and args.defense != def_name:
            continue
        def_mod = importlib.import_module(def_mod_name)
        def_cls = getattr(def_mod, def_cls_name)

        seeds = [args.seed] if args.seed is not None else [0, 1, 2]
        for seed in seeds:
            torch.manual_seed(seed)
            print(f"\n{'='*60}")
            print(f"Defense={def_name}, Seed={seed}")

            t0 = time.time()
            try:
                defense = def_cls(dataset, **def_kwargs)
                res, res_comp = defense.defend()
                elapsed = time.time() - t0

                metrics = res.compute() if hasattr(res, 'compute') else (res if isinstance(res, dict) else {})
                comp_metrics = res_comp.compute() if hasattr(res_comp, 'compute') else (res_comp if isinstance(res_comp, dict) else {})

                result = {
                    'track': 'RQ2_RQ3',
                    'dataset': args.dataset,
                    'defense': def_name,
                    'seed': seed,
                    'accuracy': metrics.get('Acc', -1),
                    'f1': metrics.get('F1', -1),
                    'fidelity': metrics.get('Fidelity', -1),
                    'wm_acc': metrics.get('WM Acc', -1),
                    'total_time': elapsed,
                    **{k: v for k, v in comp_metrics.items() if isinstance(v, (int, float))},
                    'status': 'ok',
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                result = {
                    'track': 'RQ2_RQ3',
                    'dataset': args.dataset,
                    'defense': def_name,
                    'seed': seed,
                    'status': 'error',
                    'error': str(e),
                }

            with open(outfile, 'a') as f:
                f.write(json.dumps(result) + '\n')
            print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
