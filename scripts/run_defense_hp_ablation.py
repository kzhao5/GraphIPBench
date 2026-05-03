"""Defense hyperparameter sensitivity ablation.

Tests each defense with multiple hyperparameter values to measure
sensitivity of (utility, wm_acc) to defense config.
"""
import argparse
import importlib
import json
import os
import time
import torch

DATASET_MAP = {
    'Cora': 'pygip.datasets.Cora',
    'Computers': 'pygip.datasets.Computers',
    'RomanEmpire': 'pygip.datasets.RomanEmpire',
}

# Hyperparameter grids for each defense
HP_GRIDS = {
    'BackdoorWM': [
        {'trigger_rate': 0.005}, {'trigger_rate': 0.01},
        {'trigger_rate': 0.05}, {'trigger_rate': 0.1},
    ],
    'RandomWM': [
        {'wm_node': 10}, {'wm_node': 50}, {'wm_node': 100}, {'wm_node': 200},
    ],
    'SurviveWM': [
        {'defense_ratio': 0.05}, {'defense_ratio': 0.1},
        {'defense_ratio': 0.2}, {'defense_ratio': 0.3},
    ],
    'OutputPerturbation': [
        {'sigma': 0.01}, {'sigma': 0.05}, {'sigma': 0.1},
        {'sigma': 0.2}, {'sigma': 0.5},
    ],
    'PredictionRounding': [
        {'precision_bits': 1}, {'precision_bits': 2},
        {'precision_bits': 4}, {'precision_bits': 8},
    ],
}

def load_defense_class(name):
    mapping = {
        'BackdoorWM': ('pygip.models.defense.BackdoorWM', 'BackdoorWM'),
        'RandomWM': ('pygip.models.defense.RandomWM', 'RandomWM'),
        'SurviveWM': ('pygip.models.defense.SurviveWM', 'SurviveWM'),
        'OutputPerturbation': ('pygip.models.defense.NonWatermarkDefenses', 'OutputPerturbation'),
        'PredictionRounding': ('pygip.models.defense.NonWatermarkDefenses', 'PredictionRounding'),
    }
    mod_name, cls_name = mapping[name]
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--defense', required=True, choices=list(HP_GRIDS.keys()))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--output-dir', default='outputs/defense_hp')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    os.environ['PYGIP_DEVICE'] = device

    # Load dataset
    mod_name, cls_name = DATASET_MAP[args.dataset].rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    dataset = getattr(mod, cls_name)(api_type='dgl', path='./data')

    cls = load_defense_class(args.defense)
    os.makedirs(args.output_dir, exist_ok=True)
    outfile = f'{args.output_dir}/{args.dataset}.jsonl'

    for hp in HP_GRIDS[args.defense]:
        hp_str = '_'.join(f'{k}={v}' for k,v in hp.items())
        print(f"\n=== {args.defense}({hp_str}) ===")
        try:
            t0 = time.time()
            defense = cls(dataset, **hp)
            res, res_comp = defense.defend()
            elapsed = time.time() - t0

            result = {
                'dataset': args.dataset,
                'defense': args.defense,
                'hyperparams': hp,
                'hp_str': hp_str,
                'seed': args.seed,
                'accuracy': res.get('Acc', -1) * 100,
                'f1': res.get('F1', -1) * 100,
                'wm_acc': res.get('WM Acc', -1) * 100,
                'total_time': elapsed,
                'status': 'ok',
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {
                'dataset': args.dataset, 'defense': args.defense,
                'hyperparams': hp, 'hp_str': hp_str, 'seed': args.seed,
                'status': 'error', 'error': str(e)[:200],
            }
        with open(outfile, 'a') as f:
            f.write(json.dumps(result) + '\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
