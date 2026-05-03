"""Run new non-watermark defenses (RQ2/RQ3 + Joint Eval) on a dataset.

Usage:
  python scripts/run_new_defense.py --dataset Cora --mode defense --seed 0 --gpu
  python scripts/run_new_defense.py --dataset Cora --mode joint --seed 0 --defense OutputPerturbation --gpu
"""
import argparse
import importlib
import json
import os
import time
import torch

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

NEW_DEFENSES = [
    ('OutputPerturbation_low', 'OutputPerturbation', {'sigma': 0.05}),
    ('OutputPerturbation_high', 'OutputPerturbation', {'sigma': 0.2}),
    ('PredictionRounding_2bit', 'PredictionRounding', {'precision_bits': 2}),
    ('PredictionRounding_top1', 'PredictionRounding', {'precision_bits': 2, 'top_k': 1}),
    ('PRADA', 'PRADA', {'threshold': 0.85}),
    ('AdaptiveMisinformation', 'AdaptiveMisinformation', {'ood_percentile': 0.5}),
    ('GradientRedirection', 'GradientRedirection', {'redirect_strength': 1.0}),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--defense', default=None, help='filter to one defense')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--output-dir', default='outputs/new_defense')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    os.environ['PYGIP_DEVICE'] = device

    # Load dataset
    mod_name, cls_name = DATASET_MAP[args.dataset].rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    dataset = getattr(mod, cls_name)(api_type='dgl', path='./data')

    # Import new defenses
    from pygip.models.defense.NonWatermarkDefenses import (
        OutputPerturbation, PredictionRounding, PRADA,
        AdaptiveMisinformation, GradientRedirection
    )
    DEFENSE_CLASSES = {
        'OutputPerturbation': OutputPerturbation,
        'PredictionRounding': PredictionRounding,
        'PRADA': PRADA,
        'AdaptiveMisinformation': AdaptiveMisinformation,
        'GradientRedirection': GradientRedirection,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, f'{args.dataset}.jsonl')

    for name, cls_key, kwargs in NEW_DEFENSES:
        if args.defense and args.defense not in name:
            continue
        cls = DEFENSE_CLASSES[cls_key]
        try:
            t0 = time.time()
            defense = cls(dataset, **kwargs)
            res, res_comp = defense.defend()
            elapsed = time.time() - t0
            result = {
                'dataset': args.dataset,
                'defense': name,
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
                'dataset': args.dataset,
                'defense': name,
                'seed': args.seed,
                'status': 'error',
                'error': str(e),
            }
        with open(outfile, 'a') as f:
            f.write(json.dumps(result) + '\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
