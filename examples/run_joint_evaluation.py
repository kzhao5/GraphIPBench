"""Joint attack-defense evaluation and watermark survival experiment.

This script implements two key experiments for the KDD revision:
  - Table NEW-A: Watermark survival rate on extracted surrogates
  - Table NEW-B: Fidelity drop when attacking defended models

Pipeline for each (defense, attack, dataset) combination:
  1. Train defended model (with watermark)
  2. Save defended model to disk
  3. Run attack with defended model as victim
  4. Measure surrogate fidelity (joint evaluation)
  5. Measure watermark survival on surrogate

Usage:
  conda activate graphip
  PYTHONPATH=/path/to/GraphIPBench python examples/run_joint_evaluation.py \
    --dataset Cora --seed 0 --gpu
"""

import argparse
import json
import os
import sys
import time

import torch
import dgl

from pygip.datasets import Cora, CiteSeer, PubMed, Computers, Photo, CoauthorCS, CoauthorPhysics
from pygip.datasets import RomanEmpire, AmazonRatings, OGBNArxiv
from pygip.models.nn.backbones import GCN, create_model, model_forward
from pygip.utils.metrics import AttackMetric
from pygip.evaluation.watermark_survival import WatermarkSurvivalEvaluator

# ── Dataset registry ──────────────────────────────────────────────────
DATASETS = {
    'Cora': Cora,
    'CiteSeer': CiteSeer,
    'PubMed': PubMed,
    'Computers': Computers,
    'Photo': Photo,
    'CoauthorCS': CoauthorCS,
    'CoauthorPhysics': CoauthorPhysics,
    'OGBNArxiv': OGBNArxiv,
    'RomanEmpire': RomanEmpire,
    'AmazonRatings': AmazonRatings,
}

# ── Attack registry ───────────────────────────────────────────────────
def get_attack_configs():
    """Return list of (name, class, kwargs) for all 12 attacks."""
    from pygip.models.attack.mea.MEA import (
        ModelExtractionAttack0, ModelExtractionAttack1, ModelExtractionAttack2,
        ModelExtractionAttack3, ModelExtractionAttack4, ModelExtractionAttack5,
    )
    from pygip.models.attack.AdvMEA import AdvMEA
    from pygip.models.attack.CEGA import CEGA
    from pygip.models.attack.Realistic import RealisticAttack
    from pygip.models.attack.DataFreeMEA import DFEATypeI, DFEATypeII, DFEATypeIII

    configs = [
        ('MEA0', ModelExtractionAttack0, {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
        ('MEA1', ModelExtractionAttack1, {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
        ('MEA2', ModelExtractionAttack2, {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
        ('MEA3', ModelExtractionAttack3, {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
        ('MEA4', ModelExtractionAttack4, {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
        ('MEA5', ModelExtractionAttack5, {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
        ('AdvMEA', AdvMEA, {'attack_node_fraction': 1.0}),
        ('CEGA', CEGA, {'attack_node_fraction': 1.0}),
        ('Realistic', RealisticAttack, {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
        ('DFEA_I', DFEATypeI, {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
        ('DFEA_II', DFEATypeII, {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
        ('DFEA_III', DFEATypeIII, {'attack_x_ratio': 1.0, 'attack_a_ratio': 1.0}),
    ]
    return configs

# ── Defense registry ──────────────────────────────────────────────────
def get_defense_configs():
    """Return list of (name, class, kwargs) for all 5 defenses."""
    from pygip.models.defense.BackdoorWM import BackdoorWM
    from pygip.models.defense.RandomWM import RandomWM
    from pygip.models.defense.SurviveWM import SurviveWM
    from pygip.models.defense.ImperceptibleWM import ImperceptibleWM
    from pygip.models.defense.Integrity import QueryBasedVerificationDefense as Integrity

    configs = [
        ('BackdoorWM', BackdoorWM, {'trigger_rate': 0.01, 'l': 20, 'target_label': 0}),
        ('RandomWM', RandomWM, {'wm_node': 50, 'pr': 0.1, 'pg': 0.1}),
        ('SurviveWM', SurviveWM, {'defense_ratio': 0.1}),
        ('ImperceptibleWM', ImperceptibleWM, {'defense_ratio': 0.1}),
        ('Integrity', Integrity, {'defense_ratio': 0.1}),
    ]
    return configs


def save_defended_model(defense, save_path, dataset=None, device='cpu'):
    """Save the defended model's state dict for attack loading.

    If the defense uses a non-GCN architecture (e.g., GraphSAGE for RandomWM,
    GCN_PyG for ImperceptibleWM), we distill into a standard DGL GCN so that
    all attacks can load it without modification.
    """
    # Find the defended model
    model = None
    for attr in ['net1', 'defense_model', 'watermarked_model', 'model']:
        if hasattr(defense, attr) and getattr(defense, attr) is not None:
            model = getattr(defense, attr)
            break
    if model is None:
        raise ValueError(f"Cannot find model in defense {defense.__class__.__name__}")

    # Check if it's a standard DGL GCN (compatible with attacks)
    is_dgl_gcn = (model.__class__.__name__ == 'GCN' and hasattr(model, 'backend')
                  and model.backend == 'dgl')

    if is_dgl_gcn:
        checkpoint = {'arch': 'gcn', 'state_dict': model.state_dict()}
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        return 'gcn', model

    # Non-GCN model: distill into a DGL GCN so attacks can load it
    print(f"  Distilling {model.__class__.__name__} → GCN for attack compatibility...")
    graph = dataset.graph_data.to(device)
    features = graph.ndata['feat'].to(device)

    # Get teacher predictions
    model.eval()
    with torch.no_grad():
        teacher_logits = model_forward(model, graph, features)
    soft_targets = torch.nn.functional.softmax(teacher_logits / 3.0, dim=1)

    # Train GCN student
    from pygip.models.nn.backbones import GCN as DGL_GCN
    student = DGL_GCN(dataset.num_features, dataset.num_classes).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=0.01, weight_decay=5e-4)
    train_mask = graph.ndata['train_mask'].bool()

    student.train()
    for _ in range(200):
        opt.zero_grad()
        s_logits = model_forward(student, graph, features)
        # Combined: KD on all nodes + CE on train nodes
        loss_kd = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(s_logits / 3.0, dim=1),
            soft_targets, reduction='batchmean') * 9.0
        loss_ce = torch.nn.functional.cross_entropy(
            s_logits[train_mask], teacher_logits.argmax(dim=1)[train_mask])
        (0.7 * loss_kd + 0.3 * loss_ce).backward()
        opt.step()
    student.eval()

    # Verify distillation quality
    with torch.no_grad():
        s_preds = model_forward(student, graph, features).argmax(dim=1)
        t_preds = teacher_logits.argmax(dim=1)
        test_mask = graph.ndata['test_mask'].bool()
        fid = (s_preds[test_mask] == t_preds[test_mask]).float().mean().item()
        print(f"  Distillation fidelity: {fid:.4f}")

    checkpoint = {'arch': 'gcn', 'state_dict': student.state_dict()}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    return 'gcn', student


def compute_fidelity(surrogate_model, target_model, graph, features, test_mask, device):
    """Compute fidelity between surrogate and target on test nodes."""
    surrogate_model.eval()
    target_model.eval()
    graph = graph.to(device)
    features = features.to(device)

    with torch.no_grad():
        sur_out = model_forward(surrogate_model, graph, features)
        tar_out = model_forward(target_model, graph, features)
        sur_preds = sur_out.argmax(dim=1)[test_mask]
        tar_preds = tar_out.argmax(dim=1)[test_mask]
        fidelity = (sur_preds == tar_preds).float().mean().item()

    return fidelity


def run_single_experiment(dataset, defense_name, defense_cls, defense_kwargs,
                          attack_name, attack_cls, attack_kwargs,
                          seed, device, output_dir):
    """Run one (defense, attack) combination and return results."""
    torch.manual_seed(seed)

    ds_name = dataset.__class__.__name__
    print(f"\n{'='*60}")
    print(f"Dataset={ds_name}, Defense={defense_name}, Attack={attack_name}, Seed={seed}")
    print(f"{'='*60}")

    # ── Step 1: Train defended model ──
    print(f"[1/4] Training {defense_name}...")
    defense = defense_cls(dataset, **defense_kwargs)
    try:
        res_def, res_comp_def = defense.defend()
    except Exception as e:
        print(f"  FAILED: {e}")
        return None

    # ── Step 2: Save defended model ──
    model_path = os.path.join(output_dir, 'models',
                               f'{ds_name}_{defense_name}_seed{seed}.pt')
    arch, defended_model = save_defended_model(defense, model_path, dataset=dataset, device=device)
    print(f"  Defended model saved ({arch}): {model_path}")

    # ── Step 3: Run attack with defended model as victim ──
    print(f"[2/4] Running {attack_name} against defended model...")
    try:
        attacker = attack_cls(dataset, model_path=model_path, **attack_kwargs)
        res_atk, res_comp_atk = attacker.attack()
    except Exception as e:
        print(f"  FAILED: {e}")
        return None

    # Get the surrogate model (different attacks store it differently)
    surrogate = None
    for attr in ['surrogate', 'net2', 'surrogate_model']:
        if hasattr(attacker, attr) and getattr(attacker, attr) is not None:
            surrogate = getattr(attacker, attr)
            break
    if surrogate is None:
        print("  FAILED: Cannot find surrogate model")
        return None

    # ── Step 4a: Compute fidelity metrics (joint evaluation) ──
    print(f"[3/4] Computing fidelity metrics...")
    graph = dataset.graph_data.to(device)
    features = graph.ndata['feat'].to(device)
    test_mask = graph.ndata['test_mask'].bool()

    fidelity_to_defended = compute_fidelity(
        surrogate, defended_model, graph, features, test_mask, device)

    # ── Step 4b: Evaluate watermark survival ──
    print(f"[4/4] Evaluating watermark survival...")
    evaluator = WatermarkSurvivalEvaluator(device=device)
    try:
        wm_result = evaluator.evaluate(defense, surrogate)
        wm_acc = wm_result['wm_acc']
    except Exception as e:
        print(f"  Watermark eval failed: {e}")
        wm_acc = -1.0
        wm_result = {'wm_acc': -1.0, 'details': {'error': str(e)}}

    # ── Compile results ──
    result = {
        'dataset': ds_name,
        'defense': defense_name,
        'attack': attack_name,
        'seed': seed,
        'defense_arch': arch,
        # Joint evaluation metrics
        'surrogate_fidelity_to_defended': fidelity_to_defended,
        'surrogate_acc': (res_atk.compute() if hasattr(res_atk, 'compute') else res_atk if isinstance(res_atk, dict) else {}).get('Acc', -1),
        # Watermark survival
        'wm_acc_on_surrogate': wm_acc,
        'wm_details': wm_result.get('details', {}),
    }

    print(f"  Results: fidelity={fidelity_to_defended:.4f}, wm_survival={wm_acc:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Joint attack-defense evaluation')
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=list(DATASETS.keys()),
                        help='Dataset to evaluate on')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--output-dir', type=str, default='outputs/joint_eval',
                        help='Output directory')
    parser.add_argument('--defense', type=str, default=None,
                        help='Run specific defense only (e.g. BackdoorWM)')
    parser.add_argument('--attack', type=str, default=None,
                        help='Run specific attack only (e.g. MEA0)')
    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = DATASETS[args.dataset](api_type='dgl')
    print(f"  {dataset}")

    # Get configs
    defense_configs = get_defense_configs()
    attack_configs = get_attack_configs()

    # Filter if specified
    if args.defense:
        defense_configs = [(n, c, k) for n, c, k in defense_configs if n == args.defense]
    if args.attack:
        attack_configs = [(n, c, k) for n, c, k in attack_configs if n == args.attack]

    # Run experiments
    all_results = []
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'{args.dataset}_seed{args.seed}.jsonl')

    for def_name, def_cls, def_kwargs in defense_configs:
        for atk_name, atk_cls, atk_kwargs in attack_configs:
            result = run_single_experiment(
                dataset, def_name, def_cls, def_kwargs,
                atk_name, atk_cls, atk_kwargs,
                args.seed, device, args.output_dir)

            if result is not None:
                all_results.append(result)
                # Append to JSONL file
                with open(output_file, 'a') as f:
                    f.write(json.dumps(result) + '\n')

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY: {args.dataset}, seed={args.seed}")
    print(f"{'='*80}")
    print(f"{'Attack':<12} {'Defense':<16} {'Fidelity':>10} {'WM Survival':>12}")
    print(f"{'-'*12} {'-'*16} {'-'*10} {'-'*12}")
    for r in all_results:
        print(f"{r['attack']:<12} {r['defense']:<16} "
              f"{r['surrogate_fidelity_to_defended']:>10.4f} "
              f"{r['wm_acc_on_surrogate']:>12.4f}")

    print(f"\nResults saved to: {output_file}")
    print(f"Total experiments: {len(all_results)}")


if __name__ == '__main__':
    main()
