"""Generate additional appendix-only statistical figures (set 2).

Outputs to outputs/figures_v2/:
  figM_attack_budget_panels.pdf       — 12 small budget curves (one per attack), all 10 datasets
  figN_regime_drop_bars.pdf           — bar chart: fidelity drop per regime per attack
  figO_attack_time_distribution.pdf   — log-scale boxplot of attack time per attack
  figP_defense_acc_drop_per_dataset.pdf — bar chart: utility drop per defense per dataset
  figQ_attack_correlation_matrix.pdf  — Pearson correlation between attack fidelity vectors
  figR_dataset_difficulty_radar.pdf   — radar showing per-dataset ease of extraction (5 metrics)

Run from project root:
    python3 scripts/regen_appendix_figures2.py
"""
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

OUT = 'outputs/figures_v2'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

ATTACKS_12 = ['MEA0','MEA1','MEA2','MEA3','MEA4','MEA5','AdvMEA','CEGA','Realistic',
              'DFEA_I','DFEA_II','DFEA_III']
DATASETS_10 = ['Cora','CiteSeer','PubMed','Computers','Photo','CoauthorCS',
               'CoauthorPhysics','OGBNArxiv','RomanEmpire','AmazonRatings']
DS_SHORT = {'Cora':'Cora','CiteSeer':'CiteS','PubMed':'PubMed','Computers':'Comp',
            'Photo':'Photo','CoauthorCS':'CS','CoauthorPhysics':'Phys',
            'OGBNArxiv':'OGBN-A','RomanEmpire':'RomanE','AmazonRatings':'AmazR'}

DS_PALETTE = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b',
              '#e377c2','#7f7f7f','#bcbd22','#17becf']


def load_rq1():
    """Load all RQ1 records: original 7 datasets + 3 new datasets."""
    out = []
    for f in glob.glob('outputs/RQ1_efficiency/*.jsonl') + glob.glob('outputs/RQ1_new/*.jsonl'):
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if r.get('status') == 'ok':
                        out.append(r)
                except:
                    pass
    return out


# ============================================================
# Fig M — 12 small budget curves (one per attack), all 10 datasets
# ============================================================
def figM(records):
    # Restrict to regime='both'
    rs = [r for r in records if r.get('regime') == 'both']
    # (attack, dataset, budget) -> [fid values across seeds]
    by = defaultdict(list)
    for r in rs:
        by[(r['attack'], r['dataset'], r['budget'])].append(r['fidelity'])

    fig, axes = plt.subplots(3, 4, figsize=(15, 9.0), sharex=True)
    axes = axes.flatten()
    budgets = [0.05, 0.1, 0.25, 0.5, 1.0]

    for idx, atk in enumerate(ATTACKS_12):
        ax = axes[idx]
        for di, ds in enumerate(DATASETS_10):
            ys = []
            for b in budgets:
                v = by.get((atk, ds, b), [])
                ys.append(np.mean(v) * 100 if v else np.nan)
            ax.plot(budgets, ys, marker='o', markersize=4, linewidth=1.4,
                    color=DS_PALETTE[di], label=DS_SHORT[ds])
        ax.set_title(atk, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.set_xticks(budgets)
        ax.tick_params(axis='x', labelsize=8, rotation=25)
        ax.grid(True, linestyle=':', alpha=0.5)
        if idx % 4 == 0:
            ax.set_ylabel('Fidelity (%)')
        if idx >= 8:
            ax.set_xlabel('Budget ($\\times$)')

    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=10, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(f'{OUT}/figM_attack_budget_panels.pdf', bbox_inches='tight')
    plt.close()
    print(f'  Saved {OUT}/figM_attack_budget_panels.pdf')


# ============================================================
# Fig N — Regime drop bars: average fidelity drop vs `both` per regime per attack
# ============================================================
def figN(records):
    # Aggregate: (attack, regime) -> mean fidelity across all (dataset, budget, seed)
    by = defaultdict(list)
    for r in records:
        by[(r['attack'], r['regime'])].append(r['fidelity'])
    means = {k: np.mean(v) * 100 for k, v in by.items() if v}

    regimes = ['x_only', 'a_only', 'data_free']
    regime_label = {'x_only':'features only', 'a_only':'structure only', 'data_free':'data-free'}
    regime_color = {'x_only':'#1f77b4', 'a_only':'#2ca02c', 'data_free':'#d62728'}
    drops = {reg: [] for reg in regimes}
    for atk in ATTACKS_12:
        base = means.get((atk, 'both'), 0)
        for reg in regimes:
            v = means.get((atk, reg), 0)
            drops[reg].append(base - v)  # positive value = drop

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(ATTACKS_12))
    w = 0.27
    for i, reg in enumerate(regimes):
        ax.bar(x + (i - 1) * w, drops[reg], w, label=regime_label[reg],
               color=regime_color[reg], edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(ATTACKS_12, rotation=25, ha='right')
    ax.set_ylabel('Mean fidelity drop vs.\\ \\texttt{both} (pp)')
    ax.set_title('Regime sensitivity: fidelity loss when one input modality is removed')
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.legend(title='Regime', loc='upper left', frameon=True, ncol=3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/figN_regime_drop_bars.pdf')
    plt.close()
    print(f'  Saved {OUT}/figN_regime_drop_bars.pdf')


# ============================================================
# Fig O — Attack time distribution per attack (log scale)
# ============================================================
def figO(records):
    # Aggregate attack_time across all (dataset, regime, budget, seed)
    by = defaultdict(list)
    for r in records:
        if 'attack_time' in r and r['attack_time'] > 0:
            by[r['attack']].append(r['attack_time'] / 60.0)  # minutes

    fig, ax = plt.subplots(figsize=(10, 4.5))
    data = [by[a] for a in ATTACKS_12]
    bp = ax.boxplot(data, positions=range(len(ATTACKS_12)),
                    widths=0.62, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='.', markersize=2, alpha=0.4))
    palette = plt.cm.tab20(np.linspace(0, 1, len(ATTACKS_12)))
    for patch, color in zip(bp['boxes'], palette):
        patch.set_facecolor(color); patch.set_alpha(0.65); patch.set_edgecolor('black')
    for med in bp['medians']:
        med.set_color('black'); med.set_linewidth(1.5)
    ax.set_yscale('log')
    ax.set_xticks(range(len(ATTACKS_12)))
    ax.set_xticklabels(ATTACKS_12, rotation=25, ha='right')
    ax.set_ylabel('Attack wall-clock time (min, log scale)')
    ax.set_title('Per-attack runtime distribution across all (dataset, regime, budget, seed) runs')
    ax.grid(True, axis='y', which='both', linestyle=':', alpha=0.45)
    plt.tight_layout()
    plt.savefig(f'{OUT}/figO_attack_time_distribution.pdf')
    plt.close()
    print(f'  Saved {OUT}/figO_attack_time_distribution.pdf')


# ============================================================
# Fig P — Per-dataset utility loss of the 7 information-limiting defenses
# ============================================================
def figP():
    """Bar chart: utility drop = undefended GCN_16 acc - defended-model acc.
    Defended values come from Table 40 / app_rq2_new_defenses_full (in the paper).
    Baseline values come from outputs/baseline_utility/."""
    # Defended-model accuracy from Table 40 (mean over 3 seeds), 7 defenses x 10 datasets
    defenses = ['OP_low','OP_high','PR_2bit','PR_top1','PRADA','AdaptMisinfo','GradRedir']
    defense_short = ['OPlo','OPhi','PR2b','PRtop1','PRADA','AdaptM','GradR']
    # rows = datasets in DATASETS_10 order
    defended = np.array([
        [79.4,79.2,73.3,79.6,40.2,41.0,79.8],   # Cora
        [67.6,66.3,53.9,68.8,69.3,39.8,68.4],   # CiteSeer
        [77.9,75.9,77.6,78.2,78.0,44.1,78.3],   # PubMed
        [44.0,37.6,36.1,34.8,46.0,28.0,52.4],   # Computers
        [89.1,90.7,90.4,95.5,87.0,46.3,66.6],   # Photo
        [87.8,88.2,87.5,88.1,75.0,52.4,88.2],   # CoauthorCS
        [89.4,89.1,90.2,89.5,83.4,59.0,89.7],   # CoauthorPhysics
        [37.7,37.8,30.2,39.5,37.0,19.9,38.2],   # OGBNArxiv
        [42.7,40.6,35.2,42.7,19.7,22.5,42.5],   # RomanEmpire
        [42.0,41.2,39.4,41.6,41.8,33.9,41.7],   # AmazonRatings
    ])
    # Undefended GCN_16 baseline (from outputs/baseline_utility/)
    baseline = np.array([79.40,67.83,78.00,44.57,90.07,87.40,89.33,37.67,42.83,41.71])
    # Utility drop = baseline - defended (positive = defense costs accuracy)
    drop = baseline[:, None] - defended  # shape (10, 7)

    palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#17becf']
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    x = np.arange(len(DATASETS_10))
    w = 0.115
    for fi, d in enumerate(defenses):
        offset = (fi - 3) * w
        ax.bar(x + offset, drop[:, fi], w, label=defense_short[fi],
               color=palette[fi], edgecolor='black', linewidth=0.4)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([DS_SHORT[d] for d in DATASETS_10], rotation=20, ha='right')
    ax.set_ylabel('Utility loss (pp): undefended acc $-$ defended acc')
    ax.set_title('Per-dataset utility cost of the seven information-limiting and query-detection defenses')
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.legend(title='Defense', loc='upper left', frameon=True, ncol=7,
              fontsize=9, columnspacing=0.6)
    ax.set_ylim(-12, 50)
    plt.tight_layout()
    plt.savefig(f'{OUT}/figP_defense_acc_drop_per_dataset.pdf')
    plt.close()
    print(f'  Saved {OUT}/figP_defense_acc_drop_per_dataset.pdf')


# ============================================================
# Fig Q — Attack correlation matrix
# ============================================================
def figQ(records):
    # For each attack, build a 50-dim vector of fidelity = (10 datasets x 5 budgets) at regime=both
    rs = [r for r in records if r.get('regime') == 'both']
    by = defaultdict(list)
    for r in rs:
        by[(r['attack'], r['dataset'], r['budget'])].append(r['fidelity'])

    budgets = [0.05, 0.1, 0.25, 0.5, 1.0]
    vec = {}
    for atk in ATTACKS_12:
        v = []
        for ds in DATASETS_10:
            for b in budgets:
                vals = by.get((atk, ds, b), [])
                v.append(np.mean(vals) if vals else np.nan)
        vec[atk] = np.array(v)

    n = len(ATTACKS_12)
    corr = np.zeros((n, n))
    for i, a in enumerate(ATTACKS_12):
        for j, b in enumerate(ATTACKS_12):
            x, y = vec[a], vec[b]
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= 5:
                corr[i, j] = np.corrcoef(x[mask], y[mask])[0, 1]
            else:
                corr[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(n)); ax.set_xticklabels(ATTACKS_12, rotation=35, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(ATTACKS_12)
    for i in range(n):
        for j in range(n):
            v = corr[i, j]
            if not np.isnan(v):
                color = 'white' if abs(v) > 0.55 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=7, color=color)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cb.set_label('Pearson correlation of fidelity vectors')
    ax.set_title('Attack-attack correlation across (dataset, budget) profiles', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUT}/figQ_attack_correlation_matrix.pdf')
    plt.close()
    print(f'  Saved {OUT}/figQ_attack_correlation_matrix.pdf')


# ============================================================
# Fig R — Per-attack mean fidelity bar grouped by 4 regimes
# ============================================================
def figR(records):
    # (attack, regime) -> mean fidelity
    by = defaultdict(list)
    for r in records:
        by[(r['attack'], r['regime'])].append(r['fidelity'] * 100)
    means = {k: np.mean(v) for k, v in by.items() if v}
    stds = {k: np.std(v) for k, v in by.items() if v}

    regimes = ['both', 'x_only', 'a_only', 'data_free']
    rlabels = ['both', 'features only', 'structure only', 'data-free']
    rcolors = ['#2c7fb8', '#41b6c4', '#a1dab4', '#ffd966']

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(ATTACKS_12))
    w = 0.21
    for i, reg in enumerate(regimes):
        ms = [means.get((a, reg), 0) for a in ATTACKS_12]
        ss = [stds.get((a, reg), 0) for a in ATTACKS_12]
        ax.bar(x + (i - 1.5) * w, ms, w, yerr=ss, label=rlabels[i],
               color=rcolors[i], edgecolor='black', linewidth=0.5,
               error_kw=dict(elinewidth=0.7, capsize=1.6))
    ax.set_xticks(x); ax.set_xticklabels(ATTACKS_12, rotation=25, ha='right')
    ax.set_ylabel('Mean fidelity (%)')
    ax.set_ylim(0, 105)
    ax.set_title('Per-attack fidelity by regime, averaged over all datasets, budgets, and seeds')
    ax.grid(True, axis='y', linestyle=':', alpha=0.45)
    ax.legend(title='Regime', loc='upper right', frameon=True, ncol=4)
    plt.tight_layout()
    plt.savefig(f'{OUT}/figR_attack_fidelity_by_regime.pdf')
    plt.close()
    print(f'  Saved {OUT}/figR_attack_fidelity_by_regime.pdf')


if __name__ == '__main__':
    print('Generating set-2 appendix figures...')
    rec = load_rq1()
    print(f'  loaded {len(rec)} RQ1 records (original 7 + new 3 datasets)')
    figM(rec)
    figN(rec)
    figO(rec)
    figP()
    figQ(rec)
    figR(rec)
    print(f'All set-2 figures saved to {OUT}/')
