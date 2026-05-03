"""Generate appendix-only statistical figures.

Outputs to outputs/figures_v2/:
  figH_wm_survival_violin.pdf      — violin plot, survival distributions per defense
  figI_homophily_vs_survival.pdf   — scatter, edge homophily vs mean WM survival per (dataset, defense)
  figJ_fidelity_ecdf.pdf           — ECDF of surrogate fidelity per defense
  figK_per_dataset_heatmap_grid.pdf — small-heatmap grid of joint fidelity per dataset
  figL_acc_vs_fid_density.pdf      — 2D density / hexbin of surrogate accuracy vs fidelity

Run from project root:
    python3 scripts/regen_appendix_figures.py
"""
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

OUT = 'outputs/figures_v2'
os.makedirs(OUT, exist_ok=True)

# Use serif fonts to match other paper figures
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

ATTACKS_12 = ['MEA0','MEA1','MEA2','MEA3','MEA4','MEA5','AdvMEA','CEGA','Realistic',
              'DFEA_I','DFEA_II','DFEA_III']
WM_DEFS = ['BackdoorWM','SurviveWM','Integrity','RandomWM','ImperceptibleWM']
WM_SHORT = ['Back','Surv','Integ','Rand','Imp']
DATASETS_10 = ['Cora','CiteSeer','PubMed','Computers','Photo','CoauthorCS',
               'CoauthorPhysics','OGBNArxiv','RomanEmpire','AmazonRatings']
DS_SHORT = {'Cora':'Cora','CiteSeer':'CiteS.','PubMed':'PubMed','Computers':'Comp.',
            'Photo':'Photo','CoauthorCS':'CS','CoauthorPhysics':'Phys.',
            'OGBNArxiv':'OGBN-A','RomanEmpire':'RomanE','AmazonRatings':'AmazR'}


def load_records():
    out = []
    for f in glob.glob('outputs/joint_eval_v2/*.jsonl'):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                # Filter out negative wm_acc (imperceptibleWM stores -1 for "missing")
                if r.get('wm_acc_on_surrogate') is not None and r['wm_acc_on_surrogate'] < 0:
                    r['wm_acc_on_surrogate'] = None
                out.append(r)
    return out


def load_structure():
    with open('outputs/structure_analysis/graph_properties.json') as f:
        return json.load(f)


# ============================================================
# Fig H — Violin plot of WM survival per defense
# ============================================================
def figH(records):
    bydef = defaultdict(list)
    for r in records:
        if r['defense'] in WM_DEFS and r.get('wm_acc_on_surrogate') is not None:
            bydef[r['defense']].append(r['wm_acc_on_surrogate'] * 100.0)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    data = [bydef[d] for d in WM_DEFS]
    parts = ax.violinplot(data, positions=range(len(WM_DEFS)),
                          showmeans=False, showmedians=True, widths=0.85)
    palette = ['#1b9e77', '#7570b3', '#d95f02', '#e7298a', '#66a61e']
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(palette[i])
        body.set_alpha(0.55)
        body.set_edgecolor('black')
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.3)
    # Overlay scatter of group means
    means = [np.mean(d) if d else 0 for d in data]
    ax.scatter(range(len(WM_DEFS)), means, marker='D', color='white',
               edgecolor='black', s=55, zorder=3, label='mean')
    ax.set_xticks(range(len(WM_DEFS)))
    ax.set_xticklabels(WM_SHORT)
    ax.set_ylabel('Watermark survival on surrogate (%)')
    ax.set_xlabel('Defense')
    ax.set_title('Survival distribution across all (dataset, attack, seed) runs')
    ax.set_ylim(-5, 105)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    plt.savefig(f'{OUT}/figH_wm_survival_violin.pdf')
    plt.close()
    print(f'  Saved {OUT}/figH_wm_survival_violin.pdf')


# ============================================================
# Fig I — Scatter: edge homophily vs mean WM survival
# ============================================================
def figI(records, struct):
    pts = defaultdict(list)  # defense -> list of (homophily, mean_survival)
    for ds in DATASETS_10:
        if ds not in struct:
            continue
        h = struct[ds]['edge_homophily']
        for d in WM_DEFS:
            vals = [r['wm_acc_on_surrogate'] * 100.0 for r in records
                    if r['dataset'] == ds and r['defense'] == d
                    and r.get('wm_acc_on_surrogate') is not None]
            if vals:
                pts[d].append((h, float(np.mean(vals)), ds))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    palette = ['#1b9e77', '#7570b3', '#d95f02', '#e7298a', '#66a61e']
    markers = ['o', 's', 'D', '^', 'v']
    for i, d in enumerate(WM_DEFS):
        if not pts[d]:
            continue
        xs = [p[0] for p in pts[d]]
        ys = [p[1] for p in pts[d]]
        ax.scatter(xs, ys, marker=markers[i], color=palette[i], s=80,
                   edgecolor='black', linewidth=0.8, label=WM_SHORT[i], alpha=0.85)
        # Trend line per defense
        if len(xs) >= 3:
            slope, intercept = np.polyfit(xs, ys, 1)
            xline = np.linspace(min(xs), max(xs), 50)
            ax.plot(xline, slope * xline + intercept, color=palette[i],
                    linestyle='--', linewidth=1.0, alpha=0.65)
    ax.set_xlabel('Edge homophily of dataset')
    ax.set_ylabel('Mean watermark survival (%)')
    ax.set_title('Watermark survival vs. graph homophily (per defense)')
    ax.set_ylim(-3, 100)
    ax.grid(True, linestyle=':', alpha=0.55)
    ax.legend(title='Defense', loc='upper left', frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig(f'{OUT}/figI_homophily_vs_survival.pdf')
    plt.close()
    print(f'  Saved {OUT}/figI_homophily_vs_survival.pdf')


# ============================================================
# Fig J — Empirical CDF of surrogate fidelity per defense
# ============================================================
def figJ(records):
    bydef = defaultdict(list)
    for r in records:
        if r['defense'] in WM_DEFS:
            bydef[r['defense']].append(r['surrogate_fidelity_to_defended'] * 100.0)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    palette = ['#1b9e77', '#7570b3', '#d95f02', '#e7298a', '#66a61e']
    for i, d in enumerate(WM_DEFS):
        v = sorted(bydef[d])
        if not v:
            continue
        cdf = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, cdf, label=WM_SHORT[i], color=palette[i], linewidth=2.0)
    ax.set_xlabel('Surrogate fidelity to defended target (%)')
    ax.set_ylabel('Empirical CDF')
    ax.set_title('Distribution of surrogate fidelity (all attacks, datasets, seeds)')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=':', alpha=0.55)
    ax.legend(title='Defense', loc='lower right', frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig(f'{OUT}/figJ_fidelity_ecdf.pdf')
    plt.close()
    print(f'  Saved {OUT}/figJ_fidelity_ecdf.pdf')


# ============================================================
# Fig K — Per-dataset heatmap grid of joint fidelity (12 × 5)
# ============================================================
def figK(records):
    # Compute per (dataset, attack, defense) mean fidelity
    bymat = {ds: np.full((len(ATTACKS_12), len(WM_DEFS)), np.nan) for ds in DATASETS_10}
    for ds in DATASETS_10:
        for ai, a in enumerate(ATTACKS_12):
            for dj, d in enumerate(WM_DEFS):
                vals = [r['surrogate_fidelity_to_defended'] * 100.0 for r in records
                        if r['dataset'] == ds and r['attack'] == a and r['defense'] == d]
                if vals:
                    bymat[ds][ai, dj] = float(np.mean(vals))

    fig, axes = plt.subplots(2, 5, figsize=(16.5, 8.6))
    cmap = plt.get_cmap('RdYlGn')
    vmin, vmax = 0, 100
    for k, ds in enumerate(DATASETS_10):
        ax = axes[k // 5, k % 5]
        mat = bymat[ds]
        im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(DS_SHORT[ds], fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(WM_DEFS)))
        ax.set_xticklabels(WM_SHORT, rotation=30, ha='right', fontsize=9)
        if k % 5 == 0:
            ax.set_yticks(range(len(ATTACKS_12)))
            ax.set_yticklabels(ATTACKS_12, fontsize=9)
        else:
            ax.set_yticks([])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    ax.text(j, i, '--', ha='center', va='center', fontsize=7, color='black')
                else:
                    norm = (v - vmin) / max(vmax - vmin, 1e-9)
                    color = 'white' if (norm < 0.30 or norm > 0.80) else 'black'
                    ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                            fontsize=7.5, color=color)
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.012)
    cbar.set_label('Surrogate fidelity (%)', fontsize=11)
    fig.suptitle('Joint surrogate fidelity per dataset (rows: attacks, cols: watermarking defenses)',
                 fontsize=13, y=0.99)
    plt.savefig(f'{OUT}/figK_per_dataset_heatmap_grid.pdf', bbox_inches='tight')
    plt.close()
    print(f'  Saved {OUT}/figK_per_dataset_heatmap_grid.pdf')


# ============================================================
# Fig L — 2D density (hexbin) of surrogate accuracy vs fidelity
# ============================================================
def figL(records):
    accs, fids, defenses = [], [], []
    for r in records:
        if r['defense'] in WM_DEFS:
            accs.append(r['surrogate_acc'] * 100.0)
            fids.append(r['surrogate_fidelity_to_defended'] * 100.0)
            defenses.append(r['defense'])
    accs = np.array(accs); fids = np.array(fids)
    defenses = np.array(defenses)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                             gridspec_kw={'width_ratios': [1.05, 1.0]})

    # Left: pooled hexbin density
    ax = axes[0]
    hb = ax.hexbin(accs, fids, gridsize=22, cmap='viridis', mincnt=1)
    ax.set_xlabel('Surrogate test accuracy (%)')
    ax.set_ylabel('Surrogate fidelity to defended (%)')
    ax.set_title('Pooled density across all (defense, attack, dataset, seed)')
    ax.plot([0, 100], [0, 100], color='red', linestyle='--', linewidth=1.2,
            label='fidelity = accuracy')
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='lower right', frameon=True)
    cb = fig.colorbar(hb, ax=ax, pad=0.015, fraction=0.045)
    cb.set_label('# runs')

    # Right: per-defense scatter with mean ellipse
    ax = axes[1]
    palette = ['#1b9e77', '#7570b3', '#d95f02', '#e7298a', '#66a61e']
    for i, d in enumerate(WM_DEFS):
        mask = defenses == d
        if mask.sum() == 0:
            continue
        ax.scatter(accs[mask], fids[mask], s=18, color=palette[i],
                   alpha=0.45, edgecolor='none', label=WM_SHORT[i])
        cx, cy = np.mean(accs[mask]), np.mean(fids[mask])
        ax.scatter([cx], [cy], marker='X', color=palette[i], s=130,
                   edgecolor='black', linewidth=1.0, zorder=4)
    ax.plot([0, 100], [0, 100], color='red', linestyle='--', linewidth=1.0)
    ax.set_xlabel('Surrogate test accuracy (%)')
    ax.set_ylabel('Surrogate fidelity to defended (%)')
    ax.set_title('Per-defense cloud (X = group mean)')
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(title='Defense', loc='lower right', frameon=True, ncol=2)

    plt.tight_layout()
    plt.savefig(f'{OUT}/figL_acc_vs_fid_density.pdf')
    plt.close()
    print(f'  Saved {OUT}/figL_acc_vs_fid_density.pdf')


if __name__ == '__main__':
    print('Generating appendix-only statistical figures...')
    rec = load_records()
    struct = load_structure()
    print(f'  loaded {len(rec)} per-seed records, {len(struct)} datasets in structure file')
    figH(rec)
    figI(rec, struct)
    figJ(rec)
    figK(rec)
    figL(rec)
    print(f'All appendix figures saved to {OUT}/')
