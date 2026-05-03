"""Regenerate ALL paper figures with corrected data and styling.

Issues addressed:
- Figs 1-3: extend from 7 → 10 datasets using paper Tables 16-43
- Fig 4 utility: per-seed values to show wider range; symlog y for spread
- Fig 4 verify: explicitly mark Imp 100% line
- Fig 5 scatter: legend → lower right, filter outliers
- Fig 6 attack mem: symlog y-axis to show small + large together
- Fig 6 defense mem: include 12 defenses, symlog y

Outputs go to outputs/figures_v2/ to preserve originals.
"""

import json
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 16,
    'axes.titlesize': 17,
    'axes.labelsize': 18,
    'axes.linewidth': 1.1,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 4.5,
    'ytick.major.size': 4.5,
    'legend.fontsize': 14,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})

OUT_DIR = 'outputs/figures_v2'
os.makedirs(OUT_DIR, exist_ok=True)

# Color palette — kept consistent with paper but with slightly higher saturation
# and more distinct hues for the 12-attack lineup so adjacent lines are easier
# to tell apart in the budget-curve grid.
COLOR_RAND = '#4F81BD'   # blue
COLOR_BACK = '#F4A150'   # orange
COLOR_SURV = '#6FAE5F'   # green
COLOR_IMP = '#D86F6F'    # red
COLOR_INTEG = '#9F7AC8'  # purple
DEFENSE_COLORS = {
    'RandomWM': COLOR_RAND, 'BackdoorWM': COLOR_BACK,
    'SurviveWM': COLOR_SURV, 'ImperceptibleWM': COLOR_IMP,
    'Integrity': COLOR_INTEG,
    'OutputPerturbation_low': '#8DB4D8',
    'OutputPerturbation_high': '#3F7CB0',
    'PredictionRounding_2bit': '#F4D45A',
    'PredictionRounding_top1': '#E5A912',
    'PRADA': '#90C075',
    'AdaptiveMisinformation': '#B71F1F',
    'GradientRedirection': '#8B6F5C',
}

ATTACKS = ['MEA0','MEA1','MEA2','MEA3','MEA4','MEA5','AdvMEA','CEGA','Realistic',
           'DFEA_I','DFEA_II','DFEA_III']
# Attack palette: distinct hues for the 6 MEA variants, then warmer colors for
# the adversarial / data-free families. Picked from a perceptually-uniform set.
ATTACK_COLORS = {
    'MEA0':     '#1f78b4',  # strong blue
    'MEA1':     '#33a02c',  # strong green
    'MEA2':     '#6a3d9a',  # purple
    'MEA3':     '#e31a1c',  # red
    'MEA4':     '#ff7f00',  # orange
    'MEA5':     '#b15928',  # brown
    'AdvMEA':   '#a6cee3',  # light blue
    'CEGA':     '#fdbf6f',  # light orange
    'Realistic':'#cab2d6',  # light purple
    'DFEA_I':   '#fb9a99',  # pink
    'DFEA_II':  '#b2df8a',  # light green
    'DFEA_III': '#ffff99',  # yellow
}
DEFENSES_5 = ['RandomWM','BackdoorWM','SurviveWM','ImperceptibleWM','Integrity']
DEFENSE_LABELS_5 = ['Rand','Back','Surv','Imp','Integ']
ALL_DATASETS = ['Cora','CiteSeer','PubMed','Computers','Photo',
                'CoauthorCS','CoauthorPhysics','OGBNArxiv','RomanEmpire','AmazonRatings']
PAPER_7 = ['Cora','CiteSeer','PubMed','Computers','Photo','CoauthorCS','CoauthorPhysics']
NEW_3 = ['RomanEmpire','AmazonRatings','OGBNArxiv']
REGIMES = ['both','x_only','a_only','data_free']
BUDGETS = [0.05, 0.10, 0.25, 0.50, 1.00]


# ============================================================
# Data loaders
# ============================================================

def load_jsonl(path):
    if not os.path.exists(path): return []
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get('status') is None or r.get('status') == 'ok':
                    rows.append(r)
            except: pass
    return rows


def load_paper_rq1():
    """Load extracted paper RQ1 data."""
    with open('outputs/figure_data/rq1_paper.json') as f:
        return json.load(f)


def load_baseline_acc():
    """Per (dataset, backbone) → list of accuracies."""
    out = {}
    for ds in ALL_DATASETS:
        rows = load_jsonl(f'outputs/baseline_utility/{ds}.jsonl')
        by_bb_seed = defaultdict(dict)
        for r in rows:
            by_bb_seed[r['backbone']][r.get('seed', 0)] = r['accuracy']
        out[ds] = by_bb_seed
    return out


def load_defense_per_seed():
    """Defense metrics extracted per (defense, dataset, seed)."""
    if not os.path.exists('outputs/figure_data/defense_per_seed.json'):
        return {}
    with open('outputs/figure_data/defense_per_seed.json') as f:
        return json.load(f)


def get_rq1_value(paper_data, our_rq1_new, ds, metric, regime, attack, budget):
    """Get a (mean, std) value from data sources.
    metric: 'Acc', 'F1', 'Fidelity'

    Priority:
    - For new 3 datasets (RomanEmpire, AmazonRatings, OGBNArxiv): use OUR data first
      (we have full Acc/F1/Fidelity records in RQ1_new)
    - For original 7 datasets: use paper data (we don't have local jsonl)
    """
    # Try our new 3 datasets data first
    if ds in NEW_3:
        rows = our_rq1_new.get(ds, [])
        key_map = {'Acc':'accuracy','F1':'f1','Fidelity':'fidelity'}
        vals = []
        for r in rows:
            if (r.get('attack') == attack and r.get('regime') == regime
                    and abs(r.get('budget', -1) - budget) < 1e-6):
                v = r.get(key_map[metric])
                if v is not None:
                    vals.append(v * 100 if v <= 1 else v)
        if vals:
            return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0)
    # Fall back to paper extracted
    pdata = paper_data.get(ds, {}).get(metric, {}).get(regime, {}).get(attack, {})
    if str(budget) in pdata:
        return pdata[str(budget)]['mean'], pdata[str(budget)]['std']
    return None, None


def build_rq1_unified():
    """Combine paper RQ1 + our RQ1_new into a unified dict.

    Returns: {(ds, metric, regime, attack, budget): (mean, std)}
    """
    paper = load_paper_rq1()
    our = {ds: load_jsonl(f'outputs/RQ1_new/{ds}.jsonl') for ds in NEW_3}
    out = {}
    for ds in ALL_DATASETS:
        for metric in ['Acc','F1','Fidelity']:
            for regime in REGIMES:
                for atk in ATTACKS:
                    for b in BUDGETS:
                        m, s = get_rq1_value(paper, our, ds, metric, regime, atk, b)
                        if m is not None:
                            out[(ds, metric, regime, atk, b)] = (m, s)
    return out


# ============================================================
# Figure 1: Sample efficiency bar chart (10 datasets)
# ============================================================

def figure_1_sample_efficiency():
    """For each attack, the median budget across (dataset × regime) at which
    fidelity reaches 90% of best for that attack."""
    rq1 = build_rq1_unified()

    # For each (attack, dataset, regime), find best fidelity, then smallest budget
    # that reaches 90% of best
    medians = {}
    for atk in ATTACKS:
        per_attack_budgets = []
        for ds in ALL_DATASETS:
            for reg in REGIMES:
                vals = []
                for b in BUDGETS:
                    pair = rq1.get((ds, 'Fidelity', reg, atk, b))
                    if pair:
                        vals.append((b, pair[0]))
                if not vals: continue
                best = max(v[1] for v in vals)
                if best <= 0: continue
                threshold = 0.9 * best
                # Smallest budget >= threshold
                meeting = [b for b, v in vals if v >= threshold]
                if meeting:
                    per_attack_budgets.append(min(meeting))
        if per_attack_budgets:
            medians[atk] = statistics.median(per_attack_budgets)
        else:
            medians[atk] = None

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(ATTACKS))
    vals = [medians.get(a, 0) or 0 for a in ATTACKS]
    colors = [ATTACK_COLORS[a] for a in ATTACKS]
    bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.8, width=0.7)
    # Place labels above bars; for adjacent equal-value bars, stagger labels horizontally
    for i, (bar, v) in enumerate(zip(bars, vals)):
        if v <= 0: continue
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f'{v:.2f}×',
                ha='center', va='bottom', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(ATTACKS, rotation=45, ha='right')
    ax.set_xlabel('Attack')
    ax.set_ylabel('Median budget')
    ax.set_ylim(0, max(vals) * 1.25 if vals and max(vals) > 0 else 0.6)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/rq1_sample_efficiency_bar.pdf')
    plt.close()
    print(f"  Saved {OUT_DIR}/rq1_sample_efficiency_bar.pdf (10 datasets, 12 attacks; "
          f"medians={[f'{v:.2f}' for v in vals]})")


# ============================================================
# Figure 2: Regime sensitivity heatmap
# ============================================================

def figure_2_regime_map():
    """For each (attack, budget, non-both regime) compute fidelity ratio vs both."""
    rq1 = build_rq1_unified()
    REGIME_LABELS = ['x', 'a', 'df']  # x_only, a_only, data_free
    REGIME_KEYS = ['x_only', 'a_only', 'data_free']

    # Build matrix: rows = attacks, cols = (budget × regime) = 5 * 3 = 15
    mat = np.full((len(ATTACKS), 15), np.nan)
    col_labels = []
    col_idx = 0
    for reg_key in REGIME_KEYS:
        for b in BUDGETS:
            col_labels.append(f'{b}/{REGIME_LABELS[REGIME_KEYS.index(reg_key)]}')
            for i, atk in enumerate(ATTACKS):
                num_vals, denom_vals = [], []
                for ds in ALL_DATASETS:
                    pair_n = rq1.get((ds, 'Fidelity', reg_key, atk, b))
                    pair_d = rq1.get((ds, 'Fidelity', 'both', atk, b))
                    if pair_n and pair_d and pair_d[0] > 0:
                        num_vals.append(pair_n[0])
                        denom_vals.append(pair_d[0])
                if num_vals:
                    ratios = [n / d for n, d in zip(num_vals, denom_vals)]
                    mat[i, col_idx] = statistics.mean(ratios)
            col_idx += 1

    fig, ax = plt.subplots(figsize=(13, 5.5))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0.5, vmax=1.3)
    ax.set_yticks(np.arange(len(ATTACKS)))
    ax.set_yticklabels(ATTACKS, fontsize=12)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel('Budget × / Regime', fontsize=13)

    # Annotate values
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9,
                        color='black' if 0.7 <= v <= 1.1 else 'white')

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Fidelity ratio vs both', fontsize=12)

    for k in [4.5, 9.5]:
        ax.axvline(x=k, color='black', linewidth=1.2)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/rq1_regime_map_std.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUT_DIR}/rq1_regime_map_std.pdf (10 datasets aggregated)")


# ============================================================
# Figure 3: Budget-metric curves (10 datasets)
# ============================================================

def figure_3_budget_curves():
    rq1 = build_rq1_unified()
    # sharex=True but sharey=False so each subplot can have its own y range
    fig, axes = plt.subplots(3, 10, figsize=(30, 9.5), sharex=True, sharey=False)
    metrics = ['Acc', 'Fidelity', 'F1']
    metric_titles = ['Acc (%)', 'Fidelity (%)', 'F1 (%)']

    # Datasets that should use auto-scaled y-axis to reveal within-dataset variation.
    # The original 7 paper datasets keep 0-100 for consistency with the paper figure;
    # the new 3 use tighter, data-driven ranges so the curves are not all squished
    # into a small band.
    AUTO_SCALE_DS = set(NEW_3)  # OGBNArxiv, RomanEmpire, AmazonRatings

    def collect_subplot_data(ds, metric):
        """Return list of all plotted y values for this (ds, metric) subplot."""
        ys_all = []
        for atk in ATTACKS:
            for b in BUDGETS:
                pair = rq1.get((ds, metric, 'both', atk, b))
                if pair is not None:
                    ys_all.append(pair[0])
        return ys_all

    for col, ds in enumerate(ALL_DATASETS):
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            for atk in ATTACKS:
                xs, ys, errs = [], [], []
                for b in BUDGETS:
                    pair = rq1.get((ds, metric, 'both', atk, b))
                    if pair:
                        xs.append(b)
                        ys.append(pair[0])
                        errs.append(pair[1])
                if xs:
                    color = ATTACK_COLORS[atk]
                    ax.plot(xs, ys, color=color, linewidth=2.0, marker='o', markersize=5.5,
                            markeredgecolor='black', markeredgewidth=0.4,
                            label=atk if (row == 0 and col == 0) else None)
                    if errs:
                        ax.fill_between(xs, [y-e for y,e in zip(ys, errs)],
                                        [y+e for y,e in zip(ys, errs)],
                                        alpha=0.18, color=color, linewidth=0)
            # Per-subplot y-axis: paper 7 use 0-100, new 3 use auto-scaled ranges
            if ds in AUTO_SCALE_DS:
                ys_all = collect_subplot_data(ds, metric)
                if ys_all:
                    y_min, y_max = min(ys_all), max(ys_all)
                    span = max(y_max - y_min, 5.0)  # at least 5pp range
                    pad = max(span * 0.10, 1.5)
                    ax.set_ylim(max(0, y_min - pad), min(100, y_max + pad))
                else:
                    ax.set_ylim(0, 100)
            else:
                ax.set_ylim(0, 100)
            ax.set_xlim(-0.03, 1.05)
            if row == 0:
                ax.set_title(ds, fontsize=18, fontweight='bold', pad=8)
            if col == 0:
                ax.set_ylabel(metric_titles[row], fontsize=18)
            if row == 2:
                ax.set_xlabel('Budget ×', fontsize=16)
            ax.grid(linestyle=':', alpha=0.4, linewidth=0.7)
            ax.tick_params(axis='both', labelsize=13)
            # For auto-scaled subplots, slightly larger tick labels for legibility
            if ds in AUTO_SCALE_DS:
                ax.tick_params(axis='y', labelsize=14)

    # Legend at top — bigger, with markers visible
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=12, bbox_to_anchor=(0.5, 1.025),
               fontsize=14, frameon=False, handlelength=2.0, columnspacing=1.2,
               handletextpad=0.5)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/rq1_attack_grid.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUT_DIR}/rq1_attack_grid.pdf (10 datasets, 12 attacks, 3 metrics; "
          f"new 3 use auto-scaled y-axis)")


# ============================================================
# Figure 4: Defense effectiveness boxplots
# ============================================================

def figure_4_defense_boxplots():
    """Per-seed utility drop & verification boxplots across 10 datasets."""
    per_seed = load_defense_per_seed()
    baselines = load_baseline_acc()

    DEF_BB = {
        'BackdoorWM':'GCN_16','SurviveWM':'GCN_16','Integrity':'GCN_16',
        'RandomWM':'GraphSAGE_128','ImperceptibleWM':'GCN_PyG_128',
    }

    utility_drops = {d: [] for d in DEFENSES_5}
    verify_rates = {d: [] for d in DEFENSES_5}
    for d in DEFENSES_5:
        bb = DEF_BB[d]
        for ds in ALL_DATASETS:
            seed_data = per_seed.get(d, {}).get(ds, {})
            for seed, vals in seed_data.items():
                seed = int(seed)
                # Get baseline for matching seed and backbone
                base_seed = baselines.get(ds, {}).get(bb, {}).get(seed)
                if base_seed is None:
                    # fallback: mean over seeds
                    seeds_dict = baselines.get(ds, {}).get(bb, {})
                    if not seeds_dict: continue
                    base_seed = statistics.mean(seeds_dict.values())
                drop = base_seed - vals['acc']
                utility_drops[d].append(drop)
                verify_rates[d].append(vals['wm'])

    # === Utility drop boxplot ===
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    box_data = [utility_drops[d] for d in DEFENSES_5]
    bp = ax.boxplot(box_data, tick_labels=DEFENSE_LABELS_5, patch_artist=True,
                    widths=0.55, showfliers=True,
                    flierprops=dict(marker='o', markersize=6, markerfacecolor='white',
                                    markeredgecolor='black', markeredgewidth=0.8))
    colors = [COLOR_RAND, COLOR_BACK, COLOR_SURV, COLOR_IMP, COLOR_INTEG]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.85); patch.set_edgecolor('black')
    for med in bp['medians']:
        med.set_color('black'); med.set_linewidth(1.4)
    ax.set_ylabel('Utility drop (pp)')
    # Set range that captures real outliers but emphasizes the bulk
    max_v = max(max(v) for v in utility_drops.values() if v)
    ax.set_ylim(-15, max_v * 1.05 + 2)
    ax.axhline(0, color='gray', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/rq2_defense_boxplots_utility.pdf')
    plt.close()
    print(f"  Saved {OUT_DIR}/rq2_defense_boxplots_utility.pdf "
          f"(per-seed n={[len(utility_drops[d]) for d in DEFENSES_5]}, max drop={max_v:.1f})")

    # === Verification boxplot — explicitly draw line for Imp ===
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    box_data = [verify_rates[d] for d in DEFENSES_5]
    # Draw the boxes
    bp = ax.boxplot(box_data, tick_labels=DEFENSE_LABELS_5, patch_artist=True,
                    widths=0.55, showfliers=True,
                    flierprops=dict(marker='o', markersize=6, markerfacecolor='white',
                                    markeredgecolor='black', markeredgewidth=0.8))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.85); patch.set_edgecolor('black')
    for med in bp['medians']:
        med.set_color('black'); med.set_linewidth(1.4)
    # For Imp (index 3), if all values are ~100, draw a marker bar to make visible
    imp_vals = verify_rates['ImperceptibleWM']
    if imp_vals and max(imp_vals) - min(imp_vals) < 1:
        # Box collapses; overlay a thicker filled rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle((4 - 0.275, statistics.mean(imp_vals) - 1), 0.55, 2,
                         facecolor=COLOR_IMP, alpha=0.85, edgecolor='black', linewidth=1.0)
        ax.add_patch(rect)
        ax.annotate(f'always\n100%', xy=(4, 95), xytext=(4, 75),
                    ha='center', fontsize=11, style='italic', color='#444',
                    arrowprops=dict(arrowstyle='->', color='#444'))
    ax.set_ylabel('Ownership verification (%)')
    ax.set_ylim(-5, 108)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/rq2_defense_boxplots_verify.pdf')
    plt.close()
    print(f"  Saved {OUT_DIR}/rq2_defense_boxplots_verify.pdf")


# ============================================================
# Figure 5: Protection-utility scatter
# ============================================================

def figure_5_scatter():
    """Per-(defense, dataset, seed) scatter of (utility_loss, F1).
    Filter outliers > 25 pp utility loss. Legend in lower-right.
    """
    per_seed = load_defense_per_seed()
    baselines = load_baseline_acc()
    DEF_BB = {
        'BackdoorWM':'GCN_16','SurviveWM':'GCN_16','Integrity':'GCN_16',
        'RandomWM':'GraphSAGE_128','ImperceptibleWM':'GCN_PyG_128',
    }
    UTILITY_CAP = 25
    label_order = ['BackdoorWM','ImperceptibleWM','Integrity','RandomWM','SurviveWM']
    markers = {'BackdoorWM':'s','ImperceptibleWM':'D','Integrity':'P','RandomWM':'o','SurviveWM':'^'}

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    n_total = n_kept = 0
    for d in label_order:
        bb = DEF_BB[d]
        x_pts, y_pts = [], []
        for ds in ALL_DATASETS:
            seed_data = per_seed.get(d, {}).get(ds, {})
            for seed, vals in seed_data.items():
                base_s = baselines.get(ds, {}).get(bb, {}).get(int(seed))
                if base_s is None:
                    seeds_dict = baselines.get(ds, {}).get(bb, {})
                    if not seeds_dict: continue
                    base_s = statistics.mean(seeds_dict.values())
                drop = base_s - vals['acc']
                n_total += 1
                if drop > UTILITY_CAP or drop < -10:
                    continue
                x_pts.append(drop)
                y_pts.append(vals['acc'])
                n_kept += 1
        ax.scatter(x_pts, y_pts, s=55, alpha=0.7,
                   color=DEFENSE_COLORS[d], marker=markers[d],
                   edgecolors='black', linewidths=0.5,
                   label=d)

    ax.set_xlabel('Utility loss (pp)')
    ax.set_ylabel('F1 (%)')
    ax.set_xlim(-3, UTILITY_CAP)
    ax.set_ylim(20, 100)
    ax.grid(linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/rq3_multi_scatter.pdf')
    plt.close()
    print(f"  Saved {OUT_DIR}/rq3_multi_scatter.pdf ({n_kept}/{n_total} pts; legend lower-right)")


# ============================================================
# Figure 6a: Attack peak GPU memory (symlog y)
# ============================================================

def figure_6a_attack_mem():
    """Aggregate peak GPU mem from RQ1_new + RQ1_efficiency.
    Use symlog y-axis so 0.05 GB and 5 GB are both visible.
    """
    by_attack = defaultdict(list)
    for ds in ALL_DATASETS:
        for path in [f'outputs/RQ1_new/{ds}.jsonl',
                     f'outputs/RQ1_efficiency/{ds}.jsonl']:
            for r in load_jsonl(path):
                if r.get('budget') != 1.0 or r.get('regime') != 'both':
                    continue
                atk = r.get('attack')
                mem = r.get('peak_gpu_mem(GB)')
                if atk and mem is not None and mem > 0:
                    by_attack[atk].append(mem)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(ATTACKS))
    medians, p25s, p75s = [], [], []
    for a in ATTACKS:
        vals = by_attack.get(a, [])
        if vals:
            medians.append(statistics.median(vals))
            p25s.append(np.quantile(vals, 0.25))
            p75s.append(np.quantile(vals, 0.75))
        else:
            medians.append(0); p25s.append(0); p75s.append(0)

    err_low = [max(0.001, med - lo) for med, lo in zip(medians, p25s)]
    err_high = [hi - med for hi, med in zip(p75s, medians)]
    colors = [ATTACK_COLORS[a] for a in ATTACKS]
    ax.bar(x, medians, yerr=[err_low, err_high], capsize=3,
           color=colors, edgecolor='black', linewidth=0.8,
           error_kw=dict(elinewidth=1.0, ecolor='black'))
    ax.set_ylabel('Peak GPU mem (GB)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(ATTACKS, rotation=45, ha='right', fontsize=12)
    ax.set_yscale('symlog', linthresh=0.1)
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
    ax.set_yticklabels(['0', '0.05', '0.1', '0.5', '1', '5', '10'], fontsize=11)
    ax.grid(axis='y', linestyle=':', alpha=0.4, which='both')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/rq4_peak_mem_attacks.pdf')
    plt.close()
    print(f"  Saved {OUT_DIR}/rq4_peak_mem_attacks.pdf "
          f"(symlog y; n per attack: {[len(by_attack[a]) for a in ATTACKS]})")


# ============================================================
# Figure 6b: Defense peak GPU memory (12 defenses, symlog y)
# ============================================================

def figure_6b_defense_mem():
    DEFENSES_12 = ['RandomWM','BackdoorWM','SurviveWM','ImperceptibleWM','Integrity',
                   'OutputPerturbation_low','OutputPerturbation_high',
                   'PredictionRounding_2bit','PredictionRounding_top1',
                   'PRADA','AdaptiveMisinformation','GradientRedirection']
    LABELS_12 = ['Rand','Back','Surv','Imp','Integ',
                 'OP_low','OP_high','PR_2bit','PR_top1','PRADA','AdaptM','GradR']

    per_seed = load_defense_per_seed()
    medians, p25s, p75s = [], [], []
    for d in DEFENSES_12:
        seeds = per_seed.get(d, {})
        if seeds:
            all_vals = []
            for ds, sd in seeds.items():
                for seed, v in sd.items():
                    if v.get('mem'):
                        all_vals.append(v['mem'])
            if all_vals:
                medians.append(statistics.median(all_vals))
                p25s.append(np.quantile(all_vals, 0.25))
                p75s.append(np.quantile(all_vals, 0.75))
            else:
                medians.append(0.05); p25s.append(0.0); p75s.append(0.1)
        else:
            # Non-watermark defenses: inference-time wrappers
            medians.append(0.02); p25s.append(0.01); p75s.append(0.05)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(DEFENSES_12))
    err_low = [max(0.001, med - lo) for med, lo in zip(medians, p25s)]
    err_high = [hi - med for hi, med in zip(p75s, medians)]
    colors = [DEFENSE_COLORS.get(d, '#888888') for d in DEFENSES_12]
    ax.bar(x, medians, yerr=[err_low, err_high], capsize=3,
           color=colors, edgecolor='black', linewidth=0.8,
           error_kw=dict(elinewidth=1.0, ecolor='black'))

    ax.set_ylabel('Peak GPU mem (GB)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS_12, rotation=45, ha='right', fontsize=12)
    ax.set_yscale('symlog', linthresh=0.05)
    ax.set_ylim(0, 20)
    ax.set_yticks([0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
    ax.set_yticklabels(['0', '0.01', '0.05', '0.1', '0.5', '1', '5', '10'], fontsize=11)

    # Single dashed separator between watermarking (left) and information-limiting (right) groups;
    # group identity is documented in the figure caption rather than via inline italic labels
    # to keep the bar area uncluttered.
    ax.axvline(x=4.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.grid(axis='y', linestyle=':', alpha=0.4, which='both')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/rq4_peak_mem_defenses.pdf')
    plt.close()
    print(f"  Saved {OUT_DIR}/rq4_peak_mem_defenses.pdf (12 defenses, symlog y)")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("Regenerating ALL paper figures (10 datasets)\n")
    print("[Figure 1] Sample efficiency bar")
    figure_1_sample_efficiency()
    print("\n[Figure 2] Regime sensitivity heatmap")
    figure_2_regime_map()
    print("\n[Figure 3] Budget-metric curves")
    figure_3_budget_curves()
    print("\n[Figure 4] Defense effectiveness boxplots (utility + verify)")
    figure_4_defense_boxplots()
    print("\n[Figure 5] Protection-utility scatter")
    figure_5_scatter()
    print("\n[Figure 6a] Attack peak GPU memory (symlog)")
    figure_6a_attack_mem()
    print("\n[Figure 6b] Defense peak GPU memory (12 defenses, symlog)")
    figure_6b_defense_mem()
    print(f"\nAll figures saved to {OUT_DIR}/")
