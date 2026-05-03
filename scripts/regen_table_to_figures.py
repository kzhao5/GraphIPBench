"""Convert main-paper tables to academic-style figures.

Replaces: Tables 2, 3, 4, 5+6, 7+8+9, 11, 12 with 7 publication-quality figures:
  Fig A — RQ1 extension on 3 additional graphs (grouped horizontal bar)
  Fig B — RQ2 watermark profile (radar chart)
  Fig C — Information-limiting defenses (annotated dual heatmap)
  Fig D — Computational cost (combined log bar, attacks + defenses)
  Fig E — Joint evaluation on Cora (3-panel heatmap: WM fid + non-WM fid + survival)
  Fig F — Cross-architecture extraction (2x2 mini heatmaps)
  Fig G — Cross-task generalization (LP + GC grouped bar)

All figures use a unified style consistent with Figures 1-6 already produced.
Outputs go to outputs/figures_v2/.
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
from matplotlib.patches import Patch

# Unified style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 16,
    'axes.titlesize': 17,
    'axes.labelsize': 18,
    'axes.linewidth': 1.1,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'legend.frameon': True,
    'legend.framealpha': 0.92,
    'legend.edgecolor': 'black',
    'lines.linewidth': 1.8,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})

OUT = 'outputs/figures_v2'
os.makedirs(OUT, exist_ok=True)

# ----------------------- Color palettes -----------------------
DEFENSE_COLORS = {
    'RandomWM': '#4F81BD',
    'BackdoorWM': '#F4A150',
    'SurviveWM': '#6FAE5F',
    'ImperceptibleWM': '#D86F6F',
    'Integrity': '#9F7AC8',
    'OutputPerturbation_low': '#8DB4D8',
    'OutputPerturbation_high': '#3F7CB0',
    'PredictionRounding_2bit': '#F4D45A',
    'PredictionRounding_top1': '#E5A912',
    'PRADA': '#90C075',
    'AdaptiveMisinformation': '#B71F1F',
    'GradientRedirection': '#8B6F5C',
}
ATTACK_COLORS = {
    'MEA0':'#1f78b4','MEA1':'#33a02c','MEA2':'#6a3d9a','MEA3':'#e31a1c',
    'MEA4':'#ff7f00','MEA5':'#b15928','AdvMEA':'#a6cee3','CEGA':'#fdbf6f',
    'Realistic':'#cab2d6','DFEA_I':'#fb9a99','DFEA_II':'#b2df8a','DFEA_III':'#ffff99',
}

# A perceptually-uniform diverging colormap good for fidelity heatmaps
def make_cmap_fid():
    return plt.get_cmap('RdYlGn')

ATTACKS_12 = ['MEA0','MEA1','MEA2','MEA3','MEA4','MEA5','AdvMEA','CEGA','Realistic',
              'DFEA_I','DFEA_II','DFEA_III']


# =============================================================
# Figure A — RQ1 Extension on 3 additional graphs (Table 2)
# =============================================================
def figure_A_rq1_extension():
    """Grouped horizontal bar chart: 3 datasets × 8 attacks @ b=0.25."""
    # From Table 2 in the paper
    data = {
        'RomanEmpire':   {'MEA0': (77.0, 0.8), 'MEA3': (74.1, 1.1), 'MEA5': (76.7, 1.2),
                         'AdvMEA': (21.9, 6.2), 'CEGA': (73.4, 2.3), 'Realistic': (58.2, 1.5),
                         'DFEA_I': (13.8, 0.3), 'DFEA_III': (13.8, 0.3)},
        'AmazonRatings': {'MEA0': (93.8, 0.7), 'MEA3': (93.8, 1.3), 'MEA5': (94.7, 0.9),
                         'AdvMEA': (68.7, 3.4), 'CEGA': (89.1, 1.2), 'Realistic': (90.6, 2.6),
                         'DFEA_I': (70.4, 1.6), 'DFEA_III': (70.4, 1.6)},
        'OGBN-Arxiv':    {'MEA0': (77.1, 4.6), 'MEA3': (74.2, 5.1), 'MEA5': (77.2, 4.8),
                         'AdvMEA': (26.2, 21.2), 'CEGA': (77.4, 4.6), 'Realistic': (75.3, 2.7),
                         'DFEA_I': (6.2, 1.8), 'DFEA_III': (6.3, 1.9)},
    }
    attacks = ['MEA0','MEA3','MEA5','AdvMEA','CEGA','Realistic','DFEA_I','DFEA_III']
    datasets = ['OGBN-Arxiv', 'RomanEmpire', 'AmazonRatings']  # large→small heterophily
    DS_COLORS = {'OGBN-Arxiv': '#3F7CB0', 'RomanEmpire': '#D86F6F', 'AmazonRatings': '#F4A150'}

    # Compact figure: smaller absolute size + smaller fonts so it fits in a wrapfigure
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    y = np.arange(len(attacks))
    h = 0.27
    for i, ds in enumerate(datasets):
        means = [data[ds][a][0] for a in attacks]
        stds = [data[ds][a][1] for a in attacks]
        offset = (i - 1) * h
        bars = ax.barh(y + offset, means, h, xerr=stds, capsize=1.8,
                       label=ds, color=DS_COLORS[ds],
                       edgecolor='black', linewidth=0.5,
                       error_kw=dict(elinewidth=0.6, ecolor='black'))
        # Annotate bar values (smaller, only on the bar tip)
        for j, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + 1.2, y[j] + offset, f'{m:.0f}',
                    va='center', fontsize=7, color='black')

    ax.set_yticks(y)
    ax.set_yticklabels(attacks, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Surrogate fidelity (%) at $0.25\\times$', fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlim(0, 110)
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black',
              fontsize=7.5, title='Dataset', title_fontsize=8)
    ax.grid(axis='x', linestyle=':', alpha=0.45, linewidth=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f'{OUT}/figA_rq1_extension.pdf')
    plt.close()
    print(f"  Saved {OUT}/figA_rq1_extension.pdf")


# =============================================================
# Figure B — RQ2 watermark profile (Table 3): radar chart
# =============================================================
def figure_B_rq2_radar():
    """Radar chart of 5 watermarking defenses across 6 metrics (medians from Table 3).
    All metrics normalized to 0-1 with consistent direction (higher = better).
    """
    # From Table 3
    raw = {
        'RandomWM':        {'F1': 64.99, 'Fidelity': 74.13, 'Verif': 72.00,
                            'UtilityKept': 100 - 3.93, 'TimeFast': None, 'MemLight': None},
        'BackdoorWM':      {'F1': 69.13, 'Fidelity': 80.07, 'Verif': 100.00,
                            'UtilityKept': 100 - 3.27, 'TimeFast': None, 'MemLight': None},
        'SurviveWM':       {'F1': 67.47, 'Fidelity': 79.93, 'Verif': 21.76,
                            'UtilityKept': 100 - 0.13, 'TimeFast': None, 'MemLight': None},
        'ImperceptibleWM': {'F1': 69.49, 'Fidelity': 77.63, 'Verif': 100.00,
                            'UtilityKept': 100 - 1.65, 'TimeFast': None, 'MemLight': None},
        'Integrity':       {'F1': 73.43, 'Fidelity': 76.03, 'Verif': 66.67,
                            'UtilityKept': 100 - 4.03, 'TimeFast': None, 'MemLight': None},
    }
    # Time / memory normalized: higher=better → invert raw seconds/GB
    time_raw = {'RandomWM': 34.8, 'BackdoorWM': 1.98, 'SurviveWM': 2.27,
                'ImperceptibleWM': 676, 'Integrity': 1.38}
    mem_raw = {'RandomWM': 0.09, 'BackdoorWM': 0.16, 'SurviveWM': 0.32,
               'ImperceptibleWM': 2.30, 'Integrity': 0.20}
    # Map to 0-100 score where 100 = fastest/lightest
    t_min, t_max = min(time_raw.values()), max(time_raw.values())
    m_min, m_max = min(mem_raw.values()), max(mem_raw.values())
    for d in raw:
        # Use log scale for time/memory normalization since range is huge
        raw[d]['TimeFast'] = 100 * (1 - (np.log10(time_raw[d]) - np.log10(t_min)) /
                                    (np.log10(t_max) - np.log10(t_min) + 1e-9))
        raw[d]['MemLight'] = 100 * (1 - (np.log10(mem_raw[d]) - np.log10(m_min)) /
                                   (np.log10(m_max) - np.log10(m_min) + 1e-9))

    metrics = ['F1', 'Fidelity', 'Verif', 'UtilityKept', 'TimeFast', 'MemLight']
    metric_labels = ['F1', 'Fidelity', 'Verification', 'Utility-kept', 'Speed', 'Mem-efficient']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 6.2), subplot_kw=dict(polar=True))
    for d in ['BackdoorWM', 'ImperceptibleWM', 'Integrity', 'RandomWM', 'SurviveWM']:
        values = [raw[d][m] for m in metrics]
        values += values[:1]
        c = DEFENSE_COLORS[d]
        ax.plot(angles, values, color=c, linewidth=2.4, label=d, marker='o', markersize=5)
        ax.fill(angles, values, color=c, alpha=0.13)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=11)
    ax.set_rlabel_position(0)
    ax.grid(linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', bbox_to_anchor=(1.35, 0.05),
              frameon=True, fancybox=False, edgecolor='black', fontsize=12)
    ax.set_title('Watermark defense profile (radar; higher = better on all axes)',
                 fontsize=15, pad=20)

    plt.tight_layout()
    plt.savefig(f'{OUT}/figB_rq2_radar.pdf')
    plt.close()
    print(f"  Saved {OUT}/figB_rq2_radar.pdf")


# =============================================================
# Figure C — Information-limiting defenses heatmap (Table 4)
# =============================================================
def figure_C_info_limiting_heatmap():
    """Two-panel annotated heatmap: 10 datasets × 7 defenses, Acc + WM Acc."""
    DS = ['Cora','CiteSeer','PubMed','Computers','Photo','CoauthorCS','CoauthorPhysics',
          'OGBN-Arxiv','RomanEmpire','AmazonRatings']
    DEFS = ['OP_low','OP_high','PR_2bit','PR_top1','PRADA','AdaptM','GradR']
    # Table 4: Acc / WM Acc per cell
    raw = {
        'Cora':            [(79.4,98.6),(79.2,93.9),(73.3,83.7),(79.6,100.0),(40.2,43.0),(41.0,48.5),(79.8,100.0)],
        'CiteSeer':        [(67.6,97.8),(66.3,91.0),(53.9,70.3),(68.8,100.0),(69.3,100.0),(39.8,52.5),(68.4,100.0)],
        'PubMed':          [(77.9,99.0),(75.9,94.7),(77.6,93.4),(78.2,100.0),(78.0,100.0),(44.1,48.6),(78.3,100.0)],
        'Computers':       [(44.0,89.2),(37.6,55.3),(36.1,61.7),(34.8,100.0),(46.0,100.0),(28.0,64.3),(52.4,100.0)],
        'Photo':           [(89.1,98.9),(90.7,96.6),(90.4,96.7),(95.5,100.0),(87.0,100.0),(46.3,49.6),(66.6,100.0)],
        'CoauthorCS':      [(87.8,99.5),(88.2,98.8),(87.5,98.1),(88.1,100.0),(75.0,79.6),(52.4,58.7),(88.2,100.0)],
        'CoauthorPhysics': [(89.4,99.8),(89.1,99.3),(90.2,98.7),(89.5,100.0),(83.4,89.0),(59.0,63.1),(89.7,100.0)],
        'OGBN-Arxiv':      [(37.7,95.5),(37.8,81.2),(30.2,59.6),(39.5,100.0),(37.0,100.0),(19.9,52.3),(38.2,100.0)],
        'RomanEmpire':     [(42.7,95.3),(40.6,82.2),(35.2,56.4),(42.7,100.0),(19.7,25.4),(22.5,50.8),(42.5,100.0)],
        'AmazonRatings':   [(42.0,94.8),(41.2,80.0),(39.4,70.6),(41.6,100.0),(41.8,100.0),(33.9,48.9),(41.7,100.0)],
    }
    acc_mat = np.array([[raw[ds][i][0] for i in range(7)] for ds in DS])
    wm_mat = np.array([[raw[ds][i][1] for i in range(7)] for ds in DS])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    cmap_acc = plt.get_cmap('YlGnBu')
    cmap_wm = plt.get_cmap('OrRd')

    # Panel 1: Accuracy
    ax = axes[0]
    im = ax.imshow(acc_mat, aspect='auto', cmap=cmap_acc, vmin=15, vmax=95)
    for i in range(acc_mat.shape[0]):
        for j in range(acc_mat.shape[1]):
            v = acc_mat[i, j]
            color = 'white' if v > 60 else 'black'
            ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                    fontsize=12, color=color, fontweight='bold')
    ax.set_xticks(np.arange(len(DEFS)))
    ax.set_xticklabels(DEFS, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(DS)))
    ax.set_yticklabels(DS)
    ax.set_title('(a) Protected-model accuracy (%)', fontsize=15, pad=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    # Panel 2: Verification
    ax = axes[1]
    im = ax.imshow(wm_mat, aspect='auto', cmap=cmap_wm, vmin=20, vmax=100)
    for i in range(wm_mat.shape[0]):
        for j in range(wm_mat.shape[1]):
            v = wm_mat[i, j]
            color = 'white' if v > 65 else 'black'
            ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                    fontsize=12, color=color, fontweight='bold')
    ax.set_xticks(np.arange(len(DEFS)))
    ax.set_xticklabels(DEFS, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(DS)))
    ax.set_yticklabels([])  # share with left panel
    ax.set_title('(b) Verification proxy (%)', fontsize=15, pad=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    plt.tight_layout()
    plt.savefig(f'{OUT}/figC_info_limiting_heatmap.pdf')
    plt.close()
    print(f"  Saved {OUT}/figC_info_limiting_heatmap.pdf")


# =============================================================
# Figure D — Computational cost combined (Tables 5+6)
# =============================================================
def figure_D_cost_combined():
    """Two-panel grouped bar with log y-axis: attack time + defense time."""
    DS = ['Cora','CiteSeer','CoauthorCS','CoauthorPhys','Computers','Photo','PubMed']
    # Table 5 attack time (min)
    attack_time = {
        'MEA0':[0.69,0.72,0.86,1.36,2.18,0.79,1.21],
        'MEA1':[0.69,0.73,0.86,1.34,2.18,0.76,1.17],
        'MEA2':[1.51,1.73,2.01,2.25,2.21,1.59,2.02],
        'MEA3':[0.66,0.74,0.70,0.86,3.27,1.14,1.20],
        'MEA4':[0.76,0.94,2.58,6.46,2.50,0.84,1.73],
        'MEA5':[0.69,0.75,0.77,0.92,2.66,1.25,1.23],
        'AdvMEA':[2.35,4.39,8.88,4.93,13.0,10.3,4.51],
        'CEGA':[1.07,1.03,1.48,2.15,3.51,1.07,1.75],
        'Realistic':[90.3,111,472,976,840,248,529],
        'DFEA_I':[0.96,1.11,1.02,0.97,2.82,1.00,1.32],
        'DFEA_II':[0.82,0.88,0.80,0.86,1.28,0.87,1.07],
        'DFEA_III':[1.48,1.55,1.46,1.52,3.30,1.62,1.86],
    }
    # Table 6 defense time (s)
    defense_time = {
        'RandomWM':       [24.3,23.9,57.3,34.8,41.3,36.0,21.8],
        'BackdoorWM':     [1.88,2.17,2.54,3.89,1.98,1.92,1.88],
        'SurviveWM':      [1.59,1.62,2.75,5.62,2.31,1.52,2.27],
        'ImperceptibleWM':[676,709,906,950,461,209,196],
        'Integrity':      [1.29,1.10,1.52,2.37,1.92,1.38,1.25],
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel 1: attacks (median across datasets, IQR error)
    ax = axes[0]
    attacks_order = list(attack_time.keys())
    medians = [statistics.median(attack_time[a]) for a in attacks_order]
    p25s = [np.quantile(attack_time[a], 0.25) for a in attacks_order]
    p75s = [np.quantile(attack_time[a], 0.75) for a in attacks_order]
    err_low = [max(0.001, m - lo) for m, lo in zip(medians, p25s)]
    err_high = [hi - m for hi, m in zip(p75s, medians)]
    x = np.arange(len(attacks_order))
    colors = [ATTACK_COLORS[a] for a in attacks_order]
    ax.bar(x, medians, yerr=[err_low, err_high], capsize=3, color=colors,
           edgecolor='black', linewidth=0.7, error_kw=dict(elinewidth=0.9, ecolor='black'))
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(attacks_order, rotation=45, ha='right')
    ax.set_ylabel('Attack time (min, log scale)')
    ax.set_title('(a) Total attack time at $1.00\\times$ budget', fontsize=15, pad=8)
    ax.grid(axis='y', linestyle=':', alpha=0.4, which='both')
    ax.set_axisbelow(True)

    # Panel 2: defenses
    ax = axes[1]
    DEFS = list(defense_time.keys())
    DEF_LABELS = ['Rand','Back','Surv','Imp','Integ']
    medians = [statistics.median(defense_time[d]) for d in DEFS]
    p25s = [np.quantile(defense_time[d], 0.25) for d in DEFS]
    p75s = [np.quantile(defense_time[d], 0.75) for d in DEFS]
    err_low = [max(0.001, m - lo) for m, lo in zip(medians, p25s)]
    err_high = [hi - m for hi, m in zip(p75s, medians)]
    x = np.arange(len(DEFS))
    colors = [DEFENSE_COLORS[d] for d in DEFS]
    ax.bar(x, medians, yerr=[err_low, err_high], capsize=3, color=colors,
           edgecolor='black', linewidth=0.7, error_kw=dict(elinewidth=0.9, ecolor='black'))
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(DEF_LABELS, rotation=0)
    ax.set_ylabel('Defense time (s, log scale)')
    ax.set_title('(b) Total defense time', fontsize=15, pad=8)
    ax.grid(axis='y', linestyle=':', alpha=0.4, which='both')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f'{OUT}/figD_cost_combined.pdf')
    plt.close()
    print(f"  Saved {OUT}/figD_cost_combined.pdf")


# =============================================================
# Figure E — Joint evaluation on Cora (Tables 7+8+9): 3-panel
# =============================================================
def figure_E_joint_cora():
    """Three-panel annotated heatmap: WM joint fid + non-WM joint fid + WM survival."""
    # Table 7: 12 attacks × 5 watermarks (joint fidelity)
    wm_defs = ['BackdoorWM','SurviveWM','Integrity','RandomWM','ImperceptibleWM']
    wm_labels = ['Back','Surv','Integ','Rand','Imp']
    table7 = np.array([
        [92.5,92.3,91.9,91.7,92.9],
        [91.5,93.7,89.9,90.9,92.9],
        [23.1,20.3,19.2,16.7,24.1],
        [91.2,91.9,87.8,90.2,91.9],
        [91.4,91.4,94.0,89.9,91.9],
        [91.4,92.2,91.9,91.9,91.9],
        [75.6,76.8,72.3,77.6,76.4],
        [87.4,85.2,54.1,79.9,88.5],
        [26.0,22.9,36.6,20.6,26.9],
        [25.6,25.6,19.0,24.9,27.2],
        [25.3,23.0,60.5,21.9,26.9],
        [91.5,90.1,72.3,91.2,91.0],
    ])

    # Table 8: 12 attacks × 7 non-WM defenses
    nonwm_defs = ['OP_low','OP_high','PR_2bit','PR_top1','PRADA','AdaptM','GradR']
    table8 = np.array([
        [92.3,89.7,86.8,92.9,42.4,59.2,92.9],
        [92.0,88.9,86.2,92.4,41.4,57.7,92.4],
        [18.1,18.0,18.8,22.5,17.1,17.8,23.5],
        [91.4,89.3,84.0,91.4,44.1,56.1,91.4],
        [91.0,89.3,86.5,92.4,43.2,58.5,92.4],
        [89.9,88.6,83.2,90.3,43.3,57.4,90.3],
        [78.4,76.4,70.4,72.4,40.4,43.7,77.4],
        [87.7,86.4,81.3,91.9,40.3,55.2,89.4],
        [25.7,25.7,17.8,25.7,19.5,21.2,25.7],
        [25.9,25.6,23.5,25.7,19.3,23.4,25.7],
        [25.7,25.7,23.5,25.7,19.4,21.1,25.7],
        [90.9,89.4,83.6,83.8,54.5,58.2,91.7],
    ])

    # Table 9: 12 attacks × 5 watermarks (WM survival)
    table9 = np.array([
        [0.0,14.9,42.3,12.0,0.0],
        [0.0,14.6,20.6,18.7,0.0],
        [50.0,14.1,15.1,12.7,0.0],
        [16.7,14.2,28.6,15.3,0.0],
        [16.7,14.2,0.0,13.7,0.0],
        [50.0,13.6,11.5,13.7,0.0],
        [50.0,15.0,13.3,13.7,66.7],
        [16.7,13.7,0.0,15.7,0.0],
        [0.0,13.3,0.0,12.7,0.0],
        [0.0,13.5,20.4,14.0,0.0],
        [0.0,12.8,9.1,18.7,0.0],
        [16.7,13.6,0.0,17.7,33.3],
    ])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5),
                             gridspec_kw={'width_ratios': [5, 7, 5]})
    cmap_fid = plt.get_cmap('RdYlGn')

    def draw_heatmap(ax, mat, col_labels, vmin, vmax, title, cmap, fmt='{:.0f}'):
        im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                # contrast color choice
                norm_v = (v - vmin) / max(vmax - vmin, 1e-9)
                color = 'white' if (norm_v < 0.30 or norm_v > 0.80) else 'black'
                ax.text(j, i, fmt.format(v), ha='center', va='center',
                        fontsize=11, color=color, fontweight='bold')
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha='right')
        ax.set_yticks(np.arange(len(ATTACKS_12)))
        ax.set_yticklabels(ATTACKS_12)
        ax.set_title(title, fontsize=14, pad=8)
        return im

    # Panel a
    im1 = draw_heatmap(axes[0], table7, wm_labels, 0, 100,
                       '(a) Surrogate fidelity vs. WM defenses', cmap_fid)
    # Panel b: y-tick labels off
    im2 = draw_heatmap(axes[1], table8, nonwm_defs, 0, 100,
                       '(b) Surrogate fidelity vs. info-limiting defenses', cmap_fid)
    axes[1].set_yticklabels([])
    # Panel c: survival
    im3 = draw_heatmap(axes[2], table9, wm_labels, 0, 70,
                       '(c) Watermark survival on surrogate', plt.get_cmap('Reds'))
    axes[2].set_yticklabels([])

    fig.colorbar(im1, ax=axes[0], fraction=0.05, pad=0.02, label='Fidelity (%)')
    fig.colorbar(im2, ax=axes[1], fraction=0.04, pad=0.02, label='Fidelity (%)')
    fig.colorbar(im3, ax=axes[2], fraction=0.05, pad=0.02, label='Survival (%)')

    plt.tight_layout()
    plt.savefig(f'{OUT}/figE_joint_cora.pdf')
    plt.close()
    print(f"  Saved {OUT}/figE_joint_cora.pdf")


# =============================================================
# Figure F — Cross-architecture extraction (Table 11): 2x2 mini heatmaps
# =============================================================
def figure_F_cross_arch():
    """4 datasets × 3×3 backbone matrix as a 2×2 grid of mini heatmaps."""
    archs = ['GCN', 'GAT', 'SAGE']
    data = {
        'Cora': np.array([[87.5, 89.4, 88.5],
                          [83.3, 90.6, 87.0],
                          [84.0, 86.0, 96.0]]),
        'Computers': np.array([[75.8, 60.5, 45.9],
                               [76.1, 89.8, 77.0],
                               [60.7, 81.2, 72.9]]),
        'OGBN-Arxiv': np.array([[83.1, 62.4, 63.0],
                                [44.9, 80.7, 90.9],
                                [41.9, 71.4, 90.3]]),
        'RomanEmpire': np.array([[63.1, 74.4, 82.7],
                                 [53.9, 76.7, 78.3],
                                 [38.5, 47.9, 92.3]]),
    }
    DS_LIST = ['Cora','Computers','OGBN-Arxiv','RomanEmpire']

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5))
    plt.subplots_adjust(hspace=0.45, wspace=0.30, top=0.93)
    cmap = plt.get_cmap('RdYlGn')
    for idx, ds in enumerate(DS_LIST):
        r, c = idx // 2, idx % 2
        ax = axes[r, c]
        mat = data[ds]
        im = ax.imshow(mat, aspect='equal', cmap=cmap, vmin=35, vmax=100)
        for i in range(3):
            for j in range(3):
                v = mat[i, j]
                norm_v = (v - 35) / 65.0
                color = 'white' if (norm_v < 0.30 or norm_v > 0.80) else 'black'
                # Highlight diagonal
                weight = 'bold' if i == j else 'normal'
                ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                        fontsize=14, color=color, fontweight=weight)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(archs, fontsize=13)
        ax.set_yticks(np.arange(3))
        ax.set_yticklabels(archs, fontsize=13)
        # Only put x-label on bottom row, y-label on left col
        if r == 1:
            ax.set_xlabel('Surrogate backbone', fontsize=13)
        if c == 0:
            ax.set_ylabel('Target backbone', fontsize=13)
        ax.set_title(ds, fontsize=15, fontweight='bold', pad=10)
        # Highlight diagonal cells with thicker border
        for i in range(3):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor='black', linewidth=2.0))

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.022, pad=0.04,
                 label='Surrogate fidelity (%)')
    fig.suptitle('Cross-architecture extraction (matched-pair fidelity in bold; diagonal = matched)',
                 fontsize=15, y=0.985)
    plt.savefig(f'{OUT}/figF_cross_arch.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUT}/figF_cross_arch.pdf")


# =============================================================
# Figure G — LP + GC cross-task generalization (Table 12)
# =============================================================
def figure_G_cross_task():
    """Three-panel grouped bar: LP Cora, ENZYMES, PROTEINS."""
    # Table 12(a) LP Cora
    lp_attacks = ['MEA0','MEA1','MEA2','MEA3','MEA4','MEA5','AdvMEA','CEGA','DFEA_I','DFEA_II','DFEA_III']
    lp_defs = ['None','OP_low','OP_high','PR_2bit','GradR']
    lp = np.array([
        [95.3,96.1,94.2,84.9,70.9],
        [94.8,93.7,92.8,87.2,71.9],
        [95.6,92.8,90.4,28.9,94.3],
        [73.1,71.0,70.2,68.8,70.4],
        [85.3,84.4,81.6,28.9,79.2],
        [93.8,92.4,89.3,81.9,71.1],
        [97.9,96.0,93.1,82.4,80.7],
        [92.1,92.0,92.8,87.8,71.0],
        [94.4,94.4,89.9,28.9,92.6],
        [94.0,92.1,91.3,28.9,93.4],
        [94.4,92.9,91.1,28.9,28.9],
    ])
    # Table 12(b) ENZYMES
    gc_attacks = ['MEA0','MEA1','AdvMEA','CEGA','DFEA_I','DFEA_II']
    gc_defs = ['None','OP_low','OP_high','PR_2bit','PR_top1','GradR']
    enz = np.array([
        [92.2,88.9,86.9,16.7,91.1,90.9],
        [29.4,28.1,24.4, 3.1,32.4,29.6],
        [84.6,85.6,83.0,16.5,86.1,85.6],
        [88.7,92.2,90.2,59.1,90.9,85.0],
        [93.0,91.3,91.1,60.0,91.3,83.0],
        [90.6,90.2,84.1,16.7,91.3,91.7],
    ])
    prot = np.array([
        [95.5,90.9,96.2,83.8,96.2,94.9],
        [63.4,61.9,63.3,62.4,65.1,63.2],
        [90.2,88.7,89.9,85.0,91.6,92.5],
        [97.1,96.8,97.5,94.7,97.8,97.1],
        [97.5,97.4,97.9,95.0,97.7,96.6],
        [96.3,94.7,94.2,84.9,95.9,95.5],
    ])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    cmap = plt.get_cmap('RdYlGn')

    for ax, mat, attacks, defs, title in [
        (axes[0], lp, lp_attacks, lp_defs, '(a) Link Prediction (Cora)'),
        (axes[1], enz, gc_attacks, gc_defs, '(b) Graph Class. (ENZYMES)'),
        (axes[2], prot, gc_attacks, gc_defs, '(c) Graph Class. (PROTEINS)'),
    ]:
        im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=100)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                norm = v / 100.0
                color = 'white' if (norm < 0.30 or norm > 0.80) else 'black'
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')
        ax.set_xticks(np.arange(len(defs)))
        ax.set_xticklabels(defs, rotation=30, ha='right', fontsize=12)
        ax.set_yticks(np.arange(len(attacks)))
        ax.set_yticklabels(attacks, fontsize=12)
        ax.set_title(title, fontsize=14, pad=8)

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label='Fidelity (%)')
    fig.suptitle('Cross-task generalisation: surrogate fidelity (%) at budget $0.25\\times$',
                 fontsize=15, y=1.02)
    plt.savefig(f'{OUT}/figG_cross_task.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUT}/figG_cross_task.pdf")


# =============================================================
# Main
# =============================================================
if __name__ == '__main__':
    print("Building academic figures from main-paper tables...\n")
    print("[Fig A] RQ1 extension on 3 additional graphs")
    figure_A_rq1_extension()
    print("\n[Fig B] RQ2 watermark profile (radar)")
    figure_B_rq2_radar()
    print("\n[Fig C] Information-limiting defenses (dual heatmap)")
    figure_C_info_limiting_heatmap()
    print("\n[Fig D] Computational cost (combined log bar)")
    figure_D_cost_combined()
    print("\n[Fig E] Joint evaluation on Cora (3-panel heatmap)")
    figure_E_joint_cora()
    print("\n[Fig F] Cross-architecture extraction (2x2 heatmaps)")
    figure_F_cross_arch()
    print("\n[Fig G] Cross-task generalisation (LP + GC heatmaps)")
    figure_G_cross_task()
    print(f"\nAll 7 new figures saved to {OUT}/")
