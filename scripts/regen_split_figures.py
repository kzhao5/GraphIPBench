"""Split-panel figure regeneration for tight LaTeX subfigure composition.

For figures that originally had multiple panels in one PDF, we now emit one
PDF per panel so that LaTeX can pack them with \\subfigure and minimal whitespace.

Output naming convention:
  figE_joint_a_orig.pdf    (was Fig E panel a)
  figE_joint_b_new.pdf     (was Fig E panel b)
  figE_joint_c_survival.pdf
  figF_cross_arch_cora.pdf
  figF_cross_arch_computers.pdf
  figF_cross_arch_ogbn.pdf
  figF_cross_arch_roman.pdf
  figG_cross_task_lp.pdf
  figG_cross_task_enzymes.pdf
  figG_cross_task_proteins.pdf
"""

import os
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 16,
    'axes.linewidth': 1.0,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})

OUT = 'outputs/figures_v2'
os.makedirs(OUT, exist_ok=True)

ATTACKS_12 = ['MEA0','MEA1','MEA2','MEA3','MEA4','MEA5','AdvMEA','CEGA','Realistic',
              'DFEA_I','DFEA_II','DFEA_III']

# ============================================================
# Fig E — Joint evaluation on Computers: 3 separate panels
# (Computers is the default dataset for in-text RQ examples,
#  matching Table 1 / RQ1 main table.)
# ============================================================
def figE_split():
    wm_labels = ['Back','Surv','Integ','Rand','Imp']
    nonwm_labels = ['OP_low','OP_high','PR_2bit','PR_top1','PRADA','AdaptM','GradR']
    # Surrogate fidelity (%) on Computers, 5 watermarking defenses
    # (rows = MEA0, MEA1, MEA2, MEA3, MEA4, MEA5, AdvMEA, CEGA, Realistic, DFEA_I, DFEA_II, DFEA_III)
    table7 = np.array([
        [80.0,82.6,99.6,88.4,88.4],[78.3,77.7,100.0,92.0,87.9],[44.0,59.3,44.8,34.1,27.0],
        [84.2,83.5,100.0,88.7,84.7],[82.5,80.2,100.0,89.1,85.9],[80.0,81.0,99.9,87.4,83.9],
        [48.7,54.5,100.0,56.0,42.1],[70.3,68.4,66.7,84.5,85.1],[39.6,33.9,43.5,44.2,79.6],
        [36.8,32.0,51.1,17.0,28.3],[20.8,40.7,100.0,27.8,29.3],[ 9.9,32.1,100.0,28.6,31.4],
    ])
    # Surrogate fidelity (%) on Computers, 7 information-limiting and query-detection defenses
    table8 = np.array([
        [73.4,62.6,75.7,82.5,24.9,52.9,82.0],[72.9,66.6,84.8,83.7,34.6,55.4,84.3],
        [41.9,34.3,17.5,42.5,18.6,12.2,50.7],[79.7,67.9,83.5,82.5,31.4,52.6,84.3],
        [77.5,64.4,77.4,80.9,31.3,54.2,84.0],[71.8,56.3,84.4,81.6,27.5,50.1,82.0],
        [43.3,34.4,44.7,34.3,23.1,37.1,28.5],[55.8,44.3,48.7,82.2,27.8,38.7,71.7],
        [80.1,76.4,88.7,83.5,38.5,56.9,87.7],
        [22.8,17.3, 6.9,28.4,17.3, 8.7,28.4],[23.1,17.4,22.2,28.4,17.2,23.2,28.4],
        [23.2,17.3,22.2,28.4,17.1,19.5,28.4],
    ])
    # Watermark survival rate (%) on the surrogate, on Computers
    table9 = np.array([
        [60.0, 9.9,100.0,15.7,0.0],[51.7, 9.6,100.0,15.3,0.0],[31.7,10.6, 33.3,12.0,0.0],
        [60.0,10.0,100.0,14.7,0.0],[58.3,10.1,100.0,14.7,0.0],[55.0,10.2,100.0,12.0,0.0],
        [65.0,10.4,100.0,12.3,0.0],[61.7, 9.8, 66.7,16.0,0.0],[ 5.0,10.2, 13.2, 9.0,0.0],
        [100.0,10.0, 45.6,12.0,0.0],[66.7,10.6,100.0,11.3,0.0],[ 0.0,10.5,100.0,18.0,0.0],
    ])
    cmap_fid = plt.get_cmap('RdYlGn')
    cmap_surv = plt.get_cmap('Reds')

    # Reorder attacks so MEA family then heuristic then data-free for visual story
    attack_order = ATTACKS_12

    def make_panel(filename, mat, col_labels, vmin, vmax, title, cmap, cbar_label):
        fig, ax = plt.subplots(figsize=(4.0, 5.6))
        im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                norm = (v - vmin) / max(vmax - vmin, 1e-9)
                color = 'white' if (norm < 0.30 or norm > 0.80) else 'black'
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=11)
        ax.set_yticks(np.arange(len(attack_order)))
        ax.set_yticklabels(attack_order, fontsize=11)
        ax.set_title(title, fontsize=13, pad=6)
        cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.025)
        cbar.set_label(cbar_label, fontsize=11)
        cbar.ax.tick_params(labelsize=10)
        plt.tight_layout()
        plt.savefig(f'{OUT}/{filename}')
        plt.close()
        print(f"  Saved {OUT}/{filename}")

    make_panel('figE_joint_a_orig.pdf', table7, wm_labels, 0, 100,
               'Watermark defenses', cmap_fid, 'Fidelity (%)')
    make_panel('figE_joint_b_new.pdf', table8, nonwm_labels, 0, 100,
               'Information-limiting defenses', cmap_fid, 'Fidelity (%)')
    make_panel('figE_joint_c_survival.pdf', table9, wm_labels, 0, 100,
               'Watermark survival on surrogate', cmap_surv, 'Survival (%)')


# ============================================================
# Fig F — Cross-architecture extraction: 4 separate mini heatmaps
# ============================================================
def figF_split():
    archs = ['GCN', 'GAT', 'SAGE']
    data = {
        'cora':       np.array([[87.5, 89.4, 88.5],[83.3, 90.6, 87.0],[84.0, 86.0, 96.0]]),
        'computers':  np.array([[75.8, 60.5, 45.9],[76.1, 89.8, 77.0],[60.7, 81.2, 72.9]]),
        'ogbn':       np.array([[83.1, 62.4, 63.0],[44.9, 80.7, 90.9],[41.9, 71.4, 90.3]]),
        'roman':      np.array([[63.1, 74.4, 82.7],[53.9, 76.7, 78.3],[38.5, 47.9, 92.3]]),
    }
    titles = {
        'cora': 'Cora (homophilic, h=0.81)',
        'computers': 'Computers (deg=37, h=0.78)',
        'ogbn': 'OGBN-Arxiv (169K, 40 cls)',
        'roman': 'RomanEmpire (h=0.29)',
    }
    cmap = plt.get_cmap('RdYlGn')

    for key, mat in data.items():
        fig, ax = plt.subplots(figsize=(3.8, 3.6))
        im = ax.imshow(mat, aspect='equal', cmap=cmap, vmin=35, vmax=100)
        for i in range(3):
            for j in range(3):
                v = mat[i, j]
                norm = (v - 35) / 65.0
                color = 'white' if (norm < 0.30 or norm > 0.80) else 'black'
                weight = 'bold' if i == j else 'normal'
                ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                        fontsize=12, color=color, fontweight=weight)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(archs, fontsize=11)
        ax.set_yticks(np.arange(3))
        ax.set_yticklabels(archs, fontsize=11)
        ax.set_xlabel('Surrogate backbone', fontsize=11)
        ax.set_ylabel('Target backbone', fontsize=11)
        ax.set_title(titles[key], fontsize=12, fontweight='bold', pad=6)
        for i in range(3):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor='black', linewidth=1.6))
        plt.tight_layout()
        plt.savefig(f'{OUT}/figF_cross_arch_{key}.pdf')
        plt.close()
        print(f"  Saved {OUT}/figF_cross_arch_{key}.pdf")

    # Stand-alone colorbar for the 4-panel composition (use same vmin/vmax)
    fig, ax = plt.subplots(figsize=(0.55, 3.2))
    norm = plt.matplotlib.colors.Normalize(vmin=35, vmax=100)
    sm = plt.matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=ax)
    cb.set_label('Fidelity (%)', fontsize=11)
    cb.ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(f'{OUT}/figF_cross_arch_cbar.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUT}/figF_cross_arch_cbar.pdf")


# ============================================================
# Fig G — Cross-task generalization: 3 separate heatmaps
# ============================================================
def figG_split():
    lp_attacks = ['MEA0','MEA1','MEA2','MEA3','MEA4','MEA5','AdvMEA','CEGA','DFEA_I','DFEA_II','DFEA_III']
    lp_defs = ['None','OP_low','OP_high','PR_2bit','GradR']
    lp = np.array([
        [95.3,96.1,94.2,84.9,70.9],[94.8,93.7,92.8,87.2,71.9],[95.6,92.8,90.4,28.9,94.3],
        [73.1,71.0,70.2,68.8,70.4],[85.3,84.4,81.6,28.9,79.2],[93.8,92.4,89.3,81.9,71.1],
        [97.9,96.0,93.1,82.4,80.7],[92.1,92.0,92.8,87.8,71.0],
        [94.4,94.4,89.9,28.9,92.6],[94.0,92.1,91.3,28.9,93.4],[94.4,92.9,91.1,28.9,28.9],
    ])
    gc_attacks = ['MEA0','MEA1','AdvMEA','CEGA','DFEA_I','DFEA_II']
    gc_defs = ['None','OP_low','OP_high','PR_2bit','PR_top1','GradR']
    enz = np.array([
        [92.2,88.9,86.9,16.7,91.1,90.9],[29.4,28.1,24.4, 3.1,32.4,29.6],
        [84.6,85.6,83.0,16.5,86.1,85.6],[88.7,92.2,90.2,59.1,90.9,85.0],
        [93.0,91.3,91.1,60.0,91.3,83.0],[90.6,90.2,84.1,16.7,91.3,91.7],
    ])
    prot = np.array([
        [95.5,90.9,96.2,83.8,96.2,94.9],[63.4,61.9,63.3,62.4,65.1,63.2],
        [90.2,88.7,89.9,85.0,91.6,92.5],[97.1,96.8,97.5,94.7,97.8,97.1],
        [97.5,97.4,97.9,95.0,97.7,96.6],[96.3,94.7,94.2,84.9,95.9,95.5],
    ])
    cmap = plt.get_cmap('RdYlGn')

    def make(filename, mat, attacks, defs, title, w=4.2, h=5.5):
        fig, ax = plt.subplots(figsize=(w, h))
        im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=100)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                norm = v / 100.0
                color = 'white' if (norm < 0.30 or norm > 0.80) else 'black'
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')
        ax.set_xticks(np.arange(len(defs)))
        ax.set_xticklabels(defs, rotation=30, ha='right', fontsize=11)
        ax.set_yticks(np.arange(len(attacks)))
        ax.set_yticklabels(attacks, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=6)
        cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.025)
        cbar.set_label('Fidelity (%)', fontsize=11)
        cbar.ax.tick_params(labelsize=10)
        plt.tight_layout()
        plt.savefig(f'{OUT}/{filename}')
        plt.close()
        print(f"  Saved {OUT}/{filename}")

    make('figG_lp_cora.pdf', lp, lp_attacks, lp_defs, 'Link prediction (Cora)', w=4.0, h=5.8)
    make('figG_gc_enzymes.pdf', enz, gc_attacks, gc_defs, 'Graph class. (ENZYMES)', w=4.4, h=4.0)
    make('figG_gc_proteins.pdf', prot, gc_attacks, gc_defs, 'Graph class. (PROTEINS)', w=4.4, h=4.0)


if __name__ == '__main__':
    print('Splitting multi-panel figures into individual PDFs for LaTeX subfigure...\n')
    print('[Fig E] Joint evaluation panels (3 PDFs)')
    figE_split()
    print('\n[Fig F] Cross-arch mini heatmaps (4 PDFs + colorbar)')
    figF_split()
    print('\n[Fig G] Cross-task heatmaps (3 PDFs)')
    figG_split()
    print(f'\nAll split panels saved to {OUT}/')
