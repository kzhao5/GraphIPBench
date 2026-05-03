"""Graph structure property analysis (Table NEW-D + correlation analysis).

Computes structural properties for all datasets and correlates them
with attack/defense performance from RQ1 results.

Usage:
  python scripts/run_structure_analysis.py --output-dir outputs/structure_analysis
"""

import argparse
import json
import os
import numpy as np
import torch
from collections import defaultdict


def compute_graph_properties(graph, labels):
    """Compute structural properties of a DGL graph."""
    src, dst = graph.edges()
    src, dst = src.cpu().numpy(), dst.cpu().numpy()
    num_nodes = graph.num_nodes()
    num_edges = len(src) // 2  # undirected, counted twice
    labels_np = labels.cpu().numpy()

    # 1. Edge homophily
    same_label = (labels_np[src] == labels_np[dst]).sum()
    edge_homophily = same_label / len(src) if len(src) > 0 else 0.0

    # 2. Node homophily (average fraction of same-label neighbors per node)
    node_homo = []
    for n in range(num_nodes):
        neighbors = dst[src == n]
        if len(neighbors) > 0:
            same = (labels_np[neighbors] == labels_np[n]).sum()
            node_homo.append(same / len(neighbors))
    node_homophily = np.mean(node_homo) if node_homo else 0.0

    # 3. Average degree
    degrees = graph.in_degrees().float().cpu().numpy()
    avg_degree = degrees.mean()

    # 4. Graph density
    density = 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0

    # 5. Degree distribution entropy
    deg_counts = np.bincount(degrees.astype(int))
    deg_probs = deg_counts / deg_counts.sum()
    deg_probs = deg_probs[deg_probs > 0]
    degree_entropy = -np.sum(deg_probs * np.log2(deg_probs))

    # 6. Clustering coefficient (sample-based for large graphs)
    import networkx as nx
    # Build simple graph from edges (avoid multigraph from DGL)
    g_nx = nx.Graph()
    g_nx.add_nodes_from(range(num_nodes))
    edges = set(zip(src.tolist(), dst.tolist()))
    g_nx.add_edges_from(edges)

    if num_nodes > 50000:
        sample_nodes = np.random.choice(num_nodes, min(5000, num_nodes), replace=False)
        cc_values = [nx.clustering(g_nx, int(n)) for n in sample_nodes]
        clustering_coeff = np.mean(cc_values)
    else:
        clustering_coeff = nx.average_clustering(g_nx)

    # 7. Number of classes
    num_classes = len(np.unique(labels_np))

    # 8. Feature dimension
    num_features = graph.ndata['feat'].shape[1] if 'feat' in graph.ndata else 0

    return {
        'num_nodes': int(num_nodes),
        'num_edges': int(num_edges),
        'num_classes': int(num_classes),
        'num_features': int(num_features),
        'edge_homophily': float(edge_homophily),
        'node_homophily': float(node_homophily),
        'avg_degree': float(avg_degree),
        'density': float(density),
        'degree_entropy': float(degree_entropy),
        'clustering_coeff': float(clustering_coeff),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='outputs/structure_analysis')
    args = parser.parse_args()

    from pygip.datasets import (Cora, CiteSeer, PubMed, Computers, Photo,
                                 CoauthorCS, CoauthorPhysics,
                                 OGBNArxiv, RomanEmpire, AmazonRatings)

    datasets = [
        ('Cora', Cora), ('CiteSeer', CiteSeer), ('PubMed', PubMed),
        ('Computers', Computers), ('Photo', Photo),
        ('CoauthorCS', CoauthorCS), ('CoauthorPhysics', CoauthorPhysics),
        ('OGBNArxiv', OGBNArxiv),
        ('RomanEmpire', RomanEmpire), ('AmazonRatings', AmazonRatings),
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    all_props = {}

    for name, cls in datasets:
        print(f"\nComputing properties for {name}...")
        try:
            ds = cls(api_type='dgl', path='./data')
            graph = ds.graph_data
            labels = graph.ndata['label']
            props = compute_graph_properties(graph, labels)
            props['dataset'] = name
            all_props[name] = props
            print(f"  {props}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Save raw results
    outfile = os.path.join(args.output_dir, 'graph_properties.json')
    with open(outfile, 'w') as f:
        json.dump(all_props, f, indent=2)
    print(f"\nSaved to {outfile}")

    # Print LaTeX-style table
    print("\n" + "=" * 120)
    print("TABLE NEW-D: Graph Structural Properties")
    print("=" * 120)
    header = f"{'Dataset':<18} {'Nodes':>8} {'Edges':>10} {'Classes':>7} {'Features':>8} {'h_edge':>7} {'h_node':>7} {'Avg Deg':>8} {'CC':>6} {'H(deg)':>7}"
    print(header)
    print("-" * len(header))

    for name in ['Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo',
                  'CoauthorCS', 'CoauthorPhysics', 'OGBNArxiv', 'RomanEmpire', 'AmazonRatings']:
        if name not in all_props:
            continue
        p = all_props[name]
        print(f"{name:<18} {p['num_nodes']:>8,} {p['num_edges']:>10,} {p['num_classes']:>7} "
              f"{p['num_features']:>8} {p['edge_homophily']:>7.3f} {p['node_homophily']:>7.3f} "
              f"{p['avg_degree']:>8.2f} {p['clustering_coeff']:>6.3f} {p['degree_entropy']:>7.2f}")

    # Correlation analysis with RQ1 results (if available)
    print("\n" + "=" * 120)
    print("CORRELATION ANALYSIS: Homophily vs Attack Fidelity")
    print("=" * 120)

    rq1_files = [
        'outputs/RQ1_new/RomanEmpire.jsonl',
        'outputs/RQ1_new/AmazonRatings.jsonl',
        'outputs/RQ1_new/OGBNArxiv.jsonl',
    ]

    # Aggregate fidelity per dataset (regime=both, budget=1.0)
    ds_fidelity = {}
    for fpath in rq1_files:
        if not os.path.exists(fpath):
            continue
        ds_name = os.path.basename(fpath).replace('.jsonl', '')
        fids = []
        with open(fpath) as f:
            for line in f:
                r = json.loads(line)
                if r['status'] == 'ok' and r['regime'] == 'both' and r['budget'] == 1.0:
                    fid = r.get('fidelity', -1)
                    if fid > 0:
                        fids.append(fid)
        if fids:
            ds_fidelity[ds_name] = np.mean(fids)

    if ds_fidelity and len(ds_fidelity) >= 2:
        homos = []
        fids = []
        names = []
        for ds_name, fid in ds_fidelity.items():
            if ds_name in all_props:
                homos.append(all_props[ds_name]['edge_homophily'])
                fids.append(fid)
                names.append(ds_name)

        if len(homos) >= 2:
            from scipy.stats import pearsonr, spearmanr
            pr, pp = pearsonr(homos, fids)
            sr, sp = spearmanr(homos, fids)
            print(f"  Pearson(homophily, fidelity):  r={pr:.3f}, p={pp:.3f}")
            print(f"  Spearman(homophily, fidelity): r={sr:.3f}, p={sp:.3f}")
            for n, h, f in zip(names, homos, fids):
                print(f"    {n:<18} homophily={h:.3f}  fidelity={f:.3f}")
    else:
        print("  (Not enough RQ1 data yet for correlation analysis)")


if __name__ == '__main__':
    main()
