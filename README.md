# GraphIP-Bench

A unified benchmark and library for evaluating model-extraction attacks and ownership defenses on graph neural networks under a single reproducible protocol.

> **Anonymous release.** This repository is provided for double-blind peer review. It contains no author or institutional information.

## What is GraphIP-Bench?

GraphIP-Bench standardises the evaluation of two complementary tracks --- the **extraction track** (an adversary trains a surrogate that imitates a deployed GNN) and the **ownership track** (the model owner verifies a watermark or fingerprint after extraction) --- under a single black-box protocol with shared splits, queries, and budgets. The library implements:

- **12 extraction attacks**, including six MEA-style attacks (`MEA0`--`MEA5`), an adversarial-query attack (`AdvMEA`), a centrality-driven attack (`CEGA`), a structure-aware pipeline (`Realistic`), and three data-free variants (`DFEA_I/II/III`).
- **12 defenses**: 5 watermarking and integrity methods (`BackdoorWM`, `RandomWM`, `SurviveWM`, `ImperceptibleWM`, `Integrity`) and 7 information-limiting methods (`OutputPerturbation` low/high, `PredictionRounding` 2-bit/top-1, `PRADA`, `AdaptiveMisinformation`, `GradientRedirection`).
- **10 datasets** spanning four regimes: homophilic citation graphs (Cora, CiteSeer, PubMed), homophilic coauthor and product graphs (CoauthorCS, CoauthorPhysics, Computers, Photo), the large-scale OGBN-Arxiv graph, and two heterophilic graphs (RomanEmpire, AmazonRatings).
- **3 GNN backbones** (GCN, GAT, GraphSAGE) and **3 graph-learning tasks** (node classification, link prediction, graph classification).
- A **joint attack-and-defense track** that runs every extraction attack on every defended target and measures *watermark survival* on the extracted surrogate.

## Repository layout

```
pygip/
  models/
    attack/      # 12 extraction attacks
    defense/     # 12 defenses (5 watermarking + 7 information-limiting)
    nn/          # GNN backbones (GCN, GAT, GraphSAGE)
  datasets/      # Dataset loaders for all 10 graphs
  evaluation/    # Watermark-survival and joint-evaluation utilities
examples/        # Reproduction scripts, one per research question
scripts/         # Figure and table regeneration helpers
requirements.txt
```

## Installation

We recommend a fresh Conda environment with Python 3.11.

```bash
conda create -n graphip python=3.11 -y
conda activate graphip
pip install -r requirements.txt
```

The pinned stack uses PyTorch 2.2.1 (CUDA 12.1), DGL 2.1.0, PyTorch Geometric 2.7.0, and OGB 1.3.6. PyTorch is pinned to 2.2.1 because the matching DGL 2.1.0 graphbolt kernels only ship pre-built shared libraries for PyTorch 2.0--2.2 on CUDA 12.1.

## Quick start

```bash
# RQ1 attack-effectiveness sweep on a single dataset.
python examples/run_node_class_experiments.py --dataset Cora --seed 0

# RQ5 joint attack-and-defense evaluation.
python examples/run_joint_evaluation.py --dataset Cora --budget 0.25 --seed 0

# RQ6 cross-architecture extraction.
python examples/run_cross_arch_node_class.py --dataset Cora --seed 0

# RQ6 link-prediction transfer.
python examples/run_link_pred_experiments.py --seed 0
```

Each script writes one JSON-Lines record per run to `outputs/`.

## Reproducing the paper

Every research question maps to a top-level script under `examples/`:

| Section | Script |
|---|---|
| RQ1 (attack effectiveness vs.\ budget) | `examples/run_node_class_experiments.py` |
| RQ2 / RQ3 (defenses, protection-utility) | `examples/run_defense_experiments.py` |
| RQ4 (efficiency profiling) | `examples/run_efficiency_experiments.py` |
| RQ5 (joint attack-and-defense, watermark survival) | `examples/run_joint_evaluation.py` |
| RQ6 (cross-architecture, link prediction, graph classification, budget grid) | `examples/run_cross_arch_node_class.py`, `examples/run_link_pred_experiments.py`, `examples/run_graph_class_experiments.py`, `examples/run_endpoint_ablation.py` |

All runs use fixed seeds (`0`, `1`, `2`) and the shared query sets defined in `pygip/datasets/`, so a single-seed run is directly comparable to the corresponding cell in the paper.

## Datasets

All datasets are downloaded automatically on first use and cached under `data/`. Planetoid (Cora, CiteSeer, PubMed), Amazon (Computers, Photo), and Coauthor (CS, Physics) are loaded through DGL; OGBN-Arxiv through the `ogb` package; RomanEmpire and AmazonRatings through DGL's `RomanEmpireDataset` and `AmazonRatingsDataset`; ENZYMES and PROTEINS through TUDataset in PyG.

## Hardware

All reported results use a single NVIDIA A100 80 GB GPU with CUDA 12.1. The lightweight attacks and defenses run comfortably on a single GPU with at most 16 GB; only the `Realistic` and `ImperceptibleWM` pipelines require the full 80 GB allocation.

## Licence

Released under the MIT licence.
