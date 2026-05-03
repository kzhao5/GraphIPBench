"""Watermark survival evaluator for testing watermark persistence on extracted surrogates.

Given a defense instance (with watermark artifacts) and a surrogate model
extracted by an attack, this evaluator checks whether the watermark transfers
to the surrogate.
"""

import torch
import dgl
from pygip.models.nn.backbones import GCN, GraphSAGE, model_forward


class WatermarkSurvivalEvaluator:
    """Unified evaluator for watermark survival on extracted surrogate models."""

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    def evaluate(self, defense, surrogate_model):
        """Evaluate watermark survival on a surrogate model.

        Args:
            defense: A defense instance after defend() has been called.
                     Must have watermark artifacts stored as instance variables.
            surrogate_model: The extracted surrogate model to verify.

        Returns:
            dict with:
                'wm_acc': float - watermark accuracy on surrogate (0.0-1.0)
                'defense_name': str - name of the defense method
                'details': dict - defense-specific details
        """
        defense_name = defense.__class__.__name__

        if defense_name == 'BackdoorWM':
            return self._eval_backdoor(defense, surrogate_model)
        elif defense_name == 'RandomWM':
            return self._eval_random(defense, surrogate_model)
        elif defense_name == 'SurviveWM':
            return self._eval_survive(defense, surrogate_model)
        elif defense_name == 'ImperceptibleWM':
            return self._eval_imperceptible(defense, surrogate_model)
        elif defense_name in ('Integrity', 'QueryBasedVerificationDefense'):
            return self._eval_integrity(defense, surrogate_model)
        else:
            raise ValueError(f"Unsupported defense: {defense_name}")

    def _eval_backdoor(self, defense, surrogate_model):
        """BackdoorWM: Check if surrogate predicts target_label on trigger nodes."""
        surrogate_model.eval()
        graph = defense.graph_data.to(self.device)
        features = defense.poisoned_features.to(self.device)
        trigger_nodes = defense.trigger_nodes
        target_label = defense.target_label

        with torch.no_grad():
            out = model_forward(surrogate_model, graph, features)
            preds = out.argmax(dim=1)[trigger_nodes]
            target = torch.full_like(preds, target_label)
            wm_acc = (preds == target).float().mean().item()

        return {
            'wm_acc': wm_acc,
            'defense_name': 'BackdoorWM',
            'details': {
                'num_trigger_nodes': len(trigger_nodes),
                'target_label': target_label,
                'correct': int((preds == target).sum()),
                'total': len(trigger_nodes),
            }
        }

    def _eval_random(self, defense, surrogate_model):
        """RandomWM: Check surrogate accuracy on the watermark graph."""
        # RandomWM already has _evaluate_attack_on_watermark() but it's
        # tightly coupled. We reimplement for generality.
        surrogate_model.eval()
        wm_graph = defense.watermark_graph.to(self.device)
        wm_features = wm_graph.ndata['feat']
        wm_labels = wm_graph.ndata['label'].to(self.device)

        with torch.no_grad():
            out = model_forward(surrogate_model, wm_graph, wm_features)
            preds = out.argmax(dim=1)
            wm_acc = (preds == wm_labels).float().mean().item()

        return {
            'wm_acc': wm_acc,
            'defense_name': 'RandomWM',
            'details': {
                'num_wm_nodes': wm_graph.num_nodes(),
                'correct': int((preds == wm_labels).sum()),
                'total': wm_graph.num_nodes(),
            }
        }

    def _eval_survive(self, defense, surrogate_model):
        """SurviveWM: Check surrogate accuracy on the trigger graph."""
        surrogate_model.eval()

        # Build trigger DGL graph from stored trigger_data
        if hasattr(defense, 'trigger_graph') and defense.trigger_graph is not None:
            trigger_graph = defense.trigger_graph.to(self.device)
        else:
            # Convert from PyG trigger_data
            td = defense.trigger_data
            trigger_graph = dgl.graph((td.edge_index[0], td.edge_index[1]),
                                      num_nodes=td.num_nodes)
            trigger_graph = dgl.add_self_loop(trigger_graph)
            trigger_graph.ndata['feat'] = td.x
            trigger_graph = trigger_graph.to(self.device)

        trigger_features = trigger_graph.ndata['feat'].to(self.device)
        trigger_labels = defense.trigger_data.y.to(self.device)

        with torch.no_grad():
            out = model_forward(surrogate_model, trigger_graph, trigger_features)
            preds = out.argmax(dim=1)
            wm_acc = (preds == trigger_labels).float().mean().item()

        return {
            'wm_acc': wm_acc,
            'defense_name': 'SurviveWM',
            'details': {
                'num_trigger_nodes': trigger_graph.num_nodes(),
                'correct': int((preds == trigger_labels).sum()),
                'total': trigger_graph.num_nodes(),
            }
        }

    def _eval_imperceptible(self, defense, surrogate_model):
        """ImperceptibleWM: Regenerate trigger graph and test surrogate."""
        surrogate_model.eval()

        # ImperceptibleWM stores the underlying graph as `defense.graph_data`
        # which is already a PyG Data object (converted from DGL in __init__).
        from pygip.models.defense.ImperceptibleWM import generate_trigger_graph
        trigger_data = generate_trigger_graph(
            defense.graph_data, defense.generator, defense.model,
            defense.num_triggers
        )
        trigger_nodes = trigger_data.trigger_nodes

        # Convert PyG trigger data to DGL for surrogate evaluation
        edge_index = trigger_data.edge_index
        num_nodes = trigger_data.x.size(0)
        trigger_dgl = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
        trigger_dgl = dgl.add_self_loop(trigger_dgl)
        trigger_dgl = trigger_dgl.to(self.device)
        trigger_features = trigger_data.x.to(self.device)

        # Get the defended model's predictions on trigger nodes (expected labels)
        defense.model.eval()
        with torch.no_grad():
            def_out = defense.model(trigger_data.x.to(self.device),
                                     trigger_data.edge_index.to(self.device))
            expected_labels = def_out[trigger_nodes].argmax(dim=1)

        # Get surrogate predictions
        with torch.no_grad():
            sur_out = model_forward(surrogate_model, trigger_dgl, trigger_features)
            sur_preds = sur_out[trigger_nodes].argmax(dim=1)
            wm_acc = (sur_preds == expected_labels).float().mean().item()

        return {
            'wm_acc': wm_acc,
            'defense_name': 'ImperceptibleWM',
            'details': {
                'num_trigger_nodes': len(trigger_nodes),
                'correct': int((sur_preds == expected_labels).sum()),
                'total': len(trigger_nodes),
            }
        }

    def _eval_integrity(self, defense, surrogate_model):
        """Integrity: Check if surrogate preserves fingerprint predictions."""
        surrogate_model.eval()

        # Regenerate fingerprints using the defended model.
        # _generate_fingerprints signature: (self, model, mode='transductive', knowledge='full', k=5)
        try:
            fingerprints = defense._generate_fingerprints(
                defense.model, mode='transductive', knowledge='full', k=defense.k
            )
        except Exception as e:
            return {
                'wm_acc': -1.0,
                'defense_name': 'Integrity',
                'details': {'error': f'fingerprint regen failed: {e}'},
            }

        flipped = 0
        total = len(fingerprints)

        # Each fingerprint is a tuple — handle both 3-tuple (graph, node_id, label)
        # and dict-like records.
        with torch.no_grad():
            for fp in fingerprints:
                if isinstance(fp, (tuple, list)) and len(fp) >= 3:
                    graph, node_id, expected_label = fp[0], fp[1], fp[2]
                elif isinstance(fp, dict):
                    graph = fp.get('graph') or fp.get('subgraph')
                    node_id = fp.get('node_id') or fp.get('node')
                    expected_label = fp.get('label') or fp.get('expected_label')
                else:
                    continue
                if graph is None or node_id is None:
                    continue
                graph = graph.to(self.device)
                features = graph.ndata['feat'].to(self.device)
                out = model_forward(surrogate_model, graph, features)
                pred = out[node_id].argmax().item()
                if pred != int(expected_label):
                    flipped += 1

        # For integrity, wm_acc = 1 - flip_rate (fingerprints should NOT flip)
        wm_acc = 1.0 - (flipped / total) if total > 0 else 0.0

        return {
            'wm_acc': wm_acc,
            'defense_name': 'Integrity',
            'details': {
                'num_fingerprints': total,
                'flipped': flipped,
                'flip_rate': flipped / total if total > 0 else 0.0,
            }
        }
