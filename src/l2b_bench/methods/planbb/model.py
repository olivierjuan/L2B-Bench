"""PlanB&B — the MuZero-style h/g/f networks over bipartite graphs.

Faithful port of ``config/milp/{model,gcnn}.py`` and ``core/{model,batch}.py``
from the official repository (github.com/abfariah/planbb, the
``EfficientZeroNet`` used with ``tree_mdp``):

- ``RepresentationNetwork`` (h) — LayerNorm+MLP encoders for the 43 variable
  and 5 constraint features, LayerNorm for the scalar edge features.
- ``DynamicsNetwork`` (g) — the two latent children of a branching decision:
  the taken variable's embedding is added to its row, then two residual
  bipartite convolutions (v→c, then c→v) per side. No reward module (the
  official default).
- ``PredictionNetwork`` (f) — two more convolutions and a shared MLP, with
  per-variable heads: one policy logit, 18 HL-Gauss value bins, a 2-way
  branchability.
- ``PlanBBNet.initial_inference``/``recurrent_inference`` — the official
  pooling (mean over variables × the padding correction), the candidate
  masking at the root, and the dual nature of the value output: bin *logits*
  in tree-MDP training mode (for the HL-Gauss cross-entropy), a decoded
  ``-2^E[centers]`` scalar otherwise (search and target bootstrap). In MCTS
  mode the branchability softmax multiplies the value and everything comes
  back as numpy.
- ``TreeHLGaussLoss`` — the official value transform over log2(-z), with the
  tree-MDP cross-entropy branch.

The batch side ports ``core/batch.py``: ``BipartiteNodeData`` (a
torch_geometric ``Data`` whose ``edge_index``/``candidates``/``actions``
increment per graph), ``geometric_batch``, ``format_batch_actions``,
``pad_tensor`` (-6e4 except the 2/18/25-wide outputs), ``pad_proj``,
``pad_array``, ``get_idx``, ``get_target_index`` — on the harness's
``NodeBipartitePlanBB`` observations instead of ecole's.
"""

from typing import List, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch

from ..bbmdp.model import TreeHLGaussLoss as _BBMDPHLGauss  # same math, re-exported shape

#: The NodeBipartitePlanBB feature dimensions.
N_ROW_FEATURES = 5
N_VAR_FEATURES = 43
NUM_BINS = 18

#: The official root-masking constant and pad value (core/model.py).
MASK_VALUE = -6e4


def mask_action_set(action_set: np.ndarray, size: int) -> np.ndarray:
    """The official binary candidate mask over all variables."""
    binary = np.zeros(size, dtype=np.bool_)
    binary[action_set] = True
    return binary


class PlanBBObs:
    """One planbb observation: the harness's ``NodeBipartitePlanBB`` graph
    plus its branching candidates (the official ``MuZeroNodeBipartiteObs``,
    which reads the same 43/5/1 features from ecole)."""

    def __init__(self, obs, action_set: np.ndarray) -> None:
        self.variable_features = np.asarray(obs.variable_features, dtype=np.float32)
        self.row_features = np.asarray(obs.row_features, dtype=np.float32)
        self.edge_indices = np.asarray(obs.edge_features.indices)
        self.edge_features = np.asarray(obs.edge_features.values, dtype=np.float32)
        self.action_set = np.asarray(action_set)
        self.masked_action_set = mask_action_set(self.action_set, self.variable_features.shape[0])


class BipartiteNodeData(torch_geometric.data.Data):
    """One (hidden or raw) bipartite graph, batched by torch_geometric —
    ``edge_index`` increments by (n_constraints, n_variables) and the
    variable-indexed fields by n_variables per graph."""

    def __init__(
        self,
        constraint_features: torch.FloatTensor = None,
        edge_indices: torch.LongTensor = None,
        edge_features: torch.FloatTensor = None,
        variable_features: torch.FloatTensor = None,
        candidates: torch.LongTensor = None,
        candidates_mask: torch.BoolTensor = None,
        action: torch.LongTensor = None,
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.nb_variables = variable_features.size(0) if variable_features is not None else 0
        self.nb_constraints = constraint_features.size(0) if variable_features is not None else 0
        self.candidates = candidates
        self.candidates_mask = candidates_mask
        self.nb_candidates = candidates.size(0) if candidates is not None else 0
        self.actions = action if action is not None else 0

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == "edge_index":
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key in ["candidates", "actions", "candidates_mask"]:
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


def format_batch_actions(action_idxs: torch.LongTensor, nb_variables: torch.LongTensor) -> torch.LongTensor:
    """Per-sample action indices → indices in the collated variable block."""
    prefix = torch.cumsum(nb_variables, dim=0) - nb_variables
    return prefix + action_idxs


def get_target_index(unroll_steps: int) -> torch.Tensor:
    """The breward-loss slots: every obs of the unrolled window except the
    left/right children of step 0 (indices 3k, k>0)."""
    length = 3 * unroll_steps + 1
    target_index = torch.arange(length).long()
    return target_index[(target_index % 3 != 0) | (target_index == 0)]


def geometric_batch(observations: List[PlanBBObs]) -> BipartiteNodeData:
    """Collate planbb observations into one disconnected bipartite graph
    (the official ``geometric_batch``)."""
    graphs = []
    for obs in observations:
        edge_weights = np.expand_dims(obs.edge_features.astype(np.float32), axis=-1)
        graph = BipartiteNodeData(
            torch.FloatTensor(obs.row_features.astype(np.float32)),
            torch.LongTensor(obs.edge_indices.astype(np.int64)),
            torch.FloatTensor(edge_weights),
            torch.FloatTensor(obs.variable_features.astype(np.float32)),
            torch.LongTensor(obs.action_set.astype(np.int64)),
            torch.tensor(obs.masked_action_set, dtype=torch.bool),
        )
        graph.num_nodes = obs.row_features.shape[0] + obs.variable_features.shape[0]
        graphs.append(graph)
    return Batch.from_data_list(graphs)


def get_idx(obs_idx_lst: List[List[int]]) -> List[List[int]]:
    """Positions of each window obs inside its own window (the official
    ``get_idx`` — ``index`` finds the *first* occurrence, so a repeated obs
    maps to its first slot)."""
    return [[obs_idx.index(obs) for obs in obs_idx] for obs_idx in obs_idx_lst]


def pad_proj(input_: torch.Tensor, pad_sizes: torch.Tensor) -> torch.Tensor:
    """Pad the SimSiam projections to one rectangle with zeros."""
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    return pad_sequence(output, batch_first=True, padding_value=0)


def pad_tensor(input_: torch.Tensor, pad_sizes: torch.Tensor, pad_value: float = None) -> torch.Tensor:
    """Per-sample blocks padded to one rectangle — -6e4 so an argmax or a
    masked softmax stays inside the sample's own variables, except the
    2/18/25-wide outputs (breward, value bins, reward) which pad with 0."""
    output_dim = input_.shape[-1]
    if pad_value is None:
        pad_value = -6e4 if output_dim not in [2, 18, 25] else 0
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    return pad_sequence(output, batch_first=True, padding_value=pad_value)


def _pad_all(obs, actor_logits, pseudo_q_value, breward):
    """Pad the per-variable heads to one rectangle — unconditionally, so a
    batch of one keeps the [B, ...] shapes the official only ever sees with
    its parallel actors (``env_nums`` >= 2); the released sequential port
    would otherwise feed 1-D logits into a pipeline that indexes ``[i][j]``."""
    nb_variables = obs.nb_variables
    if not torch.is_tensor(nb_variables):
        nb_variables = torch.tensor([nb_variables])
    return (
        pad_tensor(actor_logits, nb_variables),
        pad_tensor(pseudo_q_value, nb_variables, pad_value=0),
        pad_tensor(breward, nb_variables),
    )


def pad_array(input_: List[np.ndarray], pad_sizes: List[int]) -> np.ndarray:
    """Zero-pad a list of per-sample policy rows to one rectangle."""
    max_size = max(pad_sizes)
    padded = []
    for arr in input_:
        pad_width = [(0, 0), (0, max_size - arr.shape[-1])]
        padded.append(np.pad(arr, pad_width, mode="constant", constant_values=0))
    return np.array(padded)


class PlanBipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """The official bipartite convolution: one message per edge, aggregated
    with a sum, LayerNorm + LeakyReLU everywhere (no Gasse pre-norm)."""

    def __init__(self, aggregator="add", emb_size: int = 64) -> None:
        super().__init__(aggregator)
        self.feature_module_left = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size))
        self.feature_module_edge = torch.nn.Sequential(torch.nn.Linear(1, emb_size, bias=False))
        self.feature_module_right = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size, bias=False))
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size)
        )
        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(
                self.feature_module_left(left_features),
                self.feature_module_right(right_features),
            ),
            edge_features=self.feature_module_edge(edge_features),
        )
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        return self.feature_module_final(node_features_i + node_features_j + edge_features)


class RepresentationNetwork(torch.nn.Module):
    """h — the raw observation into a latent bipartite graph."""

    def __init__(
        self,
        cons_nfeats: int = N_ROW_FEATURES,
        edge_nfeats: int = 1,
        var_nfeats: int = N_VAR_FEATURES,
        emb_size: int = 64,
    ) -> None:
        super().__init__()
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )
        self.edge_embedding = torch.nn.Sequential(torch.nn.LayerNorm(edge_nfeats))
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )

    def forward(self, obs: BipartiteNodeData) -> BipartiteNodeData:
        hidden_obs = obs.clone()
        hidden_obs.constraint_features = self.cons_embedding(obs.constraint_features)
        hidden_obs.edge_attr = self.edge_embedding(obs.edge_attr)
        hidden_obs.variable_features = self.var_embedding(obs.variable_features)
        return hidden_obs


class DynamicsNetwork(torch.nn.Module):
    """g — the two latent children of a branching decision on the taken
    variable (the official ``forward`` over batched graphs)."""

    def __init__(self, emb_size: int = 64) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.left_child_action_embedding = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )
        self.right_child_action_embedding = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )
        self.left_child_conv_v_to_c = PlanBipartiteGraphConvolution(emb_size=emb_size)
        self.left_child_conv_c_to_v = PlanBipartiteGraphConvolution(emb_size=emb_size)
        self.right_child_conv_v_to_c = PlanBipartiteGraphConvolution(emb_size=emb_size)
        self.right_child_conv_c_to_v = PlanBipartiteGraphConvolution(emb_size=emb_size)

    def forward(self, obs: BipartiteNodeData, actions: torch.Tensor) -> Tuple[BipartiteNodeData, BipartiteNodeData]:
        assert actions.size(0) == obs.batch_size
        left_obs, right_obs = obs.clone(), obs.clone()
        action_variable_features = obs.variable_features[actions]
        reversed_edge_index = torch.stack([obs.edge_index[1], obs.edge_index[0]], dim=0)

        left_obs.variable_features[actions] = (
            left_obs.variable_features[actions] + self.left_child_action_embedding(action_variable_features)
        )
        right_obs.variable_features[actions] = (
            right_obs.variable_features[actions] + self.right_child_action_embedding(action_variable_features)
        )
        left_obs.constraint_features = obs.constraint_features + self.left_child_conv_v_to_c(
            left_obs.variable_features, reversed_edge_index, obs.edge_attr, left_obs.constraint_features
        )
        right_obs.constraint_features = obs.constraint_features + self.right_child_conv_v_to_c(
            right_obs.variable_features, reversed_edge_index, obs.edge_attr, right_obs.constraint_features
        )
        left_obs.variable_features = obs.variable_features + self.left_child_conv_c_to_v(
            left_obs.constraint_features, obs.edge_index, obs.edge_attr, left_obs.variable_features
        )
        right_obs.variable_features = obs.variable_features + self.right_child_conv_c_to_v(
            right_obs.constraint_features, obs.edge_index, obs.edge_attr, right_obs.variable_features
        )
        return left_obs, right_obs


class PredictionNetwork(torch.nn.Module):
    """f — per-variable policy logit, 18-bin value, 2-way branchability."""

    def __init__(self, emb_size: int = 64) -> None:
        super().__init__()
        self.prediction_conv_v_to_c = PlanBipartiteGraphConvolution(emb_size=emb_size)
        self.prediction_conv_c_to_v = PlanBipartiteGraphConvolution(emb_size=emb_size)
        self.num_bins = NUM_BINS
        self.common_output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )
        self.value_output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, self.num_bins),
        )
        self.policy_output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, 1),
        )
        self.breward_output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, 2),
        )

    def forward(self, obs: BipartiteNodeData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reversed_edge_indices = torch.stack([obs.edge_index[1], obs.edge_index[0]], dim=0)
        constraint_features = self.prediction_conv_v_to_c(
            obs.variable_features, reversed_edge_indices, obs.edge_attr, obs.constraint_features
        )
        variable_features = self.prediction_conv_c_to_v(
            constraint_features, obs.edge_index, obs.edge_attr, obs.variable_features
        )
        output = self.common_output_module(variable_features).squeeze(-1)
        return (
            self.policy_output_module(output),
            self.value_output_module(output),
            self.breward_output_module(output),
        )


class NetworkOutput(NamedTuple):
    policy_logits: object
    value: object
    breward: object
    hidden_state: object
    batch_size: int


class PlanBBNet(torch.nn.Module):
    """The official ``EfficientZeroNet`` with the released SimSiam dimensions
    (projection 64→16, predictor 16→16) and the tree-MDP inference modes."""

    def __init__(self, proj_hid: int = 64, proj_out: int = 16, pred_hid: int = 16, pred_out: int = 16) -> None:
        super().__init__()
        self.mcts = False  # mcts_mode(): numpy outputs, breward-scaled values
        self.representation_network = RepresentationNetwork()
        self.dynamics_network = DynamicsNetwork()
        self.prediction_network = PredictionNetwork()
        self.value_loss = TreeHLGaussLoss()

        self.projection_in_dim = 64
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(self.projection_in_dim, proj_hid),
            torch.nn.LayerNorm(proj_hid),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(proj_hid, proj_hid),
            torch.nn.LayerNorm(proj_hid),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(proj_hid, proj_out),
            torch.nn.LayerNorm(proj_out),
        )
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(proj_out, pred_hid),
            torch.nn.LayerNorm(pred_hid),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(pred_hid, pred_out),
        )

    # -- the three MuZero functions ---------------------------------------

    def representation(self, obs: BipartiteNodeData) -> BipartiteNodeData:
        return self.representation_network(obs)

    def dynamics(self, obs: BipartiteNodeData, actions: torch.Tensor):
        return self.dynamics_network(obs, actions)

    def prediction(self, obs: BipartiteNodeData):
        return self.prediction_network(obs)

    # -- inference --------------------------------------------------------

    def initial_inference(
        self, obs: BipartiteNodeData, root: bool = False, tree_mdp: bool = False
    ) -> NetworkOutput:
        """h + f on a batch of graphs. ``value`` is bin *logits* in tree-MDP
        mode (the HL-Gauss CE consumes them), a decoded scalar otherwise."""
        batch_size = obs.batch_size
        hidden_obs = self.representation(obs)
        actor_logits, pseudo_q_value, breward = self.prediction(hidden_obs)

        if root:  # mask the non-candidates
            actor_mask = torch.ones(obs.variable_features.shape[0], dtype=torch.bool)
            actor_mask[obs.candidates] = False
            actor_logits[actor_mask] = MASK_VALUE

        actor_logits, pseudo_q_value, breward = _pad_all(obs, actor_logits, pseudo_q_value, breward)

        # The official pooling: mean over the variables times the padding
        # correction, so the padded mean equals the true mean.
        max_variables = pseudo_q_value.size(-2)
        mean_correction = max_variables / obs.nb_variables
        value = pseudo_q_value.mean(dim=-2) * mean_correction.unsqueeze(-1)

        if not (root and tree_mdp):
            value = self.value_loss.value_transform(value)

        breward = torch.mean(breward, dim=-2)

        if self.mcts:
            breward = F.softmax(breward, dim=-1)[..., 1].unsqueeze(-1)
            value = value * breward
            hidden_obs = hidden_obs.detach().cpu()
            value = value.detach().cpu().numpy()
            actor_logits = actor_logits.squeeze(-1).cpu().numpy()
            breward = breward.detach().cpu().numpy()
        else:
            actor_logits = actor_logits.squeeze(-1)

        return NetworkOutput(actor_logits, value, breward, hidden_obs, batch_size)

    def recurrent_inference(
        self, obs: BipartiteNodeData, actions: torch.Tensor, tree_mdp: bool = False
    ) -> Tuple[NetworkOutput, NetworkOutput]:
        """g + f on the two latent children of the batched actions."""
        batch_size = obs.batch_size
        left_obs, right_obs = self.dynamics(obs, actions)
        left_actor_logits, left_pseudo_q_value, left_breward = self.prediction(left_obs)
        right_actor_logits, right_pseudo_q_value, right_breward = self.prediction(right_obs)

        if batch_size > 1:
            left_actor_logits, left_pseudo_q_value, left_breward = _pad_all(
                obs, left_actor_logits, left_pseudo_q_value, left_breward
            )
            right_actor_logits, right_pseudo_q_value, right_breward = _pad_all(
                obs, right_actor_logits, right_pseudo_q_value, right_breward
            )
        else:
            left_actor_logits = left_actor_logits.unsqueeze(0)
            left_pseudo_q_value = left_pseudo_q_value.unsqueeze(0)
            left_breward = left_breward.unsqueeze(0)
            right_actor_logits = right_actor_logits.unsqueeze(0)
            right_pseudo_q_value = right_pseudo_q_value.unsqueeze(0)
            right_breward = right_breward.unsqueeze(0)

        max_variables = left_pseudo_q_value.size(-2)
        mean_correction = max_variables / obs.nb_variables
        left_value = left_pseudo_q_value.mean(dim=-2) * mean_correction.unsqueeze(-1)
        right_value = right_pseudo_q_value.mean(dim=-2) * mean_correction.unsqueeze(-1)

        if not tree_mdp:
            left_value = self.value_loss.value_transform(left_value)
            right_value = self.value_loss.value_transform(right_value)

        left_breward = torch.mean(left_breward, dim=-2)
        right_breward = torch.mean(right_breward, dim=-2)

        if self.mcts:
            left_breward = F.softmax(left_breward, dim=-1)[..., 1].unsqueeze(-1)
            left_value = left_value * left_breward
            left_obs = left_obs.detach().cpu()
            left_value = left_value.detach().cpu().numpy()
            left_breward = left_breward.detach().cpu().numpy()
            left_actor_logits = left_actor_logits.squeeze(-1).cpu().numpy()

            right_breward = F.softmax(right_breward, dim=-1)[..., 1].unsqueeze(-1)
            right_value = right_value * right_breward
            right_obs = right_obs.detach().cpu()
            right_value = right_value.detach().cpu().numpy()
            right_breward = right_breward.detach().cpu().numpy()
            right_actor_logits = right_actor_logits.squeeze(-1).cpu().numpy()
        else:
            left_actor_logits = left_actor_logits.squeeze(-1)
            right_actor_logits = right_actor_logits.squeeze(-1)

        left_output = NetworkOutput(left_actor_logits, left_value, left_breward, left_obs, batch_size)
        right_output = NetworkOutput(right_actor_logits, right_value, right_breward, right_obs, batch_size)
        return left_output, right_output

    def project(self, hidden_state, with_grad: bool = True) -> torch.Tensor:
        """The SimSiam consistency branch: gradient flows through the
        projection head, never through the target."""
        proj = self.projection(hidden_state.variable_features)
        if with_grad:
            return pad_proj(self.projection_head(proj), hidden_state.nb_variables).view(
                hidden_state.batch_size, -1
            )
        return pad_proj(proj.detach(), hidden_state.nb_variables).view(hidden_state.batch_size, -1)

    # -- modes and persistence --------------------------------------------

    def mcts_mode(self) -> None:
        self.mcts = True

    def train_mode(self) -> None:
        self.mcts = False

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights) -> None:
        self.load_state_dict(weights)


class TreeHLGaussLoss(_BBMDPHLGauss):
    """The official value loss (same support as the bbmdp head) plus the
    tree-MDP branches: CE against the target's bin distribution, and the
    forward/backward transforms between scalars and bin distributions. The
    decode keeps the official's trailing singleton dim (``[B, 1]``) so the
    MCTS-mode ``value * breward`` broadcast stays per-sample."""

    def forward(self, value_logits: torch.Tensor, target_value: torch.Tensor, tree_mdp: bool = False):
        target_probs = self.inverse_value_transform(target_value).nan_to_num()
        if tree_mdp:
            return F.cross_entropy(value_logits, target_probs.squeeze(1), reduction="none")
        probs = self.inverse_value_transform(value_logits)
        return -(torch.log(probs + 1e-8) * target_probs).sum(dim=-1)

    def inverse_value_transform(self, scalar_value: torch.Tensor) -> torch.Tensor:
        return self.transform_to_probs(torch.log2(-scalar_value.clip(max=-1e-1)))

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        return super().transform_from_probs(probs).unsqueeze(-1)

    def value_transform(self, value_logits: torch.Tensor) -> torch.Tensor:
        return -torch.pow(2, self.transform_from_probs(value_logits.softmax(dim=-1)))

