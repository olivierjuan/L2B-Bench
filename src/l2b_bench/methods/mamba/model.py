"""Mamba-Branching 2026 — the contrastive TreeGateNet and the Mamba path model.

Port of ``network.py`` from the official repository
(github.com/doctor-watson626/Mamba-Branching):

- ``TreeGateNet`` embeds a node's branching candidates (25 solver stats) to
  8-dim rows through a gated pyramid (depth 3, sigmoid gates from the 61-dim
  tree state, infimum 8) — the same gated head TreeGate uses, with the
  candidate axis kept (one 8-dim row per candidate, plus its mean as a
  shortcut score);
- ``GroupedMamba`` reads the *sequentialized path* — each decision contributes
  its candidate rows and the chosen action's row — through a selective SSM
  with learned block-position embeddings.

The official ``GroupedMamba`` wraps ``mamba_ssm.Mamba`` (the CUDA package).
This cluster has no CUDA toolchain to build it, so the block is vendored below
as :class:`Mamba` — a faithful pure-PyTorch transcription of
``mamba_ssm/modules/mamba_simple.py`` (state-spaces/mamba, Mamba 1) at the
paper's hyperparameters (d_model 8, d_state 64, d_conv 4, expand 2): same
projections, same depthwise causal convolution, same dt/A/D initializations,
and the sequential selective-scan recurrence of the reference ``step()``
method. Only the fused kernels are replaced by the explicit loop.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

N_CANDIDATE_FEATURES = 25
N_NODE_FEATURES = 8
N_MIP_FEATURES = 53

# The official main.py / evaluate_all.py constants.
EMBED_DIM = 8  # TreeGateNet(infimum=8)
HIDDEN_DIM = 64  # TreeGateNet(hidden_size=64, depth=3, dim_reduce_factor=2)
D_STATE = 64
D_CONV = 4
EXPAND = 2


class Mamba(nn.Module):
    """A selective SSM block (Mamba 1), pure PyTorch.

    Vendored from ``mamba_ssm/modules/mamba_simple.py`` (state-spaces/mamba):
    ``in_proj`` (x and gate), depthwise causal conv + SiLU, ``x_proj`` /
    ``dt_proj`` with the dt initializations, S4D-real ``A``, skip ``D``, and
    the selective-scan recurrence ``h_t = h_{t-1} * exp(dt*A) + dt*B*u_t``,
    ``y_t = C_t . h_t + D*u_t`` — the reference implementation's own
    ``step()`` math, batched over the sequence with an explicit loop (the only
    deviation: no fused CUDA kernel).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # dt weight init: preserve variance at initialization.
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # dt bias init: softplus(bias) in [dt_min, dt_max] (inverse softplus).
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization.
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        # D "skip" parameter.
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """(B, L, d_model) -> (B, L, d_model)."""
        batch, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        u, z = xz.chunk(2, dim=-1)  # (B, L, d_inner) each

        # Causal depthwise convolution over the path, then SiLU.
        u = self.act(self.conv1d(u.transpose(1, 2))[..., :seqlen].transpose(1, 2))

        x_dbl = self.x_proj(u)  # (B, L, dt_rank + 2*d_state)
        dt_raw, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # dt_proj carries its bias (the dt initialization) — softplus(W dt + b),
        # the reference's delta_bias + delta_softplus.
        delta = F.softplus(self.dt_proj(dt_raw))  # (B, L, d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # The selective scan (the reference step()'s recurrence, batched).
        h = u.new_zeros(batch, self.d_inner, self.d_state)
        outputs = []
        for t in range(seqlen):
            dA = torch.exp(delta[:, t].unsqueeze(-1) * A)  # (B, d_inner, d_state)
            dBu = (
                delta[:, t].unsqueeze(-1)
                * B[:, t].unsqueeze(1)
                * u[:, t].unsqueeze(-1)
            )  # (B, d_inner, d_state)
            h = dA * h + dBu
            y = torch.einsum("bdn,bn->bd", h, C[:, t]) + self.D * u[:, t]  # (B, d_inner)
            outputs.append(y)
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)

        y = y * self.act(z)
        return self.out_proj(y)


class GroupedMamba(nn.Module):
    """The selective SSM over the path, with learned block-position embeddings."""

    def __init__(self, d_model: int = EMBED_DIM, max_seq_length: int = 100) -> None:
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model, d_state=D_STATE, d_conv=D_CONV, expand=EXPAND
        )
        self.position_embed = nn.Embedding(max_seq_length, d_model)

    def forward(self, x: Tensor, position_ids: Tensor) -> Tensor:
        """x (B, L, d_model), position_ids (B, L) -> (B, L, d_model)."""
        return self.mamba(x + self.position_embed(position_ids))


class TreeGateBranchingNet(nn.Module):
    """TreeGate's gated pyramid, in Mamba-Branching's two-input-width variant.

    A stack of layers shrinking ``hidden_size`` by ``dim_reduce_factor``
    toward ``infimum`` units (64 -> 32 -> 16 -> 8 here); a sigmoid
    ``GatingNet`` maps the tree state to one gate per unit of every layer,
    and each layer's input is multiplied by its slice of gates first. The
    output is one ``infimum``-dim row per candidate (3D input: one node's
    candidates; 4D input: a batch of sequences of nodes).
    """

    def __init__(
        self,
        branch_size: int,
        tree_state_size: int,
        dim_reduce_factor: int = 2,
        infimum: int = EMBED_DIM,
        norm: str = "none",
        depth: int = 3,
        hidden_size: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        norm_layer = {"batch": nn.BatchNorm1d, "instance": nn.InstanceNorm1d, "layer": nn.LayerNorm, "none": nn.Identity}[norm]
        self.norm = norm
        self.branch_size = branch_size
        self.tree_state_size = tree_state_size
        self.dim_reduce_factor = dim_reduce_factor
        self.infimum = infimum
        self.depth = depth
        self.hidden_size = hidden_size

        # 64 -> 32 -> 16 -> 8: the layers shrink hidden_size toward infimum.
        self.n_layers = 0
        unit_count = infimum
        while unit_count < branch_size:
            unit_count *= dim_reduce_factor
            self.n_layers += 1

        self.n_units: dict[int, int] = {}
        self.branching_net = nn.ModuleList()
        input_dim = hidden_size
        for i in range(self.n_layers):
            output_dim = int(input_dim / dim_reduce_factor)
            self.n_units[i] = input_dim
            if i < self.n_layers - 1:
                layer = [nn.Linear(input_dim, output_dim), norm_layer(output_dim), nn.ReLU(True)]
            else:
                layer = [nn.Linear(input_dim, infimum)]
            input_dim = output_dim
            self.branching_net.append(nn.Sequential(*layer))

        # One gate per unit of every layer, from the tree state.
        self.n_attentional_units = sum(self.n_units.values())
        gating: list[nn.Module] = [nn.Linear(tree_state_size, hidden_size), nn.ReLU(True)]
        for _ in range(depth - 2):
            gating += [nn.Linear(hidden_size, hidden_size), nn.ReLU(True)]
        gating += [nn.Linear(hidden_size, self.n_attentional_units), nn.Sigmoid()]
        self.gating_net = nn.Sequential(*gating)

    def forward(self, cands_state_mat: Tensor, tree_state: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (embedded (..., cands, infimum), probs (..., cands)).

        ``probs`` is the mean over the embedding dims — a shortcut score per
        candidate, NOT a distribution.
        """
        attn_weights = self.gating_net(tree_state)
        start = 0
        for index, layer in enumerate(self.branching_net):
            end = start + self.n_units[index]
            # 3D input (one node): gates (B, units) -> (B, 1, units);
            # 4D input (a sequence): gates (B, L, units) -> (B, L, 1, units).
            if cands_state_mat.dim() == 3:
                attn_slice = attn_weights[:, start:end].unsqueeze(-2)
            else:
                attn_slice = attn_weights[:, :, start:end].unsqueeze(-2)
            cands_state_mat = cands_state_mat * attn_slice
            cands_state_mat = layer(cands_state_mat)
            start = end
        probs = cands_state_mat.mean(dim=-1)
        return cands_state_mat, probs


class TreeGateNet(nn.Module):
    """The candidate embedder: 25 solver stats -> one 8-dim row per candidate."""

    def __init__(
        self,
        var_dim: int = N_CANDIDATE_FEATURES,
        node_dim: int = N_NODE_FEATURES,
        mip_dim: int = N_MIP_FEATURES,
        hidden_size: int = HIDDEN_DIM,
        depth: int = 3,
        dim_reduce_factor: int = 2,
        infimum: int = EMBED_DIM,
        norm: str = "none",
    ) -> None:
        super().__init__()
        # Recorded in the checkpoint so load_checkpoint can rebuild the net.
        self.hparams = {
            "var_dim": var_dim,
            "node_dim": node_dim,
            "mip_dim": mip_dim,
            "hidden_size": hidden_size,
            "depth": depth,
            "dim_reduce_factor": dim_reduce_factor,
            "infimum": infimum,
            "norm": norm,
        }
        self.candidate_embedding_net = nn.Sequential(nn.Linear(var_dim, hidden_size))
        self.tree_gate_branching_net = TreeGateBranchingNet(
            branch_size=hidden_size,
            tree_state_size=node_dim + mip_dim,
            dim_reduce_factor=dim_reduce_factor,
            infimum=infimum,
            norm=norm,
            depth=depth,
            hidden_size=hidden_size,
        )
        # Xavier initialization (the official constructor's loop).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self, cands_state_mat: Tensor, state: Tensor
    ) -> tuple[Tensor, Tensor]:
        """cands (..., cands, 25), state (..., 61) -> (embedded (..., cands, 8), probs (..., cands))."""
        cands_state_mat = self.candidate_embedding_net(cands_state_mat)
        return self.tree_gate_branching_net(cands_state_mat, state)
