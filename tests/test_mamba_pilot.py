"""End-to-end tests of the Mamba-Branching 2026 port: collect → train → policy.

The collection instances are small set covers; the point of the tests is the
whole machinery — relpscost-expert trajectories kept per episode (the
decision order is the training signal), the contrastive pretraining of the
TreeGateNet, the Mamba selective SSM over the sequentialized paths, and the
streaming deterministic reader — not that the model learns anything.

The vendored :class:`~l2b_bench.methods.mamba.model.Mamba` block is checked
against an independent per-step implementation of the same recurrence (the
reference's fused kernels cannot run here — no CUDA toolchain to build
``mamba_ssm`` — so the scan is the only hand-written math in the port).
"""

import numpy as np
import pytest
import torch
from gyozas.augmentation import ZarpellonAugmentation
from gyozas.dynamics.branching import BranchingDynamics, ExtraBranchingActions
from gyozas.environment import Environment
from gyozas.informations.solve_status import SolveStatus
from gyozas.instances import SetCoverGenerator
from gyozas.observations import BranchingStateObservation
from gyozas.seeds import SeedTriplet

from l2b_bench.methods.mamba import (
    EVAL_MAX_SEQ_LEN,
    GroupedMamba,
    Mamba,
    MambaSample,
    TreeGateNet,
    chunk_trajectories,
    clip_batch,
    clip_loss,
    collect_episode,
    load_checkpoint,
    mamba_method,
    sequence_batch,
    sequence_loss,
    train,
)


def _sample(n_candidates: int, rng: np.random.Generator, n_features: int = 25) -> MambaSample:
    """A synthetic decision: random rows, the expert's choice among them."""
    action_set = rng.permutation(100)[:n_candidates]
    choice = int(rng.integers(n_candidates))
    return MambaSample(
        candidate_features=rng.standard_normal((n_candidates, n_features)).astype(np.float32),
        action_set=action_set,
        action=int(action_set[choice]),
        node_features=rng.standard_normal(8).astype(np.float32),
        mip_features=rng.standard_normal(53).astype(np.float32),
    )


def _trajectories(n_episodes: int = 3, length: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [[_sample(int(rng.integers(2, 6)), rng) for _ in range(length)] for _ in range(n_episodes)]


@pytest.fixture(scope="module")
def collected():
    """Four relpscost-expert trajectories on small set covers (s=0, k=2).

    A small k so the instances still leave expert decisions after the
    random-explorer prefix. Under default SCIP the first draws of this
    family solve at the root (0 nodes); four episodes guarantee at least
    one branching trajectory (the treegate port's fixture size).
    """
    generator = SetCoverGenerator(n_rows=150, n_cols=200, density=0.10, rng=np.random.default_rng(7))
    env = Environment(
        instance_generator=generator,
        observation_function=BranchingStateObservation(),
        information_function=SolveStatus(),
        scip_params={"limits/time": 30.0, "display/verblevel": 0},
        dynamics=BranchingDynamics(
            augmentation=ZarpellonAugmentation(seed=0, k=2),
            with_extra_actions=[ExtraBranchingActions.SKIP],
        ),
    )
    trajectories = []
    try:
        for seed in range(4):
            triplet = SeedTriplet.from_base_seed(seed)
            env.seed(triplet.solver)
            triplet.seed_python()
            samples, n_nodes, status = collect_episode(env)
            trajectories.append((samples, n_nodes, status))
    finally:
        env.close()
    return trajectories


class TestVendoredMamba:
    def test_batched_scan_matches_the_per_step_recurrence(self):
        """Two independent implementations of the reference step() math."""
        torch.manual_seed(0)
        mamba = Mamba(d_model=8, d_state=16, d_conv=4, expand=2).eval()
        x = torch.randn(3, 11, 8)

        with torch.no_grad():
            batched = mamba(x)

            # The naive path: recompute every projection by hand and step
            # through the sequence one token at a time.
            xz = mamba.in_proj(x)
            u, z = xz.chunk(2, dim=-1)
            u_pad = torch.nn.functional.pad(u.transpose(1, 2), (mamba.d_conv - 1, 0))
            conv = torch.nn.functional.conv1d(
                u_pad,
                mamba.conv1d.weight,
                mamba.conv1d.bias,
                groups=mamba.d_inner,
            ).transpose(1, 2)
            u = torch.nn.functional.silu(conv)
            dt_raw, B, C = torch.split(
                mamba.x_proj(u), [mamba.dt_rank, mamba.d_state, mamba.d_state], dim=-1
            )
            A = -torch.exp(mamba.A_log)
            h = torch.zeros(3, mamba.d_inner, mamba.d_state)
            outputs = []
            for t in range(x.size(1)):
                # The reference's step(): F.linear(dt, weight) then
                # softplus(dt + dt_bias) — the bias applied exactly once.
                delta = torch.nn.functional.softplus(
                    torch.nn.functional.linear(dt_raw[:, t], mamba.dt_proj.weight, mamba.dt_proj.bias)
                )
                h = h * torch.exp(delta.unsqueeze(-1) * A) + (
                    delta.unsqueeze(-1) * B[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
                )
                y = torch.einsum("bdn,bn->bd", h, C[:, t]) + mamba.D * u[:, t]
                outputs.append(y)
            y = torch.stack(outputs, dim=1) * torch.nn.functional.silu(z)
            naive = mamba.out_proj(y)

        assert batched.shape == naive.shape
        assert torch.allclose(batched, naive, atol=1e-5)

    def test_the_convolution_is_causal(self):
        """A later token cannot change an earlier output (the reference's
        causal conv + one-directional scan)."""
        torch.manual_seed(1)
        mamba = Mamba(d_model=8, d_state=16).eval()
        x = torch.randn(1, 9, 8)
        with torch.no_grad():
            before = mamba(x)
            changed = x.clone()
            changed[0, 5:] += 10.0
            after = mamba(changed)
        assert torch.allclose(before[0, :5], after[0, :5], atol=1e-6)
        assert not torch.allclose(before[0, 5:], after[0, 5:])

    def test_grouped_mamba_adds_position_embeddings(self):
        """x + position_embed(position_ids) — the position table is learnable."""
        torch.manual_seed(2)
        grouped = GroupedMamba(d_model=8, max_seq_length=20).eval()
        x = torch.randn(2, 7, 8)
        position_ids = torch.arange(7).unsqueeze(0).expand(2, -1)
        with torch.no_grad():
            direct = grouped.mamba(x + grouped.position_embed(position_ids))
            through = grouped(x, position_ids)
        assert torch.allclose(direct, through)


class TestTreeGateNet:
    def test_embeds_one_node_and_a_sequence(self):
        """The 3D path (one node) and the 4D path (a batch of sequences)."""
        torch.manual_seed(3)
        net = TreeGateNet().eval()
        state = torch.randn(4, 61)
        with torch.no_grad():
            embedded, probs = net(torch.randn(4, 5, 25), state)
            assert embedded.shape == (4, 5, 8)
            assert probs.shape == (4, 5)
            embedded_seq, probs_seq = net(torch.randn(2, 4, 5, 25), torch.randn(2, 4, 61))
            assert embedded_seq.shape == (2, 4, 5, 8)
            assert probs_seq.shape == (2, 4, 5)
        assert torch.allclose(probs, embedded.mean(dim=-1))

    def test_hparams_rebuild_the_net(self):
        net = TreeGateNet()
        rebuilt = TreeGateNet(**net.hparams)
        assert rebuilt.hparams == net.hparams


class TestTrajectories:
    def test_collection_keeps_decision_order_and_forced_filter(self, collected):
        """Every trajectory is the episode's decisions in solve order, each
        with more than one candidate (the official's forced-decision filter)."""
        n_samples = 0
        for samples, n_nodes, status in collected:
            assert status == "optimal"
            assert n_nodes >= len(samples)
            for sample in samples:
                assert len(sample.action_set) > 1
                assert sample.action in sample.action_set
                assert sample.candidate_features.shape == (len(sample.action_set), 25)
            n_samples += len(samples)
        assert n_samples > 0  # the family must still branch

    def test_chunking_drops_short_and_cuts_long(self):
        rng = np.random.default_rng(4)
        long_trajectory = [_sample(3, rng) for _ in range(205)]
        chunks = chunk_trajectories(
            [[_sample(3, rng)], [], long_trajectory], max_seq_length=100
        )
        assert [len(chunk) for chunk in chunks] == [100, 100, 5]
        # A single trailing decision carries no order to learn.
        rng = np.random.default_rng(5)
        chunks = chunk_trajectories([[_sample(3, rng)] * 101], max_seq_length=100)
        assert len(chunks) == 1 and len(chunks[0]) == 100


class TestContrastiveStage:
    def test_loss_matches_its_formula(self):
        """2 - cos(pre, strong) + mean cos(pre, weak), weak = every non-chosen
        row including pads — recomputed independently from the net's output."""
        torch.manual_seed(6)
        net = TreeGateNet().eval()
        rng = np.random.default_rng(6)
        samples = [_sample(4, rng), _sample(7, rng)]
        batch = clip_batch(samples)
        with torch.no_grad():
            loss = clip_loss(net, batch)
            embedded, _ = net(batch["cands"], batch["states"])
            pre = embedded.max(dim=1).values
            strong = embedded[torch.arange(2), batch["targets"]]
            expected = 2 - torch.nn.functional.cosine_similarity(pre, strong).mean()
            for i in range(2):
                weak = torch.cat([embedded[i, : batch["targets"][i]], embedded[i, batch["targets"][i] + 1 :]])
                expected = expected + torch.nn.functional.cosine_similarity(
                    pre[i], weak, dim=-1
                ).mean() / 2
        assert torch.isfinite(loss)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_training_pulls_the_action_towards_the_pool(self):
        """A few epochs on one repeated node push the loss down."""
        torch.manual_seed(7)
        rng = np.random.default_rng(7)
        samples = [_sample(5, rng) for _ in range(8)]
        net = TreeGateNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        losses = []
        for _ in range(20):
            loss = clip_loss(net, clip_batch(samples))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]


class TestSequenceStage:
    def test_batch_pads_and_indexes(self):
        rng = np.random.default_rng(8)
        sequences = [[_sample(3, rng), _sample(6, rng)], [_sample(4, rng)]]
        batch = sequence_batch(sequences)
        assert batch["cands"].shape == (2, 2, 6, 25)
        assert batch["states"].shape == (2, 2, 61)
        assert batch["valid"].tolist() == [[True, True], [True, False]]
        for (sequence, actions, row) in zip(sequences, batch["actions"], batch["valid"], strict=True):
            for i, sample in enumerate(sequence):
                assert row[i]  # only the padding rows are invalid
                assert sample.action_set[actions[i]] == sample.action

    def test_loss_is_finite_and_trains_both_nets(self):
        """Backprop reaches the Mamba AND the (still training) TreeGateNet."""
        torch.manual_seed(9)
        rng = np.random.default_rng(9)
        net = TreeGateNet()
        mamba = GroupedMamba(max_seq_length=20)
        batch = sequence_batch([[_sample(4, rng) for _ in range(5)]])
        loss, acc1 = sequence_loss(net, mamba, batch)
        assert torch.isfinite(loss)
        assert 0.0 <= acc1 <= 1.0
        loss.backward()
        mamba_grads = [p.grad for p in mamba.parameters() if p.grad is not None]
        net_grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert mamba_grads and net_grads

    def test_logits_read_the_candidate_positions(self):
        """The score of a candidate is the mean of the Mamba output's 8 dims
        at its token position — the interleave, the position ids and the CE
        recomputed independently from the same nets."""
        torch.manual_seed(10)
        rng = np.random.default_rng(10)
        net = TreeGateNet().eval()
        mamba = GroupedMamba(max_seq_length=20).eval()
        samples = [_sample(4, rng), _sample(6, rng)]
        batch = sequence_batch([samples])
        with torch.no_grad():
            loss, _acc = sequence_loss(net, mamba, batch)
            embedded, _ = net(batch["cands"], batch["states"])  # (1, L, C, 8)
            n_cands = batch["cands"].shape[2]  # the padded width, pads included
            tokens, position_ids = [], []
            for i in range(len(samples)):
                tokens.append(embedded[0, i])  # the C candidate rows (pads too)
                tokens.append(embedded[0, i, batch["actions"][0, i]].unsqueeze(0))
                position_ids += [i] * (n_cands + 1)
            path = torch.cat(tokens).unsqueeze(0)
            out = mamba(path, torch.tensor([position_ids]))
            logits, position = [], 0
            for _i in range(len(samples)):
                logits.append(out[0, position : position + n_cands].mean(dim=-1))
                position += n_cands + 1
            expected = torch.nn.functional.cross_entropy(
                torch.stack(logits), batch["actions"][0]
            )
        assert torch.isfinite(loss)
        assert torch.allclose(loss, expected, atol=1e-5)


class TestTraining:
    def test_two_stage_training_saves_a_loadable_checkpoint(self, tmp_path):
        trajectories = _trajectories()
        checkpoint = tmp_path / "mamba.pt"
        logs = []
        train(trajectories, checkpoint, epochs=1, clip_epochs=1, device=torch.device("cpu"), log=logs.append)
        assert checkpoint.exists()
        assert any("contrastive" in line for line in logs)
        assert any("sequence" in line for line in logs)

        net, mamba = load_checkpoint(checkpoint, device=torch.device("cpu"))
        assert isinstance(net, TreeGateNet)
        assert mamba.position_embed.num_embeddings == 100
        # The loaded nets accept the training shapes.
        with torch.no_grad():
            embedded, _ = net(torch.randn(1, 4, 25), torch.randn(1, 61))
            assert embedded.shape == (1, 4, 8)


class TestPolicy:
    def test_streaming_policy_is_runner_ready(self, tmp_path):
        """select_action returns an action from the set, the path alternates
        state/action blocks, and the window drops the oldest blocks."""
        trajectories = _trajectories()
        checkpoint = tmp_path / "mamba.pt"
        train(trajectories, checkpoint, epochs=1, clip_epochs=1, device=torch.device("cpu"))

        method = mamba_method(checkpoint)
        assert method.name == "mamba"
        assert method.fidelity == "mamba"

        policy = method.policy()
        rng = np.random.default_rng(11)
        obs = type(
            "Obs",
            (),
            {
                "candidate_features": rng.standard_normal((5, 25)),
                "node_features": rng.standard_normal(8),
                "mip_features": rng.standard_normal(53),
            },
        )()
        action_set = np.array([7, 11, 13, 17, 19])
        for _ in range(3):
            action = policy.select_action(obs, action_set)
            assert action in action_set
        # Each decision appends a state block then an action block.
        assert len(policy.blocks) == 6
        # ...and the window is bounded by the official's 49 blocks.
        for _ in range(40):
            policy.select_action(obs, action_set)
        assert len(policy.blocks) == EVAL_MAX_SEQ_LEN
