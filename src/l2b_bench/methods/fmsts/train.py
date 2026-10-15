"""FMSTS 2020 — the training loop (Algorithm 1) and its replay buffer.

Ported from the paper (no official repository):

- **Algorithm 1** — iterate {draw an instance from the training pool -> solve
  it epsilon-greedy under the paper's DFS protocol -> store every decision's
  (s, a, Q) with its instance weight -> one loss update over a batch drawn
  from the buffer}. The paper draws from a fixed pool of 200 industrial
  instances; the pilot writes/reuses a pool on disk and redraws uniformly (a
  fresh model is read per draw).
- **Eq 5** — the MSE against the observed Q-values, weighted by 1/V(tree) so
  every instance weighs equally in the expectation.
- **Eq 6** — prioritized experience replay with the sampling probability
  proportional to the error normalised by its target, |Q - Q_hat|/Q (a plain
  proportional PER; the paper mentions no importance-sampling weights).
- **The PCA** — fitted with scikit-learn on the training pool's padded (A, b,
  c) vectors; its mean and components are part of the checkpoint. The
  component count, optimizer, batch size, and buffer size are not specified
  in the paper — pilot choices (Adam 1e-3, batches of 128, 100k buffer, 16
  components). The epsilon schedule is not given either — a linear anneal
  1.0 -> 0.05 over the iteration budget.
"""

import random
from pathlib import Path

import numpy as np
import torch

from ..bbmdp.replay import SumSegmentTree
from .collect import FMSTSExperience, FMSTSObservation, make_env, run_episode
from .model import MDANetwork

#: Pilot choices (none of these is specified in the paper).
N_PCA = 16
BATCH_SIZE = 128
BUFFER_SIZE = 100_000
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.05
#: A floor on the PER priorities so a perfectly predicted sample stays
#: reachable in principle (numerical only — Eq 6 is the raw |Q - Q_hat|/Q).
PRIOR_EPS = 1e-6


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_pca(
    instance_dir: Path,
    canonical: tuple[int, int],
    n_components: int = N_PCA,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the PCA (scikit-learn) over the pool's padded (A, b, c) vectors,
    read directly from the instance files — no solve at all: the vector is
    the original problem's data, so a plain file read is the whole visit.
    Returns (mean, components), the pieces the observation carries in the
    checkpoint."""
    from gyozas.instances import FileGenerator
    from sklearn.decomposition import PCA

    from .collect import instance_vector

    vectors = [
        np.asarray(instance_vector(model, canonical))
        for model in FileGenerator(
            instance_dir, pattern="*.mps", sampling_mode="remove", rng=seed
        )
    ]
    if not vectors:
        raise RuntimeError(f"no readable instances under {instance_dir}")
    n_components = min(n_components, len(vectors))
    pca = PCA(n_components=n_components, random_state=seed).fit(np.stack(vectors))
    print(
        f"[train] PCA fitted on {len(vectors)} instances "
        f"({n_components} components, {pca.explained_variance_ratio_.sum():.1%} variance)",
        flush=True,
    )
    return pca.mean_, pca.components_


class FMSTSReplay:
    """The proportional PER of Eq 6: p_j ~ |Q - Q_hat|/Q over the stored
    experiences (OpenAI-baselines' segment tree, as in the BBMDP port)."""

    def __init__(self, size: int = BUFFER_SIZE) -> None:
        self.buf = np.empty(size, dtype=object)
        self.max_size = size
        self.ptr = 0
        self.size = 0
        capacity = 1
        while capacity < size:
            capacity *= 2
        self.sum_tree = SumSegmentTree(capacity)

    def store(self, experience: FMSTSExperience, priority: float) -> None:
        self.buf[self.ptr] = experience
        self.sum_tree[self.ptr] = priority
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(
        self, batch_size: int, rng: random.Random
    ) -> tuple[list[int], list[FMSTSExperience]]:
        """One stratified proportional draw (the baselines convention)."""
        total = self.sum_tree.sum(0, self.size - 1)
        segment = total / batch_size
        indices = []
        for i in range(batch_size):
            upperbound = rng.uniform(segment * i, segment * (i + 1))
            indices.append(self.sum_tree.retrieve(upperbound))
        return indices, [self.buf[index] for index in indices]

    def update_priorities(self, indices: list[int], priorities: list[float]) -> None:
        for index, priority in zip(indices, priorities, strict=True):
            self.sum_tree[index] = priority

    def __len__(self) -> int:
        return self.size


def _q_at_actions(
    model: MDANetwork, experiences: list[FMSTSExperience], device: torch.device
) -> torch.Tensor:
    """Q_hat at each experience's taken action — one batched forward pass per
    episode (the pool is homogeneous, so the dynamic blocks stack)."""
    static = torch.as_tensor(
        np.stack([e.observation.static for e in experiences]),
        dtype=torch.float32,
        device=device,
    )
    dynamic = torch.as_tensor(
        np.stack([e.observation.dynamic for e in experiences]),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.as_tensor(
        [e.action for e in experiences], dtype=torch.long, device=device
    )
    with torch.no_grad():
        values = model(static, dynamic)
        return values.gather(1, actions.unsqueeze(1)).squeeze(1)


def _loss_update(
    model: MDANetwork,
    optimizer: torch.optim.Optimizer,
    buffer: FMSTSReplay,
    batch_size: int,
    sampling_rng: random.Random,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """One Eq 5 loss update over a PER batch (Eq 6 priorities), returning the
    loss and the new priorities to write back."""
    indices, experiences = buffer.sample(batch_size, sampling_rng)
    static = torch.as_tensor(
        np.stack([e.observation.static for e in experiences]),
        dtype=torch.float32,
        device=device,
    )
    dynamic = torch.as_tensor(
        np.stack([e.observation.dynamic for e in experiences]),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.as_tensor(
        [e.action for e in experiences], dtype=torch.long, device=device
    )
    targets = torch.as_tensor(
        [e.q_target for e in experiences], dtype=torch.float32, device=device
    )
    weights = torch.as_tensor(
        [e.instance_weight for e in experiences], dtype=torch.float32, device=device
    )
    values = model(static, dynamic)
    q_hat = values.gather(1, actions.unsqueeze(1)).squeeze(1)
    # Eq 5: the MSE against the observed Q, weighted by 1/V(tree).
    loss = ((q_hat - targets) ** 2 * weights).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Eq 6: the priorities are the errors normalised by their targets.
    with torch.no_grad():
        priorities = ((q_hat - targets).abs() / targets + PRIOR_EPS).cpu().numpy()
    buffer.update_priorities(indices, priorities.tolist())
    return float(loss.item()), priorities


def save_checkpoint(
    model: MDANetwork,
    mean: np.ndarray,
    components: np.ndarray,
    canonical: tuple[int, int],
    path: str | Path,
) -> None:
    """Save the policy checkpoint: the MDA state dict plus the fitted PCA
    (mean and components as tensors, so the file loads weights-only) and the
    canonical padding shape."""
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "pca_mean": torch.as_tensor(np.asarray(mean), dtype=torch.float64),
            "pca_components": torch.as_tensor(
                np.asarray(components), dtype=torch.float64
            ),
            "canonical": tuple(canonical),
        },
        path,
    )


def load_checkpoint(
    path: str | Path, device: torch.device | None = None
) -> tuple[MDANetwork, np.ndarray, np.ndarray, tuple[int, int]]:
    """Load a policy checkpoint: (model, mean, components, canonical)."""
    device = device or get_device()
    data = torch.load(path, map_location="cpu", weights_only=True)
    components = data["pca_components"].numpy()
    model = MDANetwork(n_pca=components.shape[0])
    model.load_state_dict(data["state_dict"])
    model.to(device).eval()
    return model, data["pca_mean"].numpy(), components, tuple(data["canonical"])


def train(
    instance_dir: Path,
    checkpoint: Path,
    iterations: int,
    canonical: tuple[int, int],
    time_limit: float = 60.0,
    seed: int = 42,
) -> int:
    """The Algorithm 1 loop: episodes over the pool (uniform redraws) until
    the iteration budget is spent, one gradient update per iteration. The
    rejected episodes (not solved within the limit, misaligned trees) consume
    their draw but store nothing. Returns the number of gradient updates."""
    from gyozas.instances import FileGenerator

    torch.manual_seed(seed)
    python_rng = random.Random(seed)
    sampling_rng = random.Random(seed + 1)

    mean, components = fit_pca(instance_dir, canonical, seed=seed)
    env = make_env(
        FileGenerator(instance_dir, pattern="*.mps", sampling_mode="replace", rng=seed),
        FMSTSObservation(mean, components, canonical),
        time_limit=time_limit,
        seed=seed,
    )
    device = get_device()
    model = MDANetwork(n_pca=components.shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    buffer = FMSTSReplay()
    print(f"[train] device: {device}", flush=True)

    n_updates = 0
    n_stored = 0
    for iteration in range(1, iterations + 1):
        epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * iteration / iterations

        def select_action(obs, candidates, epsilon=epsilon):
            if python_rng.random() < epsilon:
                action_idx = python_rng.randrange(len(candidates))
                return int(candidates[action_idx]), action_idx
            static = torch.as_tensor(
                obs.static, dtype=torch.float32, device=device
            ).unsqueeze(0)
            dynamic = torch.as_tensor(
                obs.dynamic, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                values = model(static, dynamic).squeeze(0)
            cand = torch.as_tensor(
                np.asarray(candidates), dtype=torch.long, device=device
            )
            action_idx = int(values[cand].argmin())  # index within the action set
            return int(candidates[action_idx]), action_idx

        experiences, n_decisions, status = run_episode(env, select_action)
        if experiences is None:
            print(f"[iter {iteration:4d}] rejected ({status})", flush=True)
            continue
        q_hats = _q_at_actions(model, experiences, device)
        for experience, q_hat in zip(experiences, q_hats.tolist(), strict=True):
            priority = (
                abs(experience.q_target - q_hat) / experience.q_target + PRIOR_EPS
            )
            buffer.store(experience, priority)
            n_stored += 1
        if len(buffer) >= BATCH_SIZE:
            loss, _priorities = _loss_update(
                model, optimizer, buffer, BATCH_SIZE, sampling_rng, device
            )
            n_updates += 1
        else:
            loss = float("nan")
        print(
            f"[iter {iteration:4d}] {status}  decisions={n_decisions}  "
            f"tree={1.0 / experiences[0].instance_weight:.0f}  stored={n_stored}  "
            f"buffer={len(buffer)}  updates={n_updates}  eps={epsilon:.3f}  loss={loss:.4f}",
            flush=True,
        )
    save_checkpoint(model, mean, components, canonical, checkpoint)
    print(f"[train] saved {checkpoint} after {n_updates} gradient updates", flush=True)
    return n_updates
