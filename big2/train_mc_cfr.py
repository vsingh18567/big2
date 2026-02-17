import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from big2.nn import combo_to_action_vector, make_policy
from big2.simulator.cards import PASS, Combo, card_name
from big2.simulator.env import Big2Env
from big2.train_helpers import evaluate_policy


@dataclass
class CFRConfig:
    """Configuration for Deep Monte-Carlo CFR training."""

    n_players: int = 2
    cards_per_player: int = 5
    iterations: int = 200
    traversals_per_player: int = 8
    regret_epochs: int = 2
    strategy_epochs: int = 2
    regret_updates_per_player: int = 128
    strategy_updates_per_iteration: int = 128
    regret_lr: float = 5e-4
    strategy_lr: float = 5e-4
    batch_size: int = 64
    regret_buffer_size: int = 200_000
    strategy_buffer_size: int = 200_000
    device: str = "cpu"
    policy_arch: str = "setpool"
    seed: int = 42
    eval_interval: int = 1
    eval_games: int = 200
    reinit_regret_nets_each_iteration: bool = False
    reinit_avg_policy_each_iteration: bool = False


@dataclass
class RegretSample:
    """A single info-set regret target for one player."""

    state: np.ndarray
    candidates: list[Combo]
    regrets: np.ndarray
    weight: float


@dataclass
class StrategySample:
    """A single info-set strategy target (for average policy)."""

    state: np.ndarray
    candidates: list[Combo]
    strategy: np.ndarray
    weight: float


class ReservoirBuffer:
    """Reservoir sampling buffer for streaming training samples."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: list[object] = []
        self.n_seen = 0

    def add(self, item: object) -> None:
        self.n_seen += 1
        if len(self.data) < self.capacity:
            self.data.append(item)
            return
        idx = random.randrange(self.n_seen)
        if idx < self.capacity:
            self.data[idx] = item

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int) -> list[object]:
        if not self.data:
            return []
        k = min(batch_size, len(self.data))
        return random.sample(self.data, k)


def _terminal_utility(env: Big2Env, player: int) -> float:
    """Terminal payoff from the perspective of `player`."""
    if env.winner == player:
        return float(sum(len(env.hands[p]) for p in range(env.n_players) if p != player))
    return -float(len(env.hands[player]))


def _regret_matching(regrets: torch.Tensor) -> torch.Tensor:
    """Convert raw regrets into a probability distribution."""
    pos = torch.clamp(regrets, min=0.0)
    total = pos.sum()
    if total.item() <= 1e-8:
        return torch.ones_like(regrets) / regrets.numel()
    return pos / total


def _strategy_from_regret_net(net: nn.Module, state: np.ndarray, candidates: list[Combo], device: str) -> torch.Tensor:
    """Compute a regret-matching strategy from the regret network."""
    if not candidates:
        candidates = [Combo(PASS, [], ())]
    action_feats = [[combo_to_action_vector(c, card_count=net.card_universe_size) for c in candidates]]
    with torch.no_grad():
        state_tensor = torch.from_numpy(state[np.newaxis, :]).long().to(device)
        logits_list, _values = net(state_tensor, action_feats)
        regrets = logits_list[0].to(device)
    return _regret_matching(regrets)


def _sample_action(candidates: list[Combo], probs: torch.Tensor) -> Combo:
    """Sample a candidate action using a probability vector."""
    dist = torch.distributions.Categorical(probs=probs)
    idx = int(dist.sample().item())
    return candidates[idx]


def _traverse_game(
    env: Big2Env,
    traversing_player: int,
    regret_nets: list[nn.Module],
    regret_buffer: ReservoirBuffer,
    strategy_buffer: ReservoirBuffer,
    iteration: int,
    device: str,
) -> float:
    """External-sampling CFR traversal for one player."""
    if env.done:
        return _terminal_utility(env, traversing_player)

    p = env.current_player
    candidates = env.legal_candidates(p)
    if not candidates:
        candidates = [Combo(PASS, [], ())]
    state = env._obs(p)
    probs = _strategy_from_regret_net(regret_nets[p], state, candidates, device)

    if p == traversing_player:
        # Traverser node: recurse over all actions to estimate counterfactual values.
        action_values = []
        for action in candidates:
            env_copy = copy.deepcopy(env)
            env_copy.step(action)
            value = _traverse_game(
                env_copy,
                traversing_player=traversing_player,
                regret_nets=regret_nets,
                regret_buffer=regret_buffer,
                strategy_buffer=strategy_buffer,
                iteration=iteration,
                device=device,
            )
            action_values.append(value)

        action_values_t = torch.tensor(action_values, device=device, dtype=torch.float32)
        node_value = torch.dot(probs.to(device), action_values_t)
        regrets = (action_values_t - node_value).cpu().numpy().astype(np.float32)
        regret_buffer.add(
            RegretSample(
                state=state.copy(),
                candidates=candidates,
                regrets=regrets,
                weight=float(iteration),
            )
        )
        return float(node_value.item())

    # Opponent node: add to average-strategy memory and sample one action (external sampling).
    strategy_buffer.add(
        StrategySample(
            state=state.copy(),
            candidates=candidates,
            strategy=probs.cpu().numpy(),
            weight=float(iteration),
        )
    )
    action = _sample_action(candidates, probs)
    env.step(action)
    return _traverse_game(
        env,
        traversing_player=traversing_player,
        regret_nets=regret_nets,
        regret_buffer=regret_buffer,
        strategy_buffer=strategy_buffer,
        iteration=iteration,
        device=device,
    )


def _train_regret_net(
    net: nn.Module,
    buffer: ReservoirBuffer,
    optimizer: optim.Optimizer,
    batch_size: int,
    epochs: int,
    updates_per_epoch: int,
    device: str,
) -> float:
    """Train regret network with LCFR-weighted MSE on sampled regrets."""
    if len(buffer) == 0:
        return 0.0
    net.train()
    total_loss = 0.0
    steps = max(1, updates_per_epoch)
    for _ in range(epochs):
        for _ in range(steps):
            batch = buffer.sample(batch_size)
            if not batch:
                continue
            states = [sample.state for sample in batch]
            candidates_batch = [sample.candidates for sample in batch]
            target_regrets = [sample.regrets for sample in batch]
            weights = torch.tensor([sample.weight for sample in batch], dtype=torch.float32, device=device)

            state_tensor = torch.from_numpy(np.array(states)).long().to(device)
            action_feats = [
                [combo_to_action_vector(c, card_count=net.card_universe_size) for c in cands]
                for cands in candidates_batch
            ]
            logits_list, _values = net(state_tensor, action_feats)

            losses = []
            for logits, target in zip(logits_list, target_regrets, strict=True):
                tgt = torch.tensor(target, dtype=torch.float32, device=device)
                losses.append(F.mse_loss(logits, tgt))
            losses_t = torch.stack(losses)
            loss = (losses_t * weights).sum() / weights.sum().clamp_min(1.0)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimizer.step()
            total_loss += loss.item()

    return total_loss / max(1, epochs * steps)


def _train_strategy_net(
    net: nn.Module,
    buffer: ReservoirBuffer,
    optimizer: optim.Optimizer,
    batch_size: int,
    epochs: int,
    updates_per_epoch: int,
    device: str,
) -> float:
    """Train average policy network to match stored strategies."""
    if len(buffer) == 0:
        return 0.0
    net.train()
    total_loss = 0.0
    steps = max(1, updates_per_epoch)
    for _ in range(epochs):
        for _ in range(steps):
            batch = buffer.sample(batch_size)
            if not batch:
                continue
            states = [sample.state for sample in batch]
            candidates_batch = [sample.candidates for sample in batch]
            target_strats = [sample.strategy for sample in batch]
            weights = torch.tensor([sample.weight for sample in batch], dtype=torch.float32, device=device)

            state_tensor = torch.from_numpy(np.array(states)).long().to(device)
            action_feats = [
                [combo_to_action_vector(c, card_count=net.card_universe_size) for c in cands]
                for cands in candidates_batch
            ]
            logits_list, _values = net(state_tensor, action_feats)

            losses = []
            for logits, target in zip(logits_list, target_strats, strict=True):
                tgt = torch.tensor(target, dtype=torch.float32, device=device)
                logp = F.log_softmax(logits, dim=0)
                losses.append(-(tgt * logp).sum())
            losses_t = torch.stack(losses)
            loss = (losses_t * weights).sum() / weights.sum().clamp_min(1.0)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()

    return total_loss / max(1, epochs * steps)


def train_mc_cfr(config: CFRConfig) -> tuple[nn.Module, list[float], list[float], list[int], list[float]]:
    """Train Deep Monte-Carlo CFR for Big 2."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    def _new_policy() -> nn.Module:
        return make_policy(
            config.policy_arch,
            n_players=config.n_players,
            cards_per_player=config.cards_per_player,
            device=config.device,
        ).to(config.device)

    regret_nets = [_new_policy() for _ in range(config.n_players)]
    avg_policy = _new_policy()

    regret_opts = [optim.Adam(net.parameters(), lr=config.regret_lr) for net in regret_nets]
    avg_opt = optim.Adam(avg_policy.parameters(), lr=config.strategy_lr)

    regret_buffers = [ReservoirBuffer(config.regret_buffer_size) for _ in range(config.n_players)]
    strategy_buffer = ReservoirBuffer(config.strategy_buffer_size)

    regret_loss_history = []
    strategy_loss_history = []
    eval_episodes = []
    win_rates = []

    for iteration in tqdm(range(1, config.iterations + 1), desc="CFR iterations"):
        if config.reinit_regret_nets_each_iteration:
            regret_nets = [_new_policy() for _ in range(config.n_players)]
            regret_opts = [optim.Adam(net.parameters(), lr=config.regret_lr) for net in regret_nets]
        if config.reinit_avg_policy_each_iteration:
            avg_policy = _new_policy()
            avg_opt = optim.Adam(avg_policy.parameters(), lr=config.strategy_lr)

        # Collect traversals per player
        for player in range(config.n_players):
            for _ in tqdm(
                range(config.traversals_per_player),
                desc=f"Traversals p{player}",
                leave=False,
            ):
                env = Big2Env(config.n_players, cards_per_player=config.cards_per_player)
                env.reset()
                _traverse_game(
                    env,
                    traversing_player=player,
                    regret_nets=regret_nets,
                    regret_buffer=regret_buffers[player],
                    strategy_buffer=strategy_buffer,
                    iteration=iteration,
                    device=config.device,
                )
        if iteration % max(1, config.eval_interval // 2) == 0:
            buffer_sizes = [len(buf) for buf in regret_buffers]
            print(f"[Iter {iteration}] Regret buffers: {buffer_sizes} | " f"Strategy buffer: {len(strategy_buffer)}")

        # Train each regret network on its buffer
        regret_losses = []
        for player in range(config.n_players):
            loss = _train_regret_net(
                regret_nets[player],
                regret_buffers[player],
                regret_opts[player],
                config.batch_size,
                config.regret_epochs,
                config.regret_updates_per_player,
                config.device,
            )
            regret_losses.append(loss)

        # Train average strategy network
        strat_loss = _train_strategy_net(
            avg_policy,
            strategy_buffer,
            avg_opt,
            config.batch_size,
            config.strategy_epochs,
            config.strategy_updates_per_iteration,
            config.device,
        )

        regret_loss_history.append(float(np.mean(regret_losses)))
        strategy_loss_history.append(strat_loss)
        print(
            f"[Iter {iteration}] Training losses -> regret: {regret_loss_history[-1]:.4f}, "
            f"strategy: {strategy_loss_history[-1]:.4f}"
        )

        if iteration % config.eval_interval == 0:
            avg_policy.eval()
            metrics = evaluate_policy(
                avg_policy,
                config.n_players,
                cards_per_player=config.cards_per_player,
                num_games=config.eval_games,
                device=config.device,
            )
            avg_policy.train()
            eval_episodes.append(iteration)
            win_rates.append(metrics.win_rate_vs_greedy)

            print(f"\n[Iter {iteration}] CFR eval vs greedy: {metrics.win_rate_vs_greedy:.2%}")
            print(f"[Iter {iteration}] CFR eval vs random: {metrics.win_rate_vs_random:.2%}")
            print(f"[Iter {iteration}] CFR eval vs smart: {metrics.win_rate_vs_smart:.2%}")
            print(f"[Iter {iteration}] Avg regret loss: {regret_loss_history[-1]:.4f}")
            print(f"[Iter {iteration}] Avg strategy loss: {strategy_loss_history[-1]:.4f}")

    return avg_policy, regret_loss_history, strategy_loss_history, eval_episodes, win_rates


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create a CLI for running Deep MC-CFR training."""
    parser = argparse.ArgumentParser(description="Train Deep Monte-Carlo CFR for Big 2.")
    parser.add_argument("--n-players", type=int, default=2, help="Number of players in the game.")
    parser.add_argument("--cards-per-player", type=int, default=5, help="Cards dealt to each player.")
    parser.add_argument("--iterations", type=int, default=200, help="Number of CFR iterations.")
    parser.add_argument("--traversals-per-player", type=int, default=64, help="Traversals per player per iteration.")
    parser.add_argument("--regret-epochs", type=int, default=2, help="SGD epochs for regret networks.")
    parser.add_argument("--strategy-epochs", type=int, default=2, help="SGD epochs for average policy network.")
    parser.add_argument(
        "--regret-updates-per-player",
        type=int,
        default=128,
        help="Fixed gradient updates per player per regret epoch (independent of buffer size).",
    )
    parser.add_argument(
        "--strategy-updates-per-iteration",
        type=int,
        default=128,
        help="Fixed gradient updates per strategy epoch (independent of buffer size).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for training.")
    parser.add_argument("--regret-lr", type=float, default=5e-4, help="Learning rate for regret networks.")
    parser.add_argument("--strategy-lr", type=float, default=5e-4, help="Learning rate for policy network.")
    parser.add_argument("--regret-buffer-size", type=int, default=200_000, help="Reservoir size for regrets.")
    parser.add_argument("--strategy-buffer-size", type=int, default=200_000, help="Reservoir size for strategies.")
    parser.add_argument(
        "--policy-arch", type=str, default="setpool", choices=["mlp", "setpool"], help="Policy architecture."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--eval-interval", type=int, default=40, help="Evaluate every N iterations.")
    parser.add_argument("--eval-games", type=int, default=200, help="Games per evaluation.")
    parser.add_argument("--device", type=str, default="", help="Override device (cpu/cuda).")
    parser.add_argument(
        "--reinit-regret-nets-each-iteration",
        action="store_true",
        help="Reinitialize regret networks from scratch at each CFR iteration.",
    )
    parser.add_argument(
        "--reinit-avg-policy-each-iteration",
        action="store_true",
        help="Reinitialize average policy network from scratch at each CFR iteration.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/big2_mc_cfr_policy.pt",
        help="Path to save the average policy.",
    )
    return parser


def _main() -> None:
    """CLI entrypoint for Deep MC-CFR training."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Resolve device preference with a simple override flag.
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = CFRConfig(
        n_players=args.n_players,
        cards_per_player=args.cards_per_player,
        iterations=args.iterations,
        traversals_per_player=args.traversals_per_player,
        regret_epochs=args.regret_epochs,
        strategy_epochs=args.strategy_epochs,
        regret_updates_per_player=args.regret_updates_per_player,
        strategy_updates_per_iteration=args.strategy_updates_per_iteration,
        regret_lr=args.regret_lr,
        strategy_lr=args.strategy_lr,
        batch_size=args.batch_size,
        regret_buffer_size=args.regret_buffer_size,
        strategy_buffer_size=args.strategy_buffer_size,
        device=device,
        policy_arch=args.policy_arch,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        reinit_regret_nets_each_iteration=args.reinit_regret_nets_each_iteration,
        reinit_avg_policy_each_iteration=args.reinit_avg_policy_each_iteration,
    )

    policy, _regret_loss_history, _strategy_loss_history, _eval_episodes, _win_rates = train_mc_cfr(config)

    # Save the average strategy policy.
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), str(save_path))
    print(f"Saved average policy to {save_path}")

    # Quick sanity print: show a random hand for reference.
    total_cards_in_play = config.n_players * config.cards_per_player
    hand = sorted(random.sample(range(total_cards_in_play), config.cards_per_player))
    print("Random starting hand:", [card_name(c) for c in hand])


if __name__ == "__main__":
    _main()
