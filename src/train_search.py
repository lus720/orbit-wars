"""Expert iteration training for search-enhanced policy.

Loop:
  1. Play games with SearchAgent → collect (state, action, outcome) data
  2. Train policy to imitate search (BC) + value to predict outcome
  3. Use improved policy in search
  4. Evaluate → repeat
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    TrainConfig,
    default_train_config_path,
    load_train_config,
)
from .env import OrbitWarsEnv
from .features import (
    TurnBatch,
    candidate_feature_dim,
    global_feature_dim,
    self_feature_dim,
)
from .opponents import build_opponent
from .policy import PlanetPolicy
from .search_agent import SearchAgent


@dataclass(slots=True)
class SearchTransition:
    self_features: np.ndarray
    candidate_features: np.ndarray
    global_features: np.ndarray
    candidate_mask: np.ndarray
    ship_option_mask: np.ndarray
    target_index: int
    ship_choice: int
    search_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(default_train_config_path()))
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class SearchTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        policy: PlanetPolicy,
        device: torch.device,
        run_name: str,
    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self.device = device
        self.run_name = run_name
        train_cfg = cfg.train_search
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=train_cfg.lr)
        self.replay_buffer: list[SearchTransition] = []
        self.max_buffer_size = train_cfg.replay_buffer_size
        self.batch_size = train_cfg.batch_size

    def collect_data(
        self, num_games: int, training: bool = True
    ) -> dict[str, float]:
        """Play games with search agent, collecting (state, action, outcome)."""
        opponent = build_opponent(self.cfg.opponent, cfg=self.cfg, device=self.device)
        search_agent = SearchAgent(
            self.policy, self.cfg, self.device, deterministic=not training
        )

        transitions: list[SearchTransition] = []
        total_steps = 0
        total_search_time = 0.0
        wins = 0
        losses = 0

        for game_idx in range(num_games):
            env = OrbitWarsEnv(self.cfg, opponent, env_index=0)
            batch = env.reset(seed=game_idx + 42)
            print(f"  [Game {game_idx + 1}/{num_games}] 对手={'self' if game_idx % 2 == 0 else 'baseline'} 开始", flush=True)

            game_transitions: list[SearchTransition] = []
            step_count = 0
            while True:
                obs = env.last_obs
                t0 = time.time()
                # Use search to select actions
                actions = search_agent.act(obs) if obs else []
                search_ms = (time.time() - t0) * 1000
                total_search_time += search_ms

                # Record the state and what search chose
                self._record_transitions(batch, actions, game_transitions)

                result = env.step(actions)
                batch = result.batch
                total_steps += 1
                step_count += 1

                total_steps_str = f"#{step_count:>4d}"
                print(
                    f"    step {total_steps_str}  "
                    f"search={search_ms:>5.0f}ms  "
                    f"sources={batch.self_features.shape[0]:>2d}  "
                    f"actions={len(actions):>2d}  "
                    f"game={game_idx + 1}/{num_games}",
                    flush=True,
                )

                if result.done:
                    # Compute outcome: did we win?
                    outcome = self._game_outcome(env)
                    if outcome > 0:
                        wins += 1
                    elif outcome < 0:
                        losses += 1

                    # Assign outcome to collected transitions
                    for t in game_transitions:
                        t.search_value = float(outcome)
                    transitions.extend(game_transitions)
                    break

        self.replay_buffer.extend(transitions)
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.max_buffer_size:]

        avg_search_ms = total_search_time / max(total_steps, 1) * 1000
        return {
            "games": num_games,
            "steps": total_steps,
            "samples": len(transitions),
            "total_samples": len(self.replay_buffer),
            "win_rate": wins / max(num_games, 1),
            "loss_rate": losses / max(num_games, 1),
            "avg_search_time_ms": avg_search_ms,
        }

    def train_step(
        self, epochs: int = 5
    ) -> dict[str, float]:
        """Train policy + value on collected data."""
        buffer = self.replay_buffer
        batch_size = self.batch_size
        if len(buffer) < batch_size:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "buffer_samples": len(buffer), "batches": 0, "epochs": 0}

        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        updates = 0

        for _ in range(epochs):
            order = np.random.permutation(len(buffer))
            for start in range(0, len(buffer), batch_size):
                idx = order[start : start + batch_size]
                batch = [buffer[i] for i in idx]

                # Stack batch
                sf = torch.from_numpy(
                    np.asarray([t.self_features for t in batch], dtype=np.float32)
                ).to(self.device)
                cf = torch.from_numpy(
                    np.asarray([t.candidate_features for t in batch], dtype=np.float32)
                ).to(self.device)
                gf = torch.from_numpy(
                    np.asarray([t.global_features for t in batch], dtype=np.float32)
                ).to(self.device)
                cm = torch.from_numpy(
                    np.asarray([t.candidate_mask for t in batch], dtype=bool)
                ).to(self.device)
                sm = torch.from_numpy(
                    np.asarray([t.ship_option_mask for t in batch], dtype=bool)
                ).to(self.device)
                targets = torch.tensor(
                    [t.target_index for t in batch], dtype=torch.long, device=self.device
                )
                ships = torch.tensor(
                    [t.ship_choice for t in batch], dtype=torch.long, device=self.device
                )
                values = torch.tensor(
                    [t.search_value for t in batch], dtype=torch.float32, device=self.device
                )

                outputs = self.policy(sf, cf, gf, cm, sm)

                # Safety: replace masked targets with noop (index 0)
                valid_targets = cm.gather(1, targets.unsqueeze(1)).squeeze(1)
                safe_targets = torch.where(valid_targets, targets, torch.zeros_like(targets))

                # Policy loss: cross-entropy on target choice
                target_logits = outputs.target_logits
                target_logits = target_logits.masked_fill(
                    ~cm, torch.finfo(target_logits.dtype).min
                )
                policy_loss = F.cross_entropy(target_logits, safe_targets)

                # Value loss: MSE
                value_loss = F.mse_loss(outputs.value, values)

                # Combined loss
                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

                total_loss += float(loss.detach())
                total_policy += float(policy_loss.detach())
                total_value += float(value_loss.detach())
                updates += 1

        num_updates = max(updates, 1)
        return {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy / num_updates,
            "value_loss": total_value / num_updates,
            "buffer_samples": len(buffer),
            "batches": updates,
            "epochs": epochs,
        }

    def evaluate(self, num_games: int = 20) -> dict[str, float]:
        """Evaluate current policy against baseline."""
        from .opponents import load_baseline_agent
        from kaggle_environments import make
        from pathlib import Path

        baseline = load_baseline_agent(
            Path("baselines/nearest_planet_sniper.py"), 999
        )
        search_agent = SearchAgent(
            self.policy, self.cfg, self.device, deterministic=True
        )

        wins = 0
        draws = 0
        losses = 0

        for game_idx in range(num_games):
            if game_idx % 5 == 0:
                print(f"    eval game {game_idx + 1}/{num_games}...", flush=True)
            env = make("orbit_wars", debug=False)
            env.reset(num_agents=2)

            while True:
                obs = env.state[0].observation
                opp_obs = env.state[1].observation

                our_actions = search_agent.act(obs) if obs else []
                opp_actions = baseline(opp_obs) if opp_obs else []

                states = env.step([our_actions, opp_actions])
                if states[0].status != "ACTIVE":
                    break

            our_score = sum(p[5] for p in states[0].observation.get("planets", []))
            for f in states[0].observation.get("fleets", []):
                our_score += f[6]
            their_score = sum(p[5] for p in states[1].observation.get("planets", []))
            for f in states[1].observation.get("fleets", []):
                their_score += f[6]

            if our_score > their_score:
                wins += 1
            elif our_score < their_score:
                losses += 1
            else:
                draws += 1

        return {
            "win_rate": wins / max(num_games, 1),
            "loss_rate": losses / max(num_games, 1),
            "draw_rate": draws / max(num_games, 1),
            "games": num_games,
        }

    def save_checkpoint(self, iteration: int) -> None:
        save_dir = Path(self.cfg.save_dir) / self.run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "iteration": iteration,
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.cfg,
            },
            save_dir / f"search_iter_{iteration:04d}.pt",
        )
        torch.save(
            {
                "iteration": iteration,
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.cfg,
            },
            save_dir / "search_last.pt",
        )

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError):
            pass
        return ckpt.get("iteration", 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_transitions(
        self,
        batch: TurnBatch,
        actions: list[list],
        game_transitions: list[SearchTransition],
    ) -> None:
        """Record one transition per source planet from this step."""
        if batch.self_features.shape[0] == 0:
            return

        # Build action lookup: src_id → (target_idx_in_candidates, ship_idx)
        action_lookup: dict[int, tuple[int, int]] = {}
        for act in actions:
            if len(act) < 3:
                continue
            src_id = int(act[0])
            angle = float(act[1])
            ships = int(act[2])
            for ctx in batch.contexts:
                if ctx.source_id != src_id:
                    continue
                best_match = None
                for cand_idx in range(1, len(ctx.candidate_ids)):
                    for ship_idx in range(ctx.ship_options.shape[1]):
                        opt_ships = int(ctx.ship_options[cand_idx, ship_idx])
                        opt_angle = float(ctx.target_angles[cand_idx, ship_idx])
                        if opt_ships == ships and abs(opt_angle - angle) < 1e-6:
                            best_match = (cand_idx, ship_idx)
                            break
                    if best_match is not None:
                        break

                # Fallback: match on ships only (greedy-first)
                if best_match is None:
                    for cand_idx in range(1, len(ctx.candidate_ids)):
                        for ship_idx in range(ctx.ship_options.shape[1]):
                            if int(ctx.ship_options[cand_idx, ship_idx]) == ships:
                                best_match = (cand_idx, ship_idx)
                                break
                        if best_match is not None:
                            break

                if best_match is not None:
                    action_lookup[src_id] = best_match
                break

        for row_idx, ctx in enumerate(batch.contexts):
            target_idx, ship_choice = action_lookup.get(ctx.source_id, (0, 0))

            game_transitions.append(
                SearchTransition(
                    self_features=batch.self_features[row_idx].copy(),
                    candidate_features=batch.candidate_features[row_idx].copy(),
                    global_features=batch.global_features[row_idx].copy(),
                    candidate_mask=batch.candidate_mask[row_idx].copy(),
                    ship_option_mask=batch.ship_option_mask[row_idx].copy(),
                    target_index=target_idx,
                    ship_choice=ship_choice,
                    search_value=0.0,  # filled after game ends
                )
            )

    def _game_outcome(self, env: OrbitWarsEnv) -> float:
        """Return +1 for win, -1 for loss, 0 for draw."""
        player_state = env.env.state[env.learner_player]
        opp_state = env.env.state[1 - env.learner_player]
        from .env import terminal_reward

        return float(terminal_reward(player_state, opp_state))


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)

    device = resolve_device(cfg.device)
    policy = PlanetPolicy(
        self_dim=self_feature_dim(),
        candidate_dim=candidate_feature_dim(),
        global_dim=global_feature_dim(),
        candidate_count=cfg.env.candidate_count,
        ship_option_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        noop_logit_bias=cfg.model.noop_logit_bias,
        heuristic_logit_scale=cfg.model.heuristic_logit_scale,
    ).to(device)
    policy.train()

    trainer = SearchTrainer(cfg, policy, device, cfg.run_name)

    start_iter = 0
    if args.checkpoint:
        start_iter = trainer.load_checkpoint(args.checkpoint)
        print(f"Loaded checkpoint at iteration {start_iter}", flush=True)

    train_cfg = cfg.train_search
    # Reduce simulation horizon for faster data collection
    cfg.search.simulation_horizon = 3

    print(
        f"\n{'='*60}\n"
        f"  开始训练: {cfg.run_name}\n"
        f"  设备: {device}\n"
        f"  迭代次数: {train_cfg.iterations}\n"
        f"  每迭代对局: {train_cfg.games_per_iter}\n"
        f"  训练 epochs: {train_cfg.epochs}\n"
        f"  batch size: {train_cfg.batch_size}\n"
        f"  学习率: {train_cfg.lr}\n"
        f"  搜索推演步数: {cfg.search.simulation_horizon}\n"
        f"{'='*60}",
        flush=True,
    )

    for iteration in range(start_iter + 1, start_iter + train_cfg.iterations + 1):
        print(f"\n========== Iteration {iteration}/{start_iter + train_cfg.iterations} ==========", flush=True)

        # 1. 收集数据: 搜索增强策略自我对局
        collect_stats = trainer.collect_data(train_cfg.games_per_iter, training=True)
        win_rate = collect_stats["win_rate"]
        loss_rate = collect_stats["loss_rate"]
        print(
            f"  >>> 数据收集完成\n"
            f"      对局: {collect_stats['games']} 局\n"
            f"      总步数: {collect_stats['steps']}\n"
            f"      样本: {collect_stats['samples']} (缓冲区共 {collect_stats['total_samples']})\n"
            f"      胜率: {win_rate:.1%}  负率: {loss_rate:.1%}\n"
            f"      搜索平均耗时: {collect_stats['avg_search_time_ms']:.0f}ms/步",
            flush=True,
        )

        # 2. 训练: 监督学习策略网络模仿搜索决策
        train_stats = trainer.train_step(epochs=train_cfg.epochs)
        print(
            f"  >>> 训练完成\n"
            f"      loss={train_stats['loss']:.4f}  "
            f"policy={train_stats['policy_loss']:.4f}  "
            f"value={train_stats['value_loss']:.4f}\n"
            f"      缓冲区样本: {train_stats['buffer_samples']}  "
            f"batches: {train_stats['batches']}  "
            f"epochs: {train_stats['epochs']}",
            flush=True,
        )

        # 3. 评估: 与 baseline 对战
        eval_stats = trainer.evaluate(num_games=10)
        print(
            f"  >>> 评估 vs baseline\n"
            f"      胜: {eval_stats['win_rate']:.0%}  "
            f"平: {eval_stats['draw_rate']:.0%}  "
            f"负: {eval_stats['loss_rate']:.0%}",
            flush=True,
        )

        # 4. 保存 checkpoint
        if iteration % train_cfg.checkpoint_every == 0 or iteration == 1:
            trainer.save_checkpoint(iteration)
            print(f"  >>> 模型已保存 (iteration {iteration})", flush=True)


if __name__ == "__main__":
    main()
