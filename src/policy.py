
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass(slots=True)
class PolicyOutput:
    target_logits: torch.Tensor
    ship_logits: torch.Tensor
    value: torch.Tensor


class PlanetPolicy(nn.Module):
    def __init__(
        self,
        self_dim: int,
        candidate_dim: int,
        global_dim: int,
        candidate_count: int,
        ship_option_count: int,
        hidden_size: int = 128,
        noop_logit_bias: float = 0.0,
        heuristic_logit_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.candidate_count = candidate_count
        self.ship_option_count = ship_option_count
        self.noop_logit_bias = noop_logit_bias
        self.heuristic_logit_scale = heuristic_logit_scale
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.target_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.ship_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ship_option_count),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        self_features: torch.Tensor,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        ship_option_mask: torch.Tensor,
    ) -> PolicyOutput:
        self_hidden = self.self_encoder(self_features)
        global_hidden = self.global_encoder(global_features)
        candidate_hidden = self.candidate_encoder(candidate_features)
        expanded_self = self_hidden.unsqueeze(1).expand(-1, self.candidate_count, -1)
        expanded_global = global_hidden.unsqueeze(1).expand(-1, self.candidate_count, -1)
        joint = torch.cat([expanded_self, expanded_global, candidate_hidden], dim=-1)
        target_logits = self.target_head(joint).squeeze(-1)
        if self.heuristic_logit_scale != 0.0 and candidate_features.shape[-1] >= 43:
            target_logits = target_logits + self.heuristic_logit_scale * heuristic_candidate_prior(candidate_features)
        target_logits = target_logits.masked_fill(~candidate_mask, torch.finfo(target_logits.dtype).min)
        if self.candidate_count > 0 and self.noop_logit_bias != 0.0:
            actionable_rows = candidate_mask[:, 1:].any(dim=-1)
            target_logits[actionable_rows, 0] = target_logits[actionable_rows, 0] + self.noop_logit_bias
        ship_logits = self.ship_head(joint)
        ship_logits = ship_logits.masked_fill(~ship_option_mask, torch.finfo(ship_logits.dtype).min)
        pooled_candidates = candidate_hidden.mean(dim=1)
        value = self.value_head(torch.cat([self_hidden, global_hidden, pooled_candidates], dim=-1)).squeeze(-1)
        return PolicyOutput(target_logits=target_logits, ship_logits=ship_logits, value=value)

    @torch.inference_mode()
    def get_action_candidates(
        self,
        self_features: torch.Tensor,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        ship_option_mask: torch.Tensor,
        contexts: list[Any],
        top_k: int = 3,
    ) -> list[list[dict]]:
        """Generate top-K action candidates per source planet for search.

        Returns a list (per source planet) of lists (candidates):
        [[{target_idx, ship_idx, ships, src_id, angle, target_id}, ...], ...]
        Each candidate is a dict with fields needed to execute and evaluate the action.
        """
        outputs = self.forward(
            self_features, candidate_features, global_features,
            candidate_mask, ship_option_mask,
        )
        target_logits = outputs.target_logits  # [B, N]
        ship_logits = outputs.ship_logits      # [B, N, S]

        source_candidates: list[list[dict]] = []
        batch_size = self_features.shape[0]

        for row_idx in range(batch_size):
            ctx = contexts[row_idx]
            mask = candidate_mask[row_idx]  # [N]

            if not mask[1:].any():
                source_candidates.append([])
                continue

            # Get top-K targets (excluding noop)
            valid_scores = target_logits[row_idx].clone()
            valid_scores[~mask] = float("-inf")
            valid_scores[0] = float("-inf")  # exclude noop
            k = min(top_k, int(mask[1:].sum().item()))
            topk_values, topk_indices = torch.topk(valid_scores, k, dim=-1)

            row_candidates: list[dict] = []
            for target_idx in topk_indices.tolist():
                # Get valid ship options for this target
                ship_mask = ship_option_mask[row_idx, target_idx]  # [S]
                ship_scores = ship_logits[row_idx, target_idx]     # [S]
                ship_scores[~ship_mask] = float("-inf")
                valid_ship_indices = torch.where(ship_mask)[0]

                for ship_idx in valid_ship_indices.tolist():
                    ships = int(ctx.ship_options[target_idx, ship_idx])
                    angle = float(ctx.target_angles[target_idx, ship_idx])
                    if ships <= 0:
                        continue
                    row_candidates.append({
                        "target_idx": target_idx,
                        "ship_idx": ship_idx,
                        "ships": ships,
                        "src_id": ctx.source_id,
                        "angle": angle,
                        "target_id": ctx.candidate_ids[target_idx],
                        "score": float(target_logits[row_idx, target_idx].cpu()),
                    })

            source_candidates.append(row_candidates)

        return source_candidates


@dataclass(slots=True)
class SampledAction:
    target_index: torch.Tensor
    ship_choice: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


def sample_actions(outputs: PolicyOutput, deterministic: bool, learn_ship_policy: bool = True) -> SampledAction:
    target_logits = safe_target_logits(outputs.target_logits)
    target_dist = Categorical(logits=target_logits)
    target_index = target_logits.argmax(dim=-1) if deterministic else target_dist.sample()
    selected_ship_logits = gather_ship_logits(outputs.ship_logits, target_index)
    if learn_ship_policy:
        selected_ship_logits = safe_target_logits(selected_ship_logits)
        ship_dist = Categorical(logits=selected_ship_logits)
        ship_choice = selected_ship_logits.argmax(dim=-1) if deterministic else ship_dist.sample()
    else:
        ship_choice = default_ship_choice(selected_ship_logits)
    log_prob, entropy = action_log_prob_and_entropy(
        outputs=outputs,
        target_index=target_index,
        ship_choice=ship_choice,
        learn_ship_policy=learn_ship_policy,
    )
    return SampledAction(target_index=target_index, ship_choice=ship_choice, log_prob=log_prob, entropy=entropy)


def action_log_prob_and_entropy(
    outputs: PolicyOutput,
    target_index: torch.Tensor,
    ship_choice: torch.Tensor,
    learn_ship_policy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_logits = safe_target_logits(outputs.target_logits)
    target_dist = Categorical(logits=target_logits)
    target_log_prob = target_dist.log_prob(target_index)
    target_entropy = target_dist.entropy()
    if not learn_ship_policy:
        return target_log_prob, target_entropy
    selected_ship_logits = gather_ship_logits(outputs.ship_logits, target_index)
    selected_ship_logits = safe_target_logits(selected_ship_logits)
    ship_dist = Categorical(logits=selected_ship_logits)
    ship_log_prob = ship_dist.log_prob(ship_choice)
    ship_entropy = ship_dist.entropy()
    return target_log_prob + ship_log_prob, target_entropy + ship_entropy


def safe_target_logits(target_logits: torch.Tensor) -> torch.Tensor:
    invalid_rows = ~torch.isfinite(target_logits).any(dim=-1)
    if not invalid_rows.any():
        return target_logits
    safe_logits = target_logits.clone()
    safe_logits[invalid_rows, 0] = 0.0
    return safe_logits


def gather_ship_logits(ship_logits: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
    gather_index = target_index.view(-1, 1, 1).expand(-1, 1, ship_logits.shape[-1])
    return ship_logits.gather(dim=1, index=gather_index).squeeze(1)


def default_ship_choice(selected_ship_logits: torch.Tensor) -> torch.Tensor:
    invalid_value = torch.finfo(selected_ship_logits.dtype).min
    valid_mask = selected_ship_logits != invalid_value
    fallback = torch.zeros((selected_ship_logits.shape[0],), dtype=torch.long, device=selected_ship_logits.device)
    if not valid_mask.any():
        return fallback
    return valid_mask.to(torch.int64).argmax(dim=-1)


def heuristic_candidate_prior(candidate_features: torch.Tensor) -> torch.Tensor:
    prior = torch.zeros_like(candidate_features[..., 0])
    if candidate_features.shape[1] <= 1:
        return prior
    owner_enemy = candidate_features[..., 1]
    owner_neutral = candidate_features[..., 2]
    distance = candidate_features[..., 22].clamp(min=0.0)
    capture_margin = candidate_features[..., 32].clamp(min=-1.0, max=1.0)
    production_efficiency = candidate_features[..., 33].clamp(min=0.0, max=3.0)
    travel_efficiency = candidate_features[..., 34].clamp(min=0.0, max=3.0)
    crosses_sun = candidate_features[..., 38].clamp(min=0.0, max=1.0)
    strategic_value = candidate_features[..., 41].clamp(min=0.0, max=3.0)
    candidate_prior = (
        2.0 * strategic_value
        + 1.0 * production_efficiency
        + 0.6 * travel_efficiency
        + 0.7 * capture_margin
        + 0.35 * owner_enemy
        + 0.2 * owner_neutral
        - 0.9 * distance
        - 5.0 * crosses_sun
    )
    prior = prior.clone()
    prior[..., 1:] = candidate_prior[..., 1:]
    return prior
