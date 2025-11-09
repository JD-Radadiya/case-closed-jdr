"""Hybrid Minimax + RL agent implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from case_closed_game import Agent, Direction, Game
from heuristics import (
    evaluate_aggressive_position,
    evaluate_exploration_value,
    evaluate_survival_probability,
    evaluate_territory_control,
    evaluate_trail_length_advantage,
    evaluate_distance_to_opponent,
    estimate_opponent_move_probabilities,
    get_safe_moves,
)
from minimax_agent import MinimaxAgent


ACTIONS: Sequence[Tuple[Direction, bool]] = (
    (Direction.UP, False),
    (Direction.DOWN, False),
    (Direction.LEFT, False),
    (Direction.RIGHT, False),
    (Direction.UP, True),
    (Direction.DOWN, True),
    (Direction.LEFT, True),
    (Direction.RIGHT, True),
)

ACTION_INDEX: Dict[Tuple[Direction, bool], int] = {
    action: index for index, action in enumerate(ACTIONS)
}


def extract_state_features(game: Game, agent_id: int) -> List[float]:
    """Extract normalized features describing the current state."""
    agent = game.agent1 if agent_id == 1 else game.agent2
    opponent = game.agent2 if agent_id == 1 else game.agent1

    territory = evaluate_territory_control(game, agent, max_depth=8)
    survival = evaluate_survival_probability(game, agent)
    aggression = evaluate_aggressive_position(game, agent)
    exploration = evaluate_exploration_value(game, agent)
    our_safe = len(get_safe_moves(game, agent)) / 3.0
    opp_safe = len(get_safe_moves(game, opponent)) / 3.0
    proximity = 1.0 - evaluate_distance_to_opponent(game, agent)
    boost_diff = (agent.boosts_remaining - opponent.boosts_remaining) / 3.0
    trail_advantage = (evaluate_trail_length_advantage(game, agent) + 1.0) / 2.0
    total_length = agent.length + opponent.length
    length_ratio = agent.length / total_length if total_length > 0 else 0.5
    turn_progress = min(game.turns / 200.0, 1.0)

    return [
        territory,
        survival,
        aggression,
        exploration,
        our_safe,
        opp_safe,
        proximity,
        boost_diff,
        trail_advantage,
        length_ratio,
        turn_progress,
    ]


@dataclass
class HybridPolicy:
    """Simple linear-softmax policy over handcrafted features."""

    num_features: int
    actions: Sequence[Tuple[Direction, bool]]
    weights: List[List[float]]
    bias: List[float]

    @classmethod
    def initialize(
        cls,
        num_features: int,
        actions: Sequence[Tuple[Direction, bool]],
        weights: List[List[float]] | None = None,
        bias: List[float] | None = None,
    ) -> "HybridPolicy":
        action_count = len(actions)
        if weights is None:
            weights = [[0.0 for _ in range(num_features)] for _ in range(action_count)]
        if bias is None:
            bias = [0.0 for _ in range(action_count)]
        return cls(num_features=num_features, actions=actions, weights=weights, bias=bias)

    def action_distribution(self, features: Sequence[float]) -> List[float]:
        logits = []
        for action_index, action_weights in enumerate(self.weights):
            logit = self.bias[action_index]
            logit += sum(w * f for w, f in zip(action_weights, features))
            logits.append(logit)

        max_logit = max(logits)
        exp_values = [math.exp(logit - max_logit) for logit in logits]
        total = sum(exp_values)
        if total == 0:
            return [1.0 / len(exp_values) for _ in exp_values]
        return [value / total for value in exp_values]

    def copy(self) -> "HybridPolicy":
        return HybridPolicy.initialize(
            self.num_features,
            self.actions,
            weights=[row[:] for row in self.weights],
            bias=self.bias[:],
        )


class HybridMinimaxRLAgent:
    """Agent that fuses a learned policy with a minimax fallback."""

    def __init__(
        self,
        agent_id: int = 1,
        policy_weights: List[List[float]] | None = None,
        policy_bias: List[float] | None = None,
        minimax_depth: int = 3,
        minimax_weight: float = 0.4,
        risk_weight: float = 0.5,
        aggressive_weight: float = 0.33,
        exploration_weight: float = 0.33,
        safety_weight: float = 0.34,
    ):
        self.agent_id = agent_id
        dummy_game = Game()
        dummy_features = extract_state_features(dummy_game, agent_id)
        self.policy = HybridPolicy.initialize(
            num_features=len(dummy_features),
            actions=ACTIONS,
            weights=policy_weights,
            bias=policy_bias,
        )
        self.minimax_agent = MinimaxAgent(
            agent_id=agent_id,
            depth=minimax_depth,
            aggressive_weight=aggressive_weight,
            exploration_weight=exploration_weight,
            safety_weight=safety_weight,
        )
        self.minimax_weight = minimax_weight
        self.risk_weight = risk_weight
        self.last_distribution: List[float] | None = None
        self.last_chosen_index: int | None = None

    def get_best_move(self, game: Game) -> Tuple[Direction, bool]:
        features = extract_state_features(game, self.agent_id)
        distribution = self.policy.action_distribution(features)
        self.last_distribution = distribution

        agent = game.agent1 if self.agent_id == 1 else game.agent2
        safe_moves = get_safe_moves(game, agent)
        minimax_move, minimax_boost = self.minimax_agent.get_best_move(game)

        if not safe_moves:
            return minimax_move, minimax_boost

        valid_actions: List[Tuple[float, int, Tuple[Direction, bool]]] = []
        for index, (direction, use_boost) in enumerate(ACTIONS):
            if not self._is_action_valid(agent, safe_moves, direction, use_boost):
                continue
            score = distribution[index]
            if direction == minimax_move and use_boost == minimax_boost:
                score += self.minimax_weight
            risk_penalty = self._estimate_collision_risk(game, direction)
            score -= self.risk_weight * risk_penalty
            valid_actions.append((score, index, (direction, use_boost)))

        if not valid_actions:
            return minimax_move, minimax_boost

        best_score, best_index, best_action = max(valid_actions, key=lambda item: item[0])
        self.last_chosen_index = best_index
        return best_action

    def _is_action_valid(
        self,
        agent: Agent,
        safe_moves: Sequence[Direction],
        direction: Direction,
        use_boost: bool,
    ) -> bool:
        current_dx, current_dy = agent.direction.value
        requested_dx, requested_dy = direction.value
        if (requested_dx, requested_dy) == (-current_dx, -current_dy):
            return False
        if use_boost and agent.boosts_remaining <= 0:
            return False
        if direction not in safe_moves:
            return False
        return True

    def _estimate_collision_risk(self, game: Game, direction: Direction) -> float:
        agent = game.agent1 if self.agent_id == 1 else game.agent2
        opponent = game.agent2 if self.agent_id == 1 else game.agent1
        safe_moves = get_safe_moves(game, opponent)
        if not safe_moves:
            return 0.0

        distribution = estimate_opponent_move_probabilities(
            game,
            agent,
            opponent,
            safe_moves,
        )
        head = agent.trail[-1]
        next_pos = game.board._torus_check((head[0] + direction.value[0], head[1] + direction.value[1]))

        risk = 0.0
        for opp_direction, probability in distribution.items():
            opp_head = opponent.trail[-1]
            opp_next = game.board._torus_check(
                (opp_head[0] + opp_direction.value[0], opp_head[1] + opp_direction.value[1])
            )
            if opp_next == next_pos:
                risk += probability
        return min(risk, 1.0)

    def get_stats(self) -> Dict[str, float]:
        stats = {}
        if self.last_distribution is not None:
            for idx, prob in enumerate(self.last_distribution):
                direction, use_boost = ACTIONS[idx]
                key = f"policy_{direction.name.lower()}_{'boost' if use_boost else 'move'}"
                stats[key] = prob
        if hasattr(self, "last_chosen_index"):
            direction, use_boost = ACTIONS[self.last_chosen_index]
            stats["last_action"] = f"{direction.name}:{int(use_boost)}"
        minimax_stats = self.minimax_agent.get_stats()
        stats.update({f"minimax_{k}": v for k, v in minimax_stats.items()})
        return stats

    def export_policy(self) -> Dict[str, List[List[float]] | List[float]]:
        return {
            "weights": [row[:] for row in self.policy.weights],
            "bias": self.policy.bias[:],
        }
