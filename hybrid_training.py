"""Training utilities for the hybrid minimax + RL agent."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from case_closed_game import Game, GameResult
from hybrid_agent import ACTIONS, ACTION_INDEX, HybridMinimaxRLAgent, extract_state_features
from match_runner import MatchRunner
from reward_functions import calculate_reward


@dataclass
class TrajectoryStep:
    features: List[float]
    action_index: int
    reward: float = 0.0


class HybridMinimaxRLTrainer:
    """Orchestrates minimax-guided rollouts and policy-gradient updates."""

    def __init__(
        self,
        match_runner: MatchRunner,
        agent_id: int = 1,
        teacher_depth: int = 4,
        learner_depth: int = 3,
        learning_rate: float = 0.05,
        gamma: float = 0.95,
        entropy_coef: float = 0.0,
        minimax_weight: float = 0.4,
        risk_weight: float = 0.5,
        aggressive_weight: float = 0.33,
        exploration_weight: float = 0.33,
        safety_weight: float = 0.34,
    ):
        self.match_runner = match_runner
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.agent = HybridMinimaxRLAgent(
            agent_id=agent_id,
            minimax_depth=learner_depth,
            minimax_weight=minimax_weight,
            risk_weight=risk_weight,
            aggressive_weight=aggressive_weight,
            exploration_weight=exploration_weight,
            safety_weight=safety_weight,
        )
        self.teacher_config = {
            "type": "minimax",
            "agent_id": agent_id,
            "depth": teacher_depth,
            "aggressive_weight": aggressive_weight,
            "exploration_weight": exploration_weight,
            "safety_weight": safety_weight,
            "use_transposition_table": True,
            "tt_max_size": 20000,
        }
        self.opponent_agent_id = 2 if agent_id == 1 else 1

    def train_iteration(
        self,
        opponents: Sequence[Tuple[str, Dict[str, float]]],
        matches_per_opponent: int,
    ) -> Dict[str, Dict[str, float]]:
        """Collect rollouts against opponents and update policy once."""
        all_trajectories: List[List[TrajectoryStep]] = []
        summary: Dict[str, Dict[str, float]] = {}

        for name, opponent_config in opponents:
            trajectories, stats = self._collect_teacher_trajectories(
                opponent_config, matches_per_opponent
            )
            if trajectories:
                all_trajectories.extend(trajectories)
            summary[name] = stats

        if all_trajectories:
            self._apply_policy_gradient(all_trajectories)

        return summary

    def _collect_teacher_trajectories(
        self, opponent_config: Dict[str, float], num_matches: int
    ) -> Tuple[List[List[TrajectoryStep]], Dict[str, float]]:
        trajectories: List[List[TrajectoryStep]] = []
        wins = losses = draws = 0
        total_reward = 0.0
        total_turns = 0

        for _ in range(num_matches):
            game = Game()
            teacher = self.match_runner._create_agent(dict(self.teacher_config))
            opponent_cfg = dict(opponent_config)
            opponent_cfg["agent_id"] = self.opponent_agent_id
            opponent = self.match_runner._create_agent(opponent_cfg)

            steps: List[TrajectoryStep] = []
            result: GameResult | None = None

            while True:
                features = extract_state_features(game, self.agent.agent_id)
                teacher_move, teacher_boost = teacher.get_best_move(game)
                opponent_move, opponent_boost = opponent.get_best_move(game)

                action_index = ACTION_INDEX.get((teacher_move, teacher_boost))
                if action_index is None:
                    # Treat unexpected boost usage as non-boost action fallback.
                    action_index = ACTION_INDEX[(teacher_move, False)]

                steps.append(TrajectoryStep(features=features, action_index=action_index))

                result = game.step(teacher_move, opponent_move, teacher_boost, opponent_boost)
                if result is not None:
                    break

            reward = calculate_reward(game, self.agent.agent_id, result)
            total_reward += reward
            total_turns += game.turns

            for step in steps:
                step.reward = reward

            if result == GameResult.DRAW:
                draws += 1
            elif (
                result == GameResult.AGENT1_WIN and self.agent.agent_id == 1
            ) or (
                result == GameResult.AGENT2_WIN and self.agent.agent_id == 2
            ):
                wins += 1
            else:
                losses += 1

            trajectories.append(steps)

        matches_played = max(1, num_matches)
        stats = {
            "avg_reward": total_reward / matches_played,
            "avg_turns": total_turns / matches_played,
            "win_rate": wins / matches_played,
            "draw_rate": draws / matches_played,
            "loss_rate": losses / matches_played,
        }

        return trajectories, stats

    def _apply_policy_gradient(self, trajectories: Iterable[List[TrajectoryStep]]) -> None:
        policy = self.agent.policy
        action_count = len(policy.actions)

        for steps in trajectories:
            if not steps:
                continue
            returns: List[float] = []
            cumulative = 0.0
            for step in reversed(steps):
                cumulative = step.reward + self.gamma * cumulative
                returns.append(cumulative)
            returns.reverse()

            baseline = sum(returns) / len(returns)

            for step, ret in zip(steps, returns):
                advantage = ret - baseline
                probs = policy.action_distribution(step.features)

                for action_index in range(action_count):
                    indicator = 1.0 if action_index == step.action_index else 0.0
                    grad = indicator - probs[action_index]

                    for feature_index, feature in enumerate(step.features):
                        policy.weights[action_index][feature_index] += (
                            self.learning_rate * advantage * grad * feature
                        )

                    policy.bias[action_index] += self.learning_rate * advantage * grad

                if self.entropy_coef > 0.0:
                    entropy = -sum(p * (0.0 if p <= 0 else math.log(p)) for p in probs)
                    entropy_grad = self.entropy_coef * entropy
                    for action_index in range(action_count):
                        for feature_index in range(policy.num_features):
                            policy.weights[action_index][feature_index] += entropy_grad

    def build_agent_config(self) -> Dict[str, object]:
        policy_data = self.agent.export_policy()
        minimax = self.agent.minimax_agent
        return {
            "agent_id": self.agent.agent_id,
            "policy_weights": policy_data["weights"],
            "policy_bias": policy_data["bias"],
            "minimax_depth": minimax.depth,
            "minimax_weight": self.agent.minimax_weight,
            "risk_weight": self.agent.risk_weight,
            "aggressive_weight": minimax.aggressive_weight,
            "exploration_weight": minimax.exploration_weight,
            "safety_weight": minimax.safety_weight,
        }
