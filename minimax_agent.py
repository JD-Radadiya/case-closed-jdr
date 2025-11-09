"""Minimax algorithm with opponent modelling for Case Closed."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from typing import Dict, Iterable, List, Optional, Tuple

from case_closed_game import Agent, Direction, Game, GameBoard
from heuristics import (
    evaluate_game_state,
    estimate_opponent_move_probabilities,
    get_safe_moves,
)


class TranspositionTable:
    """Transposition table for caching game state evaluations."""

    def __init__(self, max_size: int = 10000):
        self.table: Dict[str, Tuple[float, int]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _hash_state(self, game: Game, agent_id: int) -> str:
        """Create a hash of the game state for caching."""
        state_data = {
            "board": [[cell for cell in row] for row in game.board.grid],
            "agent1_pos": list(game.agent1.trail),
            "agent2_pos": list(game.agent2.trail),
            "agent1_dir": game.agent1.direction.name,
            "agent2_dir": game.agent2.direction.name,
            "agent1_boosts": game.agent1.boosts_remaining,
            "agent2_boosts": game.agent2.boosts_remaining,
            "agent1_alive": game.agent1.alive,
            "agent2_alive": game.agent2.alive,
            "turn": game.turns,
            "agent_id": agent_id,
        }
        state_str = json.dumps(state_data, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()

    def get(self, game: Game, agent_id: int, depth: int) -> Optional[float]:
        state_hash = self._hash_state(game, agent_id)
        cached = self.table.get(state_hash)
        if cached:
            cached_score, cached_depth = cached
            if cached_depth >= depth:
                self.hits += 1
                return cached_score
        self.misses += 1
        return None

    def store(self, game: Game, agent_id: int, score: float, depth: int) -> None:
        state_hash = self._hash_state(game, agent_id)
        if len(self.table) >= self.max_size and self.table:
            first_key = next(iter(self.table))
            del self.table[first_key]
        self.table[state_hash] = (score, depth)

    def clear(self) -> None:
        self.table.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, int]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.table),
        }


class MinimaxAgent:
    """Minimax agent with alpha-beta pruning and opponent modelling."""

    def __init__(
        self,
        agent_id: int = 1,
        depth: int = 4,
        aggressive_weight: float = 0.33,
        exploration_weight: float = 0.33,
        safety_weight: float = 0.34,
        use_transposition_table: bool = True,
        tt_max_size: int = 10000,
    ):
        self.agent_id = agent_id
        self.depth = depth
        self.aggressive_weight = aggressive_weight
        self.exploration_weight = exploration_weight
        self.safety_weight = safety_weight
        self.use_transposition_table = use_transposition_table
        self.transposition_table = (
            TranspositionTable(tt_max_size) if use_transposition_table else None
        )
        self.nodes_evaluated = 0
        self.max_depth_reached = 0
        self.expectation_blend = 0.65
        self.history_influence = 0.5
        self.opponent_move_counts: Counter[Direction] = Counter(
            {
                Direction.UP: 1.0,
                Direction.DOWN: 1.0,
                Direction.LEFT: 1.0,
                Direction.RIGHT: 1.0,
            }
        )
        self.opponent_move_history: deque[Direction] = deque(maxlen=32)

    def get_best_move(self, game: Game) -> Tuple[Direction, bool]:
        """Return the best move for the current position."""
        self.nodes_evaluated = 0
        self.max_depth_reached = 0

        if self.use_transposition_table and self.transposition_table is not None:
            self.transposition_table.clear()

        self._update_opponent_model(game)

        agent = self._get_agent(game, self.agent_id)
        safe_moves = get_safe_moves(game, agent)
        if not safe_moves:
            return agent.direction, False

        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_action = (agent.direction, False)

        for direction in self._order_moves(game, agent, safe_moves):
            for use_boost in self._boost_options(agent):
                score = self._evaluate_root_action(game, direction, use_boost, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_action = (direction, use_boost)
                alpha = max(alpha, best_score)

        return best_action

    def _order_moves(
        self, game: Game, agent: Agent, moves: Iterable[Direction]
    ) -> List[Direction]:
        current_dir = agent.direction
        ordered: List[Direction] = []
        others: List[Direction] = []
        for move in moves:
            if move == current_dir:
                ordered.insert(0, move)
            else:
                others.append(move)
        return ordered + others

    def _boost_options(self, agent: Agent) -> Iterable[bool]:
        yield False
        if agent.boosts_remaining > 0:
            yield True

    @property
    def _opponent_id(self) -> int:
        return 2 if self.agent_id == 1 else 1

    def _evaluate_root_action(
        self,
        game: Game,
        direction: Direction,
        use_boost: bool,
        alpha: float,
        beta: float,
    ) -> float:
        next_game, our_alive, opp_alive = self._simulate_action(
            game, self.agent_id, direction, use_boost
        )

        if not our_alive and not opp_alive:
            return 0.0
        if not our_alive:
            return -1000.0 - self.depth
        if not opp_alive:
            return 1000.0 + self.depth

        return self._minimax_recursive(
            next_game, self.depth - 1, alpha, beta, self._opponent_id
        )

    def _minimax_recursive(
        self,
        game: Game,
        depth: int,
        alpha: float,
        beta: float,
        current_agent_id: int,
    ) -> float:
        self.nodes_evaluated += 1
        self.max_depth_reached = max(self.max_depth_reached, self.depth - depth)

        agent = self._get_agent(game, current_agent_id)
        opponent = self._get_agent(game, 1 if current_agent_id == 2 else 2)

        if not agent.alive or not opponent.alive:
            return self._evaluate_terminal(game)

        if depth <= 0:
            return self._evaluate_state(game)

        if self.use_transposition_table and self.transposition_table is not None:
            cached_score = self.transposition_table.get(game, current_agent_id, depth)
            if cached_score is not None:
                return cached_score

        if current_agent_id == self.agent_id:
            score = self._maximizing_step(game, agent, opponent, depth, alpha, beta)
        else:
            score = self._opponent_step(game, agent, opponent, depth, alpha, beta)

        if self.use_transposition_table and self.transposition_table is not None:
            self.transposition_table.store(game, current_agent_id, score, depth)

        return score

    def _maximizing_step(
        self,
        game: Game,
        agent: Agent,
        opponent: Agent,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        safe_moves = get_safe_moves(game, agent)
        if not safe_moves:
            return self._evaluate_state(game) - 750.0

        best_score = float("-inf")
        for direction in self._order_moves(game, agent, safe_moves):
            for use_boost in self._boost_options(agent):
                next_game, our_alive, opp_alive = self._simulate_action(
                    game, self.agent_id, direction, use_boost
                )

                if not our_alive and not opp_alive:
                    score = 0.0
                elif not our_alive:
                    score = -1000.0 - depth
                elif not opp_alive:
                    score = 1000.0 + depth
                else:
                    score = self._minimax_recursive(
                        next_game, depth - 1, alpha, beta, self._opponent_id
                    )

                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    return best_score

        return best_score

    def _opponent_step(
        self,
        game: Game,
        opponent: Agent,
        our_agent: Agent,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        safe_moves = get_safe_moves(game, opponent)
        if not safe_moves:
            return self._evaluate_state(game) + 750.0

        move_distribution = estimate_opponent_move_probabilities(
            game,
            our_agent,
            opponent,
            safe_moves,
            history_bias=self._calculate_history_bias(),
        )

        expected_score = 0.0
        worst_case = float("inf")

        for direction, probability in move_distribution.items():
            next_game, opp_alive, our_alive = self._simulate_action(
                game, opponent.agent_id, direction, False
            )

            if not our_alive and not opp_alive:
                value = 0.0
            elif not our_alive:
                value = -1000.0 - depth
            elif not opp_alive:
                value = 1000.0 + depth
            else:
                value = self._minimax_recursive(
                    next_game, depth - 1, alpha, beta, self.agent_id
                )

            expected_score += probability * value
            worst_case = min(worst_case, value)

        return self.expectation_blend * expected_score + (
            1.0 - self.expectation_blend
        ) * worst_case

    def _calculate_history_bias(self) -> Dict[Direction, float]:
        total = sum(self.opponent_move_counts.values())
        if total <= 0:
            return {}
        bias: Dict[Direction, float] = {}
        for direction, count in self.opponent_move_counts.items():
            frequency = count / total
            bias[direction] = max(frequency, 1e-3) ** self.history_influence
        return bias

    def _simulate_action(
        self,
        game: Game,
        acting_agent_id: int,
        direction: Direction,
        use_boost: bool,
    ) -> Tuple[Game, bool, bool]:
        game_copy = self._copy_game(game)
        acting_agent = self._get_agent(game_copy, acting_agent_id)
        other_agent = self._get_agent(game_copy, 1 if acting_agent_id == 2 else 2)

        acting_alive = acting_agent.move(
            direction, other_agent=other_agent, use_boost=use_boost
        )
        other_alive = other_agent.alive

        return game_copy, acting_alive, other_alive

    def _evaluate_state(self, game: Game) -> float:
        return evaluate_game_state(
            game,
            self.agent_id,
            self.aggressive_weight,
            self.exploration_weight,
            self.safety_weight,
        )

    def _evaluate_terminal(self, game: Game) -> float:
        agent = self._get_agent(game, self.agent_id)
        opponent = self._get_agent(game, self._opponent_id)
        if not agent.alive and not opponent.alive:
            return 0.0
        if not agent.alive:
            return -1000.0
        if not opponent.alive:
            return 1000.0
        return self._evaluate_state(game)

    def _get_agent(self, game: Game, agent_id: int) -> Agent:
        return game.agent1 if agent_id == 1 else game.agent2

    def _update_opponent_model(self, game: Game) -> None:
        opponent = self._get_agent(game, self._opponent_id)
        if len(opponent.trail) < 2:
            return
        prev = opponent.trail[-2]
        current = opponent.trail[-1]
        direction = self._infer_direction(game.board, prev, current)
        if direction is not None:
            self.opponent_move_counts[direction] += 1.0
            self.opponent_move_history.append(direction)

    @staticmethod
    def _infer_direction(
        board: GameBoard, previous: Tuple[int, int], current: Tuple[int, int]
    ) -> Optional[Direction]:
        for direction in Direction:
            expected = board._torus_check(
                (previous[0] + direction.value[0], previous[1] + direction.value[1])
            )
            if expected == current:
                return direction
        return None

    def _copy_game(self, game: Game) -> Game:
        """Create a deep copy of the game state for simulation."""
        new_board = GameBoard(height=game.board.height, width=game.board.width)
        new_board.grid = [row[:] for row in game.board.grid]

        new_game = Game(verbose=getattr(game, "verbose", False))
        new_game.board = new_board
        new_game.turns = game.turns

        if len(game.agent1.trail) > 0:
            start_pos = game.agent1.trail[0]
            if len(game.agent1.trail) > 1:
                second_pos = game.agent1.trail[1]
            else:
                dx, dy = game.agent1.direction.value
                second_pos = (start_pos[0] + dx, start_pos[1] + dy)
        else:
            start_pos = (1, 2)
            second_pos = (2, 2)

        new_game.agent1 = Agent(
            agent_id=1,
            start_pos=start_pos,
            start_dir=game.agent1.direction,
            board=new_game.board,
            verbose=getattr(game.agent1, "verbose", False),
        )
        new_game.agent1.trail = deque(list(game.agent1.trail))
        new_game.agent1.direction = game.agent1.direction
        new_game.agent1.alive = game.agent1.alive
        new_game.agent1.length = game.agent1.length
        new_game.agent1.boosts_remaining = game.agent1.boosts_remaining
        new_game.agent1.invalid_move_count = getattr(game.agent1, "invalid_move_count", 0)
        new_game.agent1.boosts_used = getattr(game.agent1, "boosts_used", 0)
        new_game.agent1.verbose = getattr(game.agent1, "verbose", False)
        new_game.board.grid = [row[:] for row in game.board.grid]

        if len(game.agent2.trail) > 0:
            start_pos2 = game.agent2.trail[0]
            if len(game.agent2.trail) > 1:
                second_pos2 = game.agent2.trail[1]
            else:
                dx, dy = game.agent2.direction.value
                second_pos2 = (start_pos2[0] + dx, start_pos2[1] + dy)
        else:
            start_pos2 = (17, 15)
            second_pos2 = (16, 15)

        new_game.agent2 = Agent(
            agent_id=2,
            start_pos=start_pos2,
            start_dir=game.agent2.direction,
            board=new_game.board,
            verbose=getattr(game.agent2, "verbose", False),
        )
        new_game.agent2.trail = deque(list(game.agent2.trail))
        new_game.agent2.direction = game.agent2.direction
        new_game.agent2.alive = game.agent2.alive
        new_game.agent2.length = game.agent2.length
        new_game.agent2.boosts_remaining = game.agent2.boosts_remaining
        new_game.agent2.invalid_move_count = getattr(game.agent2, "invalid_move_count", 0)
        new_game.agent2.boosts_used = getattr(game.agent2, "boosts_used", 0)
        new_game.agent2.verbose = getattr(game.agent2, "verbose", False)
        new_game.board.grid = [row[:] for row in game.board.grid]

        return new_game

    def get_stats(self) -> Dict[str, any]:
        stats = {
            "nodes_evaluated": self.nodes_evaluated,
            "max_depth_reached": self.max_depth_reached,
        }
        if self.use_transposition_table and self.transposition_table is not None:
            stats["transposition_table"] = self.transposition_table.get_stats()
        return stats
