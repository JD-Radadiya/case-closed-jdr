"""Baseline agents used for evaluation during training."""

import random
from collections import deque
from typing import Tuple

from case_closed_game import Game, Direction, Agent
from heuristics import evaluate_game_state, get_safe_moves


class RandomAgent:
    """Agent that selects uniformly random safe moves."""

    def __init__(self, agent_id: int = 1):
        self.agent_id = agent_id

    def get_best_move(self, game: Game) -> Tuple[Direction, bool]:
        agent = game.agent1 if self.agent_id == 1 else game.agent2
        safe_moves = get_safe_moves(game, agent)
        if not safe_moves:
            return agent.direction, False
        return random.choice(safe_moves), False

    def get_stats(self) -> dict:
        return {}


class HeuristicAgent:
    """Agent that greedily maximizes the heuristic evaluation score."""

    def __init__(self,
                 agent_id: int = 1,
                 aggressive_weight: float = 0.33,
                 exploration_weight: float = 0.33,
                 safety_weight: float = 0.34):
        self.agent_id = agent_id
        self.aggressive_weight = aggressive_weight
        self.exploration_weight = exploration_weight
        self.safety_weight = safety_weight

    def get_best_move(self, game: Game) -> Tuple[Direction, bool]:
        agent = game.agent1 if self.agent_id == 1 else game.agent2
        safe_moves = get_safe_moves(game, agent)
        if not safe_moves:
            return agent.direction, False

        best_direction = None
        best_score = float('-inf')
        for direction in safe_moves:
            simulated_game = _simulate_one_ply(game, self.agent_id, direction)
            score = evaluate_game_state(
                simulated_game,
                self.agent_id,
                aggressive_weight=self.aggressive_weight,
                exploration_weight=self.exploration_weight,
                safety_weight=self.safety_weight,
            )
            if score > best_score:
                best_score = score
                best_direction = direction

        return best_direction or safe_moves[0], False

    def get_stats(self) -> dict:
        return {}


def _simulate_one_ply(game: Game, agent_id: int, direction: Direction) -> Game:
    """Return a shallow copy of the game after applying the given move."""
    sim_game = _copy_game(game)
    opponent = sim_game.agent2 if agent_id == 1 else sim_game.agent1

    if agent_id == 1:
        sim_game.step(direction, opponent.direction)
    else:
        sim_game.step(opponent.direction, direction)

    return sim_game


def _copy_game(game: Game) -> Game:
    """Create a deep copy of the current game state."""
    new_game = Game(verbose=getattr(game, 'verbose', False))
    new_game.board = type(game.board)(height=game.board.height, width=game.board.width)
    new_game.board.grid = [row[:] for row in game.board.grid]
    new_game.turns = game.turns

    def _clone_agent(src: Agent) -> Agent:
        cloned = Agent(agent_id=src.agent_id,
                       start_pos=src.trail[0],
                       start_dir=src.direction,
                       board=new_game.board,
                       verbose=getattr(src, 'verbose', False))
        cloned.trail = deque(list(src.trail))
        cloned.direction = src.direction
        cloned.alive = src.alive
        cloned.length = src.length
        cloned.boosts_remaining = src.boosts_remaining
        cloned.invalid_move_count = getattr(src, 'invalid_move_count', 0)
        cloned.boosts_used = getattr(src, 'boosts_used', 0)
        cloned.verbose = getattr(src, 'verbose', False)
        return cloned

    new_game.agent1 = _clone_agent(game.agent1)
    new_game.agent2 = _clone_agent(game.agent2)

    return new_game
