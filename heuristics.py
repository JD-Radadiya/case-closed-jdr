"""Heuristic helpers used by search agents."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

from case_closed_game import AGENT, EMPTY, Agent, Direction, Game, GameBoard


def get_accessible_cells(game: Game, agent: Agent, max_depth: int = 10) -> Set[Tuple[int, int]]:
    """
    Calculate accessible cells from agent's current position using BFS.
    Returns set of reachable empty cells within max_depth moves.
    """
    accessible = set()
    head = agent.trail[-1]
    queue = deque([(head, 0)])
    visited = {head}
    
    while queue:
        pos, depth = queue.popleft()
        if depth >= max_depth:
            continue
            
        accessible.add(pos)
        
        # Check all 4 directions
        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            dx, dy = direction.value
            new_pos = game.board._torus_check((pos[0] + dx, pos[1] + dy))
            
            if new_pos not in visited:
                cell_state = game.board.get_cell_state(new_pos)
                if cell_state == EMPTY:
                    visited.add(new_pos)
                    queue.append((new_pos, depth + 1))
    
    return accessible


def get_safe_moves(game: Game, agent: Agent) -> List[Direction]:
    """
    Get list of safe moves (moves that don't immediately cause collision).
    """
    safe_moves = []
    head = agent.trail[-1]
    current_dir = agent.direction
    
    for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
        # Can't move opposite to current direction
        cur_dx, cur_dy = current_dir.value
        req_dx, req_dy = direction.value
        if (req_dx, req_dy) == (-cur_dx, -cur_dy):
            continue
        
        # Check if move is safe
        dx, dy = direction.value
        new_pos = game.board._torus_check((head[0] + dx, head[1] + dy))
        cell_state = game.board.get_cell_state(new_pos)
        
        if cell_state == EMPTY:
            safe_moves.append(direction)
        elif cell_state == AGENT:
            # Check if it's opponent's head (head-on collision)
            other_agent = game.agent2 if agent.agent_id == 1 else game.agent1
            if other_agent.alive and other_agent.is_head(new_pos):
                # Head-on collision - risky but sometimes acceptable
                safe_moves.append(direction)
    
    return safe_moves


def _evaluate_local_escape_space(
    game: Game,
    start_pos: Tuple[int, int],
    max_depth: int = 3,
) -> int:
    """Estimate how many empty cells surround ``start_pos`` within ``max_depth`` moves."""

    queue: deque[Tuple[Tuple[int, int], int]] = deque([(start_pos, 0)])
    visited = {start_pos}
    reachable = 0

    while queue:
        pos, depth = queue.popleft()
        if depth >= max_depth:
            continue

        for direction in (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT):
            dx, dy = direction.value
            next_pos = game.board._torus_check((pos[0] + dx, pos[1] + dy))
            if next_pos in visited:
                continue
            visited.add(next_pos)
            cell_state = game.board.get_cell_state(next_pos)
            if cell_state == EMPTY:
                reachable += 1
                queue.append((next_pos, depth + 1))

    return reachable


def calculate_manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int],
                                 board_width: int, board_height: int) -> int:
    """
    Calculate Manhattan distance with torus wrapping.
    """
    x1, y1 = pos1
    x2, y2 = pos2
    
    # Consider wraparound in both directions
    dx = min(abs(x2 - x1), board_width - abs(x2 - x1))
    dy = min(abs(y2 - y1), board_height - abs(y2 - y1))
    
    return dx + dy


def evaluate_territory_control(game: Game, agent: Agent, max_depth: int = 8) -> float:
    """
    Evaluate territory control: ratio of accessible cells to total empty cells.
    Higher is better.
    """
    accessible = get_accessible_cells(game, agent, max_depth)
    total_empty = sum(1 for y in range(game.board.height) 
                     for x in range(game.board.width) 
                     if game.board.get_cell_state((x, y)) == EMPTY)
    
    if total_empty == 0:
        return 0.0
    
    return len(accessible) / total_empty


def evaluate_survival_probability(game: Game, agent: Agent) -> float:
    """
    Evaluate survival probability: ratio of safe moves to total possible moves.
    Higher is better.
    """
    safe_moves = get_safe_moves(game, agent)
    total_moves = 3  # Can move in 3 directions (excluding opposite)
    
    return len(safe_moves) / total_moves if total_moves > 0 else 0.0


def evaluate_distance_to_opponent(game: Game, agent: Agent) -> float:
    """
    Evaluate distance to opponent. For aggressive play, closer is better.
    For safety, farther is better. Returns normalized distance (0-1).
    """
    other_agent = game.agent2 if agent.agent_id == 1 else game.agent1
    if not other_agent.alive:
        return 1.0  # Opponent is dead, maximum advantage
    
    head1 = agent.trail[-1]
    head2 = other_agent.trail[-1]
    
    distance = calculate_manhattan_distance(head1, head2, 
                                           game.board.width, game.board.height)
    
    # Normalize: max distance is width + height (with wrapping, it's actually less)
    max_distance = (game.board.width + game.board.height) // 2
    normalized = distance / max_distance if max_distance > 0 else 0.0
    
    return min(normalized, 1.0)


def evaluate_trail_length_advantage(game: Game, agent: Agent) -> float:
    """
    Evaluate trail length advantage relative to opponent.
    Returns value in range [-1, 1], where 1.0 means much longer trail.
    """
    other_agent = game.agent2 if agent.agent_id == 1 else game.agent1
    
    if not other_agent.alive:
        return 1.0  # Opponent is dead
    
    length_diff = agent.length - other_agent.length
    total_length = agent.length + other_agent.length
    
    if total_length == 0:
        return 0.0
    
    # Normalize to [-1, 1]
    return (2 * length_diff) / total_length


def evaluate_aggressive_position(game: Game, agent: Agent) -> float:
    """
    Evaluate aggressive positioning: how well positioned to corner opponent.
    Considers distance (closer is better) and ability to cut off opponent's escape routes.
    """
    other_agent = game.agent2 if agent.agent_id == 1 else game.agent1
    if not other_agent.alive:
        return 1.0
    
    # Closer to opponent is better for aggression
    distance_score = 1.0 - evaluate_distance_to_opponent(game, agent)
    
    # Check if we can cut off opponent's escape routes
    opponent_safe_moves = get_safe_moves(game, other_agent)
    cutting_score = 1.0 - (len(opponent_safe_moves) / 3.0) if len(opponent_safe_moves) > 0 else 1.0
    
    return (distance_score * 0.6 + cutting_score * 0.4)


def evaluate_exploration_value(game: Game, agent: Agent) -> float:
    """
    Evaluate exploration value: how much unexplored territory is accessible.
    Higher is better.
    """
    accessible = get_accessible_cells(game, agent, max_depth=12)
    
    # Prefer areas with more open space
    # Count how many empty cells are in a larger radius
    head = agent.trail[-1]
    exploration_score = len(accessible) / (game.board.width * game.board.height)
    
    return exploration_score


def evaluate_game_state(game: Game, agent_id: int,
                       aggressive_weight: float = 0.33,
                       exploration_weight: float = 0.33,
                       safety_weight: float = 0.34) -> float:
    """
    Main evaluation function that combines all heuristics.
    
    Args:
        game: Current game state
        agent_id: ID of agent to evaluate (1 or 2)
        aggressive_weight: Weight for aggressive play (0.0-1.0)
        exploration_weight: Weight for exploration (0.0-1.0)
        safety_weight: Weight for safety (0.0-1.0)
    
    Returns:
        Evaluation score (higher is better for the agent)
    """
    # Normalize weights to sum to 1.0
    total_weight = aggressive_weight + exploration_weight + safety_weight
    if total_weight > 0:
        aggressive_weight /= total_weight
        exploration_weight /= total_weight
        safety_weight /= total_weight
    
    agent = game.agent1 if agent_id == 1 else game.agent2
    other_agent = game.agent2 if agent_id == 1 else game.agent1
    
    # Terminal states
    if not agent.alive and not other_agent.alive:
        return 0.0  # Draw
    elif not agent.alive:
        return -1000.0  # Loss
    elif not other_agent.alive:
        return 1000.0  # Win
    
    # Base evaluation components
    territory = evaluate_territory_control(game, agent)
    survival = evaluate_survival_probability(game, agent)
    trail_advantage = evaluate_trail_length_advantage(game, agent)
    
    # Strategy-specific evaluations
    aggressive_score = evaluate_aggressive_position(game, agent)
    exploration_score = evaluate_exploration_value(game, agent)
    safety_score = survival  # Safety is primarily about survival
    
    # Combine strategy scores
    strategy_score = (
        aggressive_weight * aggressive_score +
        exploration_weight * exploration_score +
        safety_weight * safety_score
    )
    
    # Combine all components
    # Base score (always important)
    base_score = (
        territory * 0.3 +
        survival * 0.3 +
        trail_advantage * 0.4
    )
    
    # Strategy score (weighted by strategy parameters)
    final_score = base_score * 0.7 + strategy_score * 0.3
    
    return final_score


def estimate_opponent_move_probabilities(
    game: Game,
    our_agent: Agent,
    opponent: Agent,
    safe_moves: Iterable[Direction],
    history_bias: Optional[Dict[Direction, float]] = None,
) -> Dict[Direction, float]:
    """Estimate probabilities for the opponent's safe moves.

    The heuristic blends how much free space a move grants, how aggressively it
    points towards our agent, and how frequently we have observed the opponent
    using each direction historically.
    """

    moves = list(safe_moves)
    if not moves:
        return {}

    head = opponent.trail[-1]
    our_head = our_agent.trail[-1]
    bias = history_bias or {}

    width = game.board.width
    height = game.board.height

    scores: Dict[Direction, float] = {}
    for move in moves:
        dx, dy = move.value
        next_pos = game.board._torus_check((head[0] + dx, head[1] + dy))

        # Encourage moves that keep options open.
        local_space = _evaluate_local_escape_space(game, next_pos, max_depth=3)
        space_score = local_space / max(1.0, (width * height) / 8.0)

        # Aggression towards our agent – closer distance increases pressure.
        distance = calculate_manhattan_distance(next_pos, our_head, width, height)
        proximity_score = 1.0 / (1.0 + distance)

        # Prefer continuing current momentum slightly.
        momentum_bonus = 1.2 if move == opponent.direction else 1.0

        # Historical bias acts as a multiplicative prior.
        prior = bias.get(move, 1.0)

        raw_score = (0.5 * space_score) + (0.3 * momentum_bonus) + (0.2 * proximity_score)
        scores[move] = max(raw_score, 1e-5) * prior

    total = sum(scores.values())
    if total <= 0:
        uniform = 1.0 / len(moves)
        return {move: uniform for move in moves}

    return {move: scores[move] / total for move in moves}

