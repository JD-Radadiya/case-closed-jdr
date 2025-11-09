"""
Minimax algorithm with alpha-beta pruning and transposition tables for Case Closed game.
"""

import hashlib
import json
from collections import deque
from typing import Optional, Tuple, Dict, List
from case_closed_game import Game, Direction, GameResult, Agent, GameBoard
from heuristics import evaluate_game_state, get_safe_moves


class TranspositionTable:
    """Transposition table for caching game state evaluations."""
    
    def __init__(self, max_size: int = 10000):
        self.table: Dict[str, Tuple[float, int]] = {}  # hash -> (score, depth)
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _hash_state(self, game: Game, agent_id: int) -> str:
        """
        Create a hash of the game state for caching.
        Includes board state, agent positions, directions, and boosts.
        """
        # Create a compact representation of the state
        state_data = {
            'board': [[cell for cell in row] for row in game.board.grid],
            'agent1_pos': list(game.agent1.trail),
            'agent2_pos': list(game.agent2.trail),
            'agent1_dir': game.agent1.direction.name,
            'agent2_dir': game.agent2.direction.name,
            'agent1_boosts': game.agent1.boosts_remaining,
            'agent2_boosts': game.agent2.boosts_remaining,
            'agent1_alive': game.agent1.alive,
            'agent2_alive': game.agent2.alive,
            'turn': game.turns,
            'agent_id': agent_id
        }
        
        state_str = json.dumps(state_data, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def get(self, game: Game, agent_id: int, depth: int) -> Optional[float]:
        """Get cached evaluation if available at sufficient depth."""
        state_hash = self._hash_state(game, agent_id)
        
        if state_hash in self.table:
            cached_score, cached_depth = self.table[state_hash]
            if cached_depth >= depth:
                self.hits += 1
                return cached_score
        
        self.misses += 1
        return None
    
    def store(self, game: Game, agent_id: int, score: float, depth: int):
        """Store evaluation in transposition table."""
        state_hash = self._hash_state(game, agent_id)
        
        # Evict old entries if table is full
        if len(self.table) >= self.max_size:
            # Remove a random entry (simple eviction policy)
            if self.table:
                first_key = next(iter(self.table))
                del self.table[first_key]
        
        self.table[state_hash] = (score, depth)
    
    def clear(self):
        """Clear the transposition table."""
        self.table.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about cache performance."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.table)
        }


class MinimaxAgent:
    """
    Minimax agent with alpha-beta pruning and transposition tables.
    """
    
    def __init__(self, 
                 agent_id: int = 1,
                 depth: int = 4,
                 aggressive_weight: float = 0.33,
                 exploration_weight: float = 0.33,
                 safety_weight: float = 0.34,
                 use_transposition_table: bool = True,
                 tt_max_size: int = 10000):
        self.agent_id = agent_id
        self.depth = depth
        self.aggressive_weight = aggressive_weight
        self.exploration_weight = exploration_weight
        self.safety_weight = safety_weight
        self.use_transposition_table = use_transposition_table
        self.transposition_table = TranspositionTable(tt_max_size) if use_transposition_table else None
        self.nodes_evaluated = 0
        self.max_depth_reached = 0
    
    def get_best_move(self, game: Game) -> Tuple[Direction, bool]:
        """
        Get the best move using minimax algorithm.
        
        Returns:
            Tuple of (direction, use_boost)
        """
        self.nodes_evaluated = 0
        self.max_depth_reached = 0
        
        if self.use_transposition_table:
            # Clear table for new search (or keep it for iterative deepening)
            pass
        
        agent = game.agent1 if self.agent_id == 1 else game.agent2
        safe_moves = get_safe_moves(game, agent)
        
        if not safe_moves:
            # No safe moves, return current direction
            return agent.direction, False
        
        best_move = None
        best_score = float('-inf')
        best_boost = False
        
        # Try each safe move
        moves_to_try = self._order_moves(game, agent, safe_moves)
        
        for direction in moves_to_try:
            # Try without boost first
            score = self._minimax(
                game, direction, False, self.depth,
                float('-inf'), float('inf'), True
            )
            
            if score > best_score:
                best_score = score
                best_move = direction
                best_boost = False
            
            # Try with boost if available
            if agent.boosts_remaining > 0:
                score_boost = self._minimax(
                    game, direction, True, self.depth,
                    float('-inf'), float('inf'), True
                )
                
                if score_boost > best_score:
                    best_score = score_boost
                    best_move = direction
                    best_boost = True
        
        if best_move is None:
            # Fallback to first safe move
            best_move = safe_moves[0] if safe_moves else agent.direction
        
        return best_move, best_boost
    
    def _order_moves(self, game: Game, agent: Agent, moves: List[Direction]) -> List[Direction]:
        """
        Order moves for better alpha-beta pruning.
        Prioritize moves that maintain current direction.
        """
        current_dir = agent.direction
        ordered = []
        others = []
        
        for move in moves:
            if move == current_dir:
                ordered.insert(0, move)  # Current direction first
            else:
                others.append(move)
        
        return ordered + others
    
    def _minimax(self, game: Game, move: Direction, use_boost: bool, 
                 depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            game: Current game state
            move: Move to make
            use_boost: Whether to use boost
            depth: Remaining depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player, False if minimizing
        
        Returns:
            Evaluation score
        """
        self.nodes_evaluated += 1
        self.max_depth_reached = max(self.max_depth_reached, self.depth - depth)
        
        # Create a copy of the game to simulate the move
        game_copy = self._copy_game(game)
        agent = game_copy.agent1 if self.agent_id == 1 else game_copy.agent2
        other_agent = game_copy.agent2 if self.agent_id == 1 else game_copy.agent1
        
        # Make the move
        agent_alive = agent.move(move, other_agent=other_agent, use_boost=use_boost)
        
        # Check terminal states
        if not agent_alive:
            if not other_agent.alive:
                return 0.0  # Draw
            else:
                return -1000.0 - depth  # Loss (prefer later losses)
        
        if not other_agent.alive:
            return 1000.0 + depth  # Win (prefer earlier wins)
        
        # Check transposition table
        if self.use_transposition_table and depth > 0:
            cached_score = self.transposition_table.get(game_copy, self.agent_id, depth)
            if cached_score is not None:
                return cached_score
        
        # Terminal node or depth limit reached
        if depth == 0:
            score = evaluate_game_state(
                game_copy, self.agent_id,
                self.aggressive_weight,
                self.exploration_weight,
                self.safety_weight
            )
            
            if self.use_transposition_table:
                self.transposition_table.store(game_copy, self.agent_id, score, depth)
            
            return score
        
        # Recursive case: In simultaneous-move game, we need to consider opponent's move
        # After making our move, the opponent will also make a move simultaneously
        # So we simulate opponent moves and then consider our next move
        if maximizing:
            # We just made our move, now consider opponent's simultaneous move
            max_score = float('-inf')
            opponent_safe_moves = get_safe_moves(game_copy, other_agent)
            
            if not opponent_safe_moves:
                # Opponent has no moves, we win
                score = evaluate_game_state(
                    game_copy, self.agent_id,
                    self.aggressive_weight,
                    self.exploration_weight,
                    self.safety_weight
                )
                return score + 500.0  # Bonus for winning position
            
            # Consider opponent's possible moves (they move simultaneously)
            for opp_move in opponent_safe_moves[:3]:  # Limit opponent moves for performance
                # Create a copy for this opponent move
                game_after_opp_move = self._copy_game(game_copy)
                opp_agent = game_after_opp_move.agent2 if self.agent_id == 1 else game_after_opp_move.agent1
                our_agent_after = game_after_opp_move.agent1 if self.agent_id == 1 else game_after_opp_move.agent2
                
                # Simulate opponent's move
                opp_alive = opp_agent.move(opp_move, other_agent=our_agent_after, use_boost=False)
                
                # Check if game ended
                if not our_agent_after.alive and not opp_agent.alive:
                    score = 0.0  # Draw
                elif not our_agent_after.alive:
                    score = -1000.0  # We lost
                elif not opp_agent.alive:
                    score = 1000.0  # We won
                elif depth > 1:
                    # Continue search: now it's our turn to move again
                    our_safe_moves = get_safe_moves(game_after_opp_move, our_agent_after)
                    if our_safe_moves:
                        # Try our next moves (it's our turn, so we maximize)
                        best_next_score = float('-inf')
                        for our_next_move in our_safe_moves[:2]:  # Limit for performance
                            # Create a copy for our next move
                            game_after_our_next = self._copy_game(game_after_opp_move)
                            our_agent_next = game_after_our_next.agent1 if self.agent_id == 1 else game_after_our_next.agent2
                            opp_agent_next = game_after_our_next.agent2 if self.agent_id == 1 else game_after_our_next.agent1
                            
                            # Make our next move
                            our_next_alive = our_agent_next.move(our_next_move, other_agent=opp_agent_next, use_boost=False)
                            
                            # Check terminal states
                            if not our_agent_next.alive and not opp_agent_next.alive:
                                next_score = 0.0
                            elif not our_agent_next.alive:
                                next_score = -1000.0 - (depth - 1)
                            elif not opp_agent_next.alive:
                                next_score = 1000.0 + (depth - 1)
                            elif depth > 2:
                                # Recurse: consider opponent's response
                                opp_next_safe = get_safe_moves(game_after_our_next, opp_agent_next)
                                if opp_next_safe:
                                    # Consider opponent's best response (minimizing for us)
                                    worst_opp_score = float('inf')
                                    for opp_next_move in opp_next_safe[:2]:
                                        opp_next_alive = opp_agent_next.move(opp_next_move, other_agent=our_agent_next, use_boost=False)
                                        # Evaluate or recurse further
                                        if depth > 3:
                                            # Too deep, just evaluate
                                            next_score_deep = evaluate_game_state(
                                                game_after_our_next, self.agent_id,
                                                self.aggressive_weight,
                                                self.exploration_weight,
                                                self.safety_weight
                                            )
                                        else:
                                            next_score_deep = evaluate_game_state(
                                                game_after_our_next, self.agent_id,
                                                self.aggressive_weight,
                                                self.exploration_weight,
                                                self.safety_weight
                                            )
                                        worst_opp_score = min(worst_opp_score, next_score_deep)
                                    next_score = worst_opp_score
                                else:
                                    # Opponent has no moves
                                    next_score = evaluate_game_state(
                                        game_after_our_next, self.agent_id,
                                        self.aggressive_weight,
                                        self.exploration_weight,
                                        self.safety_weight
                                    ) + 500.0
                            else:
                                # Depth limit, evaluate
                                next_score = evaluate_game_state(
                                    game_after_our_next, self.agent_id,
                                    self.aggressive_weight,
                                    self.exploration_weight,
                                    self.safety_weight
                                )
                            
                            best_next_score = max(best_next_score, next_score)
                        score = best_next_score
                    else:
                        # No moves, evaluate current state
                        score = evaluate_game_state(
                            game_after_opp_move, self.agent_id,
                            self.aggressive_weight,
                            self.exploration_weight,
                            self.safety_weight
                        ) - 500.0
                else:
                    # Depth limit reached, evaluate state
                    score = evaluate_game_state(
                        game_after_opp_move, self.agent_id,
                        self.aggressive_weight,
                        self.exploration_weight,
                        self.safety_weight
                    )
                
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            if self.use_transposition_table:
                self.transposition_table.store(game_copy, self.agent_id, max_score, depth)
            
            return max_score
        else:
            # This branch shouldn't be reached in current structure, but kept for safety
            # Evaluate current state
            score = evaluate_game_state(
                game_copy, self.agent_id,
                self.aggressive_weight,
                self.exploration_weight,
                self.safety_weight
            )
            return score
    
    def _copy_game(self, game: Game) -> Game:
        """
        Create a deep copy of the game state for simulation.
        This is expensive but necessary for minimax.
        """
        # Create new board with copied grid
        new_board = GameBoard(height=game.board.height, width=game.board.width)
        new_board.grid = [row[:] for row in game.board.grid]
        
        # Create new game object
        new_game = Game(verbose=getattr(game, 'verbose', False))
        new_game.board = new_board
        new_game.turns = game.turns
        
        # Copy agent 1 - get start position from trail
        if len(game.agent1.trail) > 0:
            start_pos = game.agent1.trail[0]
            # Get second position or calculate it
            if len(game.agent1.trail) > 1:
                second_pos = game.agent1.trail[1]
            else:
                # Calculate second position from direction
                dx, dy = game.agent1.direction.value
                second_pos = (start_pos[0] + dx, start_pos[1] + dy)
        else:
            start_pos = (1, 2)
            second_pos = (2, 2)
        
        # Create agent manually to avoid modifying board during init
        # We'll create it with dummy positions first, then restore the trail
        new_game.agent1 = Agent(agent_id=1,
                               start_pos=start_pos,
                               start_dir=game.agent1.direction,
                               board=new_game.board,
                               verbose=getattr(game.agent1, 'verbose', False))
        # Restore the actual trail and state
        new_game.agent1.trail = deque(list(game.agent1.trail))
        new_game.agent1.direction = game.agent1.direction
        new_game.agent1.alive = game.agent1.alive
        new_game.agent1.length = game.agent1.length
        new_game.agent1.boosts_remaining = game.agent1.boosts_remaining
        new_game.agent1.invalid_move_count = getattr(game.agent1, 'invalid_move_count', 0)
        new_game.agent1.boosts_used = getattr(game.agent1, 'boosts_used', 0)
        new_game.agent1.verbose = getattr(game.agent1, 'verbose', False)
        # Restore board state (Agent.__init__ may have modified it)
        new_game.board.grid = [row[:] for row in game.board.grid]
        
        # Copy agent 2
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
        
        new_game.agent2 = Agent(agent_id=2,
                               start_pos=start_pos2,
                               start_dir=game.agent2.direction,
                               board=new_game.board,
                               verbose=getattr(game.agent2, 'verbose', False))
        # Restore the actual trail and state
        new_game.agent2.trail = deque(list(game.agent2.trail))
        new_game.agent2.direction = game.agent2.direction
        new_game.agent2.alive = game.agent2.alive
        new_game.agent2.length = game.agent2.length
        new_game.agent2.boosts_remaining = game.agent2.boosts_remaining
        new_game.agent2.invalid_move_count = getattr(game.agent2, 'invalid_move_count', 0)
        new_game.agent2.boosts_used = getattr(game.agent2, 'boosts_used', 0)
        new_game.agent2.verbose = getattr(game.agent2, 'verbose', False)
        # Restore board state again (Agent.__init__ modifies it)
        new_game.board.grid = [row[:] for row in game.board.grid]
        
        return new_game
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the last search."""
        stats = {
            'nodes_evaluated': self.nodes_evaluated,
            'max_depth_reached': self.max_depth_reached,
        }
        
        if self.use_transposition_table:
            stats['transposition_table'] = self.transposition_table.get_stats()
        
        return stats

