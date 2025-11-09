"""
Monte Carlo Tree Search (MCTS) agent with time-budgeted simulations.
"""

import time
import random
from typing import Tuple, Optional, Dict
from case_closed_game import Game, Direction, Agent, GameResult
from mcts_node import MCTSNode
from heuristics import get_safe_moves


class MCTSAgent:
    """
    MCTS agent that uses time-budgeted simulations to find the best move.
    """
    
    def __init__(self,
                 agent_id: int = 1,
                 simulation_time_ms: int = 120,
                 aggressive_weight: float = 0.33,
                 exploration_weight: float = 0.33,
                 safety_weight: float = 0.34,
                 exploration_constant: float = 1.414):
        self.agent_id = agent_id
        self.simulation_time_ms = simulation_time_ms
        self.aggressive_weight = aggressive_weight
        self.exploration_weight = exploration_weight
        self.safety_weight = safety_weight
        self.exploration_constant = exploration_constant
        self.simulations_run = 0
        self.last_search_time = 0.0
    
    def get_best_move(self, game: Game) -> Tuple[Direction, bool]:
        """
        Get the best move using MCTS with time-budgeted simulations.
        
        Returns:
            Tuple of (direction, use_boost)
        """
        start_time = time.time()
        time_budget = self.simulation_time_ms / 1000.0  # Convert to seconds
        self.simulations_run = 0
        
        # Create root node
        root = MCTSNode(game, agent_id=self.agent_id)
        agent = game.agent1 if self.agent_id == 1 else game.agent2
        root.untried_moves = root.get_untried_moves(agent)
        
        # Initialize root visits to avoid division by zero
        root.visits = 1
        
        # If no moves available, return current direction
        if not root.untried_moves:
            safe_moves = get_safe_moves(game, agent)
            if safe_moves:
                return safe_moves[0], False
            return agent.direction, False
        
        # MCTS loop with time budget
        while (time.time() - start_time) < time_budget:
            # Selection: traverse from root to leaf (this also handles expansion)
            node = self._select(root)
            
            # If node is terminal, evaluate it directly
            if node.is_terminal():
                reward = self._calculate_reward(node.game_state, node.agent_id)
            else:
                # Simulation: play random game from this node
                reward = self._simulate(node)
            
            # Backpropagation: update statistics up the tree
            self._backpropagate(node, reward)
            
            self.simulations_run += 1
        
        self.last_search_time = time.time() - start_time
        
        # Get best move from root
        best_move, best_boost = root.get_best_move()
        
        if best_move is None:
            # Fallback: choose most visited child or first safe move
            if root.children:
                best_child = max(root.children, key=lambda c: c.visits)
                best_move = best_child.move
                best_boost = best_child.use_boost
            else:
                safe_moves = get_safe_moves(game, agent)
                best_move = safe_moves[0] if safe_moves else agent.direction
                best_boost = False
        
        return best_move, best_boost
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a node using UCB1, expanding if necessary.
        """
        while not node.is_terminal():
            if not node.is_fully_expanded():
                expanded_node = node.expand()
                # If expansion returned the same node, it means no moves were available
                if expanded_node == node:
                    # Terminal state or no moves available
                    return node
                return expanded_node
            else:
                # Node is fully expanded, select a child
                selected_child = node.select_child(self.exploration_constant)
                # If select_child returned the node itself, it means no children available
                if selected_child == node:
                    # This shouldn't happen, but handle it gracefully
                    return node
                node = selected_child
        return node
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate a random game from the given node to a terminal state.
        Returns reward from the perspective of the agent.
        Note: In this game, both agents move simultaneously, so we simulate both moves each turn.
        """
        # Start from the node's game state
        game_state = self._copy_game(node.game_state)
        
        # Play random game until terminal
        max_turns = 30  # Limit simulation depth for performance
        turn_count = 0
        
        while not self._is_terminal(game_state) and turn_count < max_turns:
            # Both agents move simultaneously
            agent1 = game_state.agent1
            agent2 = game_state.agent2
            
            # Get safe moves for both agents
            safe_moves1 = get_safe_moves(game_state, agent1)
            safe_moves2 = get_safe_moves(game_state, agent2)
            
            if not safe_moves1 or not safe_moves2:
                break
            
            # Choose moves for both agents
            # Our agent uses strategy-aware selection
            if node.agent_id == 1:
                dir1 = self._choose_simulation_move(game_state, agent1, safe_moves1)
                dir2 = random.choice(safe_moves2)  # Opponent plays randomly
            else:
                dir1 = random.choice(safe_moves1)  # Opponent plays randomly
                dir2 = self._choose_simulation_move(game_state, agent2, safe_moves2)
            
            # Decide on boosts (rarely in simulation)
            boost1 = agent1.boosts_remaining > 0 and random.random() < 0.05
            boost2 = agent2.boosts_remaining > 0 and random.random() < 0.05
            
            # Execute moves simultaneously
            result = game_state.step(dir1, dir2, boost1, boost2)
            if result is not None:
                break  # Game ended
            
            turn_count += 1
        
        # Calculate reward
        return self._calculate_reward(game_state, node.agent_id)
    
    def _copy_game(self, game: Game) -> Game:
        """Create a deep copy of the game state."""
        from collections import deque
        
        new_game = Game()
        new_game.board = type(game.board)(height=game.board.height, width=game.board.width)
        new_game.board.grid = [row[:] for row in game.board.grid]
        new_game.turns = game.turns
        
        # Copy agent 1
        agent1_orig = game.agent1
        new_game.agent1 = Agent(agent_id=1,
                               start_pos=agent1_orig.trail[0],
                               start_dir=agent1_orig.direction,
                               board=new_game.board)
        new_game.agent1.trail = deque(list(agent1_orig.trail))
        new_game.agent1.direction = agent1_orig.direction
        new_game.agent1.alive = agent1_orig.alive
        new_game.agent1.length = agent1_orig.length
        new_game.agent1.boosts_remaining = agent1_orig.boosts_remaining
        
        # Copy agent 2
        agent2_orig = game.agent2
        new_game.agent2 = Agent(agent_id=2,
                               start_pos=agent2_orig.trail[0],
                               start_dir=agent2_orig.direction,
                               board=new_game.board)
        new_game.agent2.trail = deque(list(agent2_orig.trail))
        new_game.agent2.direction = agent2_orig.direction
        new_game.agent2.alive = agent2_orig.alive
        new_game.agent2.length = agent2_orig.length
        new_game.agent2.boosts_remaining = agent2_orig.boosts_remaining
        
        return new_game
    
    def _choose_simulation_move(self, game: Game, agent: Agent, safe_moves: list) -> Direction:
        """
        Choose a move during simulation based on strategy parameters.
        """
        if not safe_moves:
            return agent.direction
        
        # Weight moves based on strategy
        move_scores = []
        for move in safe_moves:
            score = 0.0
            
            # Safety: prefer moves that maintain multiple escape routes
            if self.safety_weight > 0:
                # Simple heuristic: prefer continuing in current direction
                if move == agent.direction:
                    score += self.safety_weight * 0.5
            
            # Exploration: prefer moves toward unexplored areas
            if self.exploration_weight > 0:
                # Random component for exploration
                score += self.exploration_weight * random.random() * 0.3
            
            # Aggressive: prefer moves toward opponent
            if self.aggressive_weight > 0:
                other_agent = game.agent2 if agent.agent_id == 1 else game.agent1
                if other_agent.alive:
                    # Simple: prefer moves that don't move away (aggressive)
                    score += self.aggressive_weight * random.random() * 0.2
            
            move_scores.append((score, move))
        
        # Choose move with probability proportional to score
        if move_scores:
            move_scores.sort(reverse=True)
            # Use softmax-like selection: prefer higher scores but allow randomness
            total_score = sum(score for score, _ in move_scores[:3])  # Top 3 moves
            if total_score > 0:
                r = random.random() * total_score
                cumulative = 0
                for score, move in move_scores[:3]:
                    cumulative += score
                    if r <= cumulative:
                        return move
            
            # Fallback: return first move
            return move_scores[0][1]
        
        return safe_moves[0]
    
    def _is_terminal(self, game: Game) -> bool:
        """Check if game is in terminal state."""
        return (not game.agent1.alive or not game.agent2.alive or 
                game.turns >= 200)
    
    def _calculate_reward(self, game: Game, agent_id: int) -> float:
        """
        Calculate reward from simulation outcome.
        Returns reward from perspective of agent_id.
        """
        agent = game.agent1 if agent_id == 1 else game.agent2
        other_agent = game.agent2 if agent_id == 1 else game.agent1
        
        # Terminal rewards
        if not agent.alive and not other_agent.alive:
            return 0.0  # Draw
        elif not agent.alive:
            return -1.0  # Loss
        elif not other_agent.alive:
            return 1.0  # Win
        
        # Non-terminal: use heuristics as reward shaping
        # Trail length advantage
        length_diff = agent.length - other_agent.length
        length_reward = length_diff / 100.0  # Normalize
        
        # Survival bonus
        survival_reward = 0.1 if agent.alive else -1.0
        
        # Combine rewards
        return 0.7 * survival_reward + 0.3 * length_reward
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagate reward up the tree.
        Reward is always from perspective of our agent (self.agent_id).
        """
        while node is not None:
            node.update(reward)
            node = node.parent
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the last search."""
        return {
            'simulations_run': self.simulations_run,
            'search_time_ms': self.last_search_time * 1000,
            'simulations_per_second': self.simulations_run / self.last_search_time if self.last_search_time > 0 else 0
        }

