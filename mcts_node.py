"""
MCTS node structure for Monte Carlo Tree Search.
"""

import math
from typing import List, Optional, Dict
from case_closed_game import Direction, Game, Agent, GameBoard


class MCTSNode:
    """
    Node in the MCTS tree.
    Each node represents a game state after a move.
    """
    
    def __init__(self, game_state: Game, move: Optional[Direction] = None, 
                 use_boost: bool = False, parent: Optional['MCTSNode'] = None,
                 agent_id: int = 1):
        self.game_state = game_state
        self.move = move  # Move that led to this state
        self.use_boost = use_boost
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.agent_id = agent_id  # Which agent we're optimizing for
        
        # MCTS statistics
        self.visits = 0
        self.wins = 0.0  # Accumulated rewards
        self.untried_moves: List[tuple] = []  # List of (Direction, use_boost) tuples
    
    def is_fully_expanded(self) -> bool:
        """
        Check if all possible moves have been tried.
        A node is fully expanded if it has no untried moves OR if it's a terminal state.
        """
        # Terminal nodes are considered "fully expanded"
        if self.is_terminal():
            return True
        # Non-terminal nodes are fully expanded if no untried moves
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state (game over)."""
        agent = self.game_state.agent1 if self.agent_id == 1 else self.game_state.agent2
        other_agent = self.game_state.agent2 if self.agent_id == 1 else self.game_state.agent1
        return not agent.alive or not other_agent.alive or self.game_state.turns >= 200
    
    def get_untried_moves(self, agent: Agent) -> List[tuple]:
        """
        Get list of untried moves for the agent.
        Returns list of (Direction, use_boost) tuples.
        """
        from heuristics import get_safe_moves
        
        moves = []
        safe_directions = get_safe_moves(self.game_state, agent)
        
        for direction in safe_directions:
            moves.append((direction, False))
            if agent.boosts_remaining > 0:
                moves.append((direction, True))
        
        return moves
    
    def select_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """
        Select a child node using UCB1 formula.
        UCB1 = (wins / visits) + c * sqrt(ln(parent_visits) / visits)
        """
        if not self.children:
            # No children available - this shouldn't happen if node is fully expanded
            # Return self to indicate we can't select a child
            return self
        
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                # Prefer unvisited children
                return child
            
            # UCB1 formula - handle case where parent has no visits yet
            if self.visits == 0:
                # If parent has no visits, just use exploitation
                ucb1_score = child.wins / child.visits if child.visits > 0 else 0.0
            else:
                exploitation = child.wins / child.visits
                exploration = exploration_constant * math.sqrt(
                    math.log(self.visits) / child.visits
                )
                ucb1_score = exploitation + exploration
            
            if ucb1_score > best_score:
                best_score = ucb1_score
                best_child = child
        
        # Return best child, or first child if somehow best_child is None
        return best_child if best_child is not None else self.children[0]
    
    def expand(self) -> 'MCTSNode':
        """
        Expand this node by adding a new child from an untried move.
        In this game, both agents move simultaneously, so we need to simulate
        our move along with a reasonable opponent move.
        """
        if not self.untried_moves:
            # No moves to try - return self (terminal or no valid moves)
            return self
        
        # Try to expand with an untried move
        # Keep trying until we find a valid move or run out of moves
        while self.untried_moves:
            # Get a move to try
            move, use_boost = self.untried_moves.pop()
            
            try:
                # Create new game state by simulating the move
                # We'll simulate our move and make the opponent move randomly/reasonably
                new_game_state = self._simulate_move_pair(self.game_state, move, use_boost, self.agent_id)
                
                # Check if the resulting state is valid
                agent = new_game_state.agent1 if self.agent_id == 1 else new_game_state.agent2
                if not agent.alive:
                    # Move led to death, skip this move
                    continue
                
                # Create new child node (same agent_id as parent, since we're optimizing for one agent)
                child = MCTSNode(new_game_state, move=move, use_boost=use_boost, parent=self, agent_id=self.agent_id)
                
                # Initialize child's untried moves
                child.untried_moves = child.get_untried_moves(agent)
                
                self.children.append(child)
                
                return child
            except Exception as e:
                # If expansion failed, try next move
                # In production, you might want to log this
                continue
        
        # All moves failed or were invalid - return self
        return self
    
    def update(self, reward: float):
        """
        Update node statistics after a simulation.
        """
        self.visits += 1
        self.wins += reward
    
    def _simulate_move_pair(self, game: Game, our_move: Direction, our_boost: bool, agent_id: int) -> Game:
        """
        Simulate a move pair (our move + opponent's move) and return a new game state.
        In this game, both agents move simultaneously.
        """
        import random
        from collections import deque
        from heuristics import get_safe_moves
        
        # Create a deep copy of the game board
        new_board = GameBoard(height=game.board.height, width=game.board.width)
        new_board.grid = [row[:] for row in game.board.grid]
        
        # Create new game object
        new_game = Game()
        new_game.board = new_board
        new_game.turns = game.turns
        
        # Copy agent 1 - get start position from trail
        if len(game.agent1.trail) > 0:
            start_pos1 = game.agent1.trail[0]
        else:
            start_pos1 = (1, 2)
        
        # Create agent 1 (this will modify board, so we'll restore it)
        new_game.agent1 = Agent(agent_id=1,
                               start_pos=start_pos1,
                               start_dir=game.agent1.direction,
                               board=new_game.board)
        # Restore agent 1 state
        new_game.agent1.trail = deque(list(game.agent1.trail))
        new_game.agent1.direction = game.agent1.direction
        new_game.agent1.alive = game.agent1.alive
        new_game.agent1.length = game.agent1.length
        new_game.agent1.boosts_remaining = game.agent1.boosts_remaining
        # Restore board state (Agent.__init__ modifies it)
        new_game.board.grid = [row[:] for row in game.board.grid]
        
        # Copy agent 2
        if len(game.agent2.trail) > 0:
            start_pos2 = game.agent2.trail[0]
        else:
            start_pos2 = (17, 15)
        
        # Create agent 2
        new_game.agent2 = Agent(agent_id=2,
                               start_pos=start_pos2,
                               start_dir=game.agent2.direction,
                               board=new_game.board)
        # Restore agent 2 state
        new_game.agent2.trail = deque(list(game.agent2.trail))
        new_game.agent2.direction = game.agent2.direction
        new_game.agent2.alive = game.agent2.alive
        new_game.agent2.length = game.agent2.length
        new_game.agent2.boosts_remaining = game.agent2.boosts_remaining
        # Restore board state again (Agent.__init__ modifies it)
        new_game.board.grid = [row[:] for row in game.board.grid]
        
        # Determine moves for both agents
        our_agent = new_game.agent1 if agent_id == 1 else new_game.agent2
        opp_agent = new_game.agent2 if agent_id == 1 else new_game.agent1
        
        # Our move is given
        our_dir = our_move
        our_boost_flag = our_boost
        
        # Opponent makes a reasonable move (random from safe moves)
        opp_safe_moves = get_safe_moves(new_game, opp_agent)
        if opp_safe_moves:
            opp_dir = random.choice(opp_safe_moves)
            opp_boost_flag = opp_agent.boosts_remaining > 0 and random.random() < 0.1
        else:
            # Opponent has no safe moves, use current direction
            opp_dir = opp_agent.direction
            opp_boost_flag = False
        
        # Execute moves simultaneously using game.step
        if agent_id == 1:
            new_game.step(our_dir, opp_dir, our_boost_flag, opp_boost_flag)
        else:
            new_game.step(opp_dir, our_dir, opp_boost_flag, our_boost_flag)
        
        return new_game
    
    def get_best_move(self) -> tuple:
        """
        Get the best move based on visit counts.
        Returns (Direction, use_boost) tuple.
        """
        if not self.children:
            return None, False
        
        best_child = max(self.children, key=lambda c: c.visits)
        return best_child.move, best_child.use_boost

