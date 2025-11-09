"""
Local match runner for training without Flask servers.
Runs matches between two agents directly.
"""

from typing import Tuple, Optional, Dict, Any, Callable
from case_closed_game import Game, Direction, GameResult
from collections import defaultdict
from minimax_agent import MinimaxAgent
from mcts_agent import MCTSAgent
from baseline_agents import RandomAgent, HeuristicAgent


class MatchRunner:
    """
    Runs matches between two agents locally without Flask servers.
    """
    
    def __init__(self, max_turns: int = 200, verbose: bool = False):
        self.max_turns = max_turns
        self.verbose = verbose
    
    def run_match(self, agent1_config: Dict[str, Any], agent2_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single match between two agents.
        
        Args:
            agent1_config: Configuration for agent 1
                - 'type': 'minimax' or 'mcts'
                - 'agent_id': 1
                - 'aggressive_weight': float
                - 'exploration_weight': float
                - 'safety_weight': float
                - 'depth': int (for minimax)
                - 'simulation_time_ms': int (for mcts)
            agent2_config: Configuration for agent 2 (same structure, agent_id: 2)
        
        Returns:
            Dictionary with match results:
            - 'result': GameResult
            - 'turns': int
            - 'agent1_length': int
            - 'agent2_length': int
            - 'agent1_alive': bool
            - 'agent2_alive': bool
            - 'agent1_stats': dict (algorithm-specific stats)
            - 'agent2_stats': dict (algorithm-specific stats)
        """
        # Create game
        game = Game(verbose=self.verbose)
        
        # Create agents
        agent1 = self._create_agent(agent1_config)
        agent2 = self._create_agent(agent2_config)
        
        # Run match
        turn_count = 0
        result = None
        
        while turn_count < self.max_turns:
            # Get moves from both agents
            move1, boost1 = agent1.get_best_move(game)
            move2, boost2 = agent2.get_best_move(game)
            
            # Execute moves
            result = game.step(move1, move2, boost1, boost2)
            
            if result is not None:
                break  # Game ended
            
            turn_count += 1
            
            if self.verbose and turn_count % 10 == 0:
                print(f"Turn {turn_count}: Agent1 length={game.agent1.length}, "
                      f"Agent2 length={game.agent2.length}")
        
        # Handle max turns reached
        if result is None and turn_count >= self.max_turns:
            if game.agent1.length > game.agent2.length:
                result = GameResult.AGENT1_WIN
            elif game.agent2.length > game.agent1.length:
                result = GameResult.AGENT2_WIN
            else:
                result = GameResult.DRAW
        
        # Get agent statistics
        agent1_stats = agent1.get_stats() if hasattr(agent1, 'get_stats') else {}
        agent2_stats = agent2.get_stats() if hasattr(agent2, 'get_stats') else {}
        
        return {
            'result': result,
            'turns': turn_count,
            'agent1_length': game.agent1.length,
            'agent2_length': game.agent2.length,
            'agent1_alive': game.agent1.alive,
            'agent2_alive': game.agent2.alive,
            'agent1_stats': agent1_stats,
            'agent2_stats': agent2_stats,
            'agent1_invalid_moves': game.agent1.invalid_move_count,
            'agent2_invalid_moves': game.agent2.invalid_move_count,
            'agent1_boosts_used': game.agent1.boosts_used,
            'agent2_boosts_used': game.agent2.boosts_used,
            'game': game  # Include game state for reward calculation
        }

    def _create_agent(self, config: Dict[str, Any]):
        """Create an agent from configuration."""
        agent_type = config.get('type', 'minimax')
        agent_id = config.get('agent_id', 1)
        aggressive_weight = config.get('aggressive_weight', 0.33)
        exploration_weight = config.get('exploration_weight', 0.33)
        safety_weight = config.get('safety_weight', 0.34)

        if agent_type == 'minimax':
            depth = config.get('depth', 4)
            use_transposition_table = config.get('use_transposition_table', True)
            tt_max_size = config.get('tt_max_size', 10000)

            return MinimaxAgent(
                agent_id=agent_id,
                depth=depth,
                aggressive_weight=aggressive_weight,
                exploration_weight=exploration_weight,
                safety_weight=safety_weight,
                use_transposition_table=use_transposition_table,
                tt_max_size=tt_max_size
            )
        elif agent_type == 'mcts':
            simulation_time_ms = config.get('simulation_time_ms', 120)
            exploration_constant = config.get('exploration_constant', 1.414)

            return MCTSAgent(
                agent_id=agent_id,
                simulation_time_ms=simulation_time_ms,
                aggressive_weight=aggressive_weight,
                exploration_weight=exploration_weight,
                safety_weight=safety_weight,
                exploration_constant=exploration_constant
            )
        elif agent_type == 'random':
            return RandomAgent(agent_id=agent_id)
        elif agent_type == 'heuristic':
            return HeuristicAgent(
                agent_id=agent_id,
                aggressive_weight=aggressive_weight,
                exploration_weight=exploration_weight,
                safety_weight=safety_weight,
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def run_match_batch(
        self,
        agent1_config: Dict[str, Any],
        agent2_config: Dict[str, Any],
        num_matches: int = 100,
        summary_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run multiple matches and return aggregate statistics.
        
        Returns:
            Dictionary with:
            - 'agent1_wins': int
            - 'agent2_wins': int
            - 'draws': int
            - 'agent1_avg_reward': float
            - 'agent2_avg_reward': float
            - 'agent1_avg_turns': float
            - 'agent2_avg_turns': float
            - 'matches': list of individual match results
        """
        from reward_functions import calculate_rewards_for_both_agents
        
        agent1_wins = 0
        agent2_wins = 0
        draws = 0
        agent1_total_reward = 0.0
        agent2_total_reward = 0.0
        total_turns = 0
        matches = []
        agent1_invalid_total = 0
        agent2_invalid_total = 0
        agent1_boosts_total = 0
        agent2_boosts_total = 0
        agent1_component_totals = defaultdict(float)
        agent2_component_totals = defaultdict(float)
        agent1_stat_sums = defaultdict(float)
        agent2_stat_sums = defaultdict(float)
        agent1_stat_counts = defaultdict(int)
        agent2_stat_counts = defaultdict(int)

        for i in range(num_matches):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"Running match {i + 1}/{num_matches}...")

            match_result = self.run_match(agent1_config, agent2_config)
            result = match_result['result']

            # Count wins
            if result == GameResult.AGENT1_WIN:
                agent1_wins += 1
            elif result == GameResult.AGENT2_WIN:
                agent2_wins += 1
            else:
                draws += 1
            
            # Calculate rewards
            (reward1, components1), (reward2, components2) = calculate_rewards_for_both_agents(
                match_result['game'], result, return_components=True
            )
            agent1_total_reward += reward1
            agent2_total_reward += reward2
            total_turns += match_result['turns']
            agent1_invalid_total += match_result.get('agent1_invalid_moves', 0)
            agent2_invalid_total += match_result.get('agent2_invalid_moves', 0)
            agent1_boosts_total += match_result.get('agent1_boosts_used', 0)
            agent2_boosts_total += match_result.get('agent2_boosts_used', 0)

            for key, value in components1.items():
                agent1_component_totals[key] += value
            for key, value in components2.items():
                agent2_component_totals[key] += value

            for key, value in match_result.get('agent1_stats', {}).items():
                if isinstance(value, (int, float)):
                    agent1_stat_sums[key] += value
                    agent1_stat_counts[key] += 1
            for key, value in match_result.get('agent2_stats', {}).items():
                if isinstance(value, (int, float)):
                    agent2_stat_sums[key] += value
                    agent2_stat_counts[key] += 1

            matches.append(match_result)

            if summary_callback is not None:
                summary_callback(i + 1, {
                    'result': result,
                    'turns': match_result['turns'],
                    'agent1_invalid_moves': match_result.get('agent1_invalid_moves', 0),
                    'agent2_invalid_moves': match_result.get('agent2_invalid_moves', 0),
                    'agent1_boosts_used': match_result.get('agent1_boosts_used', 0),
                    'agent2_boosts_used': match_result.get('agent2_boosts_used', 0),
                })

        def _average_stats(sums: Dict[str, float], counts: Dict[str, int]) -> Dict[str, float]:
            averages = {}
            for key, total in sums.items():
                count = counts.get(key, 0)
                if count > 0:
                    averages[key] = total / count
            return averages

        return {
            'agent1_wins': agent1_wins,
            'agent2_wins': agent2_wins,
            'draws': draws,
            'agent1_win_rate': agent1_wins / num_matches,
            'agent2_win_rate': agent2_wins / num_matches,
            'draw_rate': draws / num_matches,
            'agent1_avg_reward': agent1_total_reward / num_matches,
            'agent2_avg_reward': agent2_total_reward / num_matches,
            'avg_turns': total_turns / num_matches,
            'num_matches': num_matches,
            'matches': matches,
            'agent1_reward_breakdown': {
                key: value / num_matches for key, value in agent1_component_totals.items()
            },
            'agent2_reward_breakdown': {
                key: value / num_matches for key, value in agent2_component_totals.items()
            },
            'agent1_avg_stats': _average_stats(agent1_stat_sums, agent1_stat_counts),
            'agent2_avg_stats': _average_stats(agent2_stat_sums, agent2_stat_counts),
            'agent1_avg_invalid_moves': agent1_invalid_total / num_matches,
            'agent2_avg_invalid_moves': agent2_invalid_total / num_matches,
            'agent1_avg_boosts_used': agent1_boosts_total / num_matches,
            'agent2_avg_boosts_used': agent2_boosts_total / num_matches,
        }

