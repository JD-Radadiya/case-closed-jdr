"""
Training script for Case Closed agents.
Supports self-play training with parameter optimization.
"""

import argparse
import time
from typing import Dict, List, Any
from match_runner import MatchRunner
from model_manager import ModelManager
from logger import create_logger
from parameter_optimizer import ParameterOptimizer
from reward_functions import calculate_rewards_for_both_agents


def train_minimax_vs_minimax(
    num_matches: int = 100,
    num_iterations: int = 10,
    optimization_strategy: str = 'grid_search',
    logs_dir: str = 'logs',
    models_dir: str = 'models'
):
    """Train Minimax agents against each other."""
    logger = create_logger(logs_dir=logs_dir)
    model_manager = ModelManager(models_dir=models_dir)
    match_runner = MatchRunner(verbose=False)
    
    logger.log_training_start({
        'agent_type': 'minimax',
        'num_matches': num_matches,
        'num_iterations': num_iterations,
        'optimization_strategy': optimization_strategy
    })
    
    # Get parameter ranges
    optimizer = ParameterOptimizer({})
    param_ranges = optimizer.get_default_ranges('minimax')
    optimizer.parameter_ranges = param_ranges
    
    # Generate parameter configurations
    if optimization_strategy == 'grid_search':
        configs = optimizer.grid_search()
        logger.info(f"Grid search: {len(configs)} configurations to test")
    elif optimization_strategy == 'random_search':
        configs = optimizer.random_search(num_samples=50)
        logger.info(f"Random search: {len(configs)} configurations to test")
    else:
        raise ValueError(f"Unknown optimization strategy: {optimization_strategy}")
    
    best_config = None
    best_win_rate = 0.0
    best_avg_reward = 0.0
    
    # Evaluate each configuration
    for i, config in enumerate(configs):
        logger.info(f"Evaluating configuration {i+1}/{len(configs)}: {config}")
        
        # Create agent configurations
        agent1_config = {
            'type': 'minimax',
            'agent_id': 1,
            **config
        }
        agent2_config = {
            'type': 'minimax',
            'agent_id': 2,
            **config
        }
        
        # Run matches
        results = match_runner.run_match_batch(agent1_config, agent2_config, num_matches=num_matches)
        
        # Calculate metrics (use agent1 as representative)
        win_rate = results['agent1_win_rate']
        avg_reward = results['agent1_avg_reward']
        
        logger.log_batch_results(i+1, results)
        
        # Save model
        model_config = {
            'type': 'minimax',
            **config,
            'metadata': {
                'win_rate': win_rate,
                'avg_reward': avg_reward,
                'num_matches': num_matches
            }
        }
        model_name = f"minimax_config_{i+1}"
        model_manager.save_model(model_config, model_name=model_name)
        
        # Update best model
        if win_rate > best_win_rate or (win_rate == best_win_rate and avg_reward > best_avg_reward):
            best_config = config
            best_win_rate = win_rate
            best_avg_reward = avg_reward
            model_manager.update_best_model(model_config, win_rate, avg_reward)
            logger.log_best_model_update(model_name, win_rate, avg_reward)
    
    logger.log_training_end({
        'best_win_rate': best_win_rate,
        'best_avg_reward': best_avg_reward,
        'best_config': best_config
    })
    
    return best_config, best_win_rate, best_avg_reward


def train_mcts_vs_mcts(
    num_matches: int = 100,
    num_iterations: int = 10,
    optimization_strategy: str = 'grid_search',
    logs_dir: str = 'logs',
    models_dir: str = 'models'
):
    """Train MCTS agents against each other."""
    logger = create_logger(logs_dir=logs_dir)
    model_manager = ModelManager(models_dir=models_dir)
    match_runner = MatchRunner(verbose=False)
    
    logger.log_training_start({
        'agent_type': 'mcts',
        'num_matches': num_matches,
        'num_iterations': num_iterations,
        'optimization_strategy': optimization_strategy
    })
    
    # Get parameter ranges
    optimizer = ParameterOptimizer({})
    param_ranges = optimizer.get_default_ranges('mcts')
    optimizer.parameter_ranges = param_ranges
    
    # Generate parameter configurations
    if optimization_strategy == 'grid_search':
        configs = optimizer.grid_search()
        logger.info(f"Grid search: {len(configs)} configurations to test")
    elif optimization_strategy == 'random_search':
        configs = optimizer.random_search(num_samples=50)
        logger.info(f"Random search: {len(configs)} configurations to test")
    else:
        raise ValueError(f"Unknown optimization strategy: {optimization_strategy}")
    
    best_config = None
    best_win_rate = 0.0
    best_avg_reward = 0.0
    
    # Evaluate each configuration
    for i, config in enumerate(configs):
        logger.info(f"Evaluating configuration {i+1}/{len(configs)}: {config}")
        
        # Create agent configurations
        agent1_config = {
            'type': 'mcts',
            'agent_id': 1,
            **config
        }
        agent2_config = {
            'type': 'mcts',
            'agent_id': 2,
            **config
        }
        
        # Run matches
        results = match_runner.run_match_batch(agent1_config, agent2_config, num_matches=num_matches)
        
        # Calculate metrics
        win_rate = results['agent1_win_rate']
        avg_reward = results['agent1_avg_reward']
        
        logger.log_batch_results(i+1, results)
        
        # Save model
        model_config = {
            'type': 'mcts',
            **config,
            'metadata': {
                'win_rate': win_rate,
                'avg_reward': avg_reward,
                'num_matches': num_matches
            }
        }
        model_name = f"mcts_config_{i+1}"
        model_manager.save_model(model_config, model_name=model_name)
        
        # Update best model
        if win_rate > best_win_rate or (win_rate == best_win_rate and avg_reward > best_avg_reward):
            best_config = config
            best_win_rate = win_rate
            best_avg_reward = avg_reward
            model_manager.update_best_model(model_config, win_rate, avg_reward)
            logger.log_best_model_update(model_name, win_rate, avg_reward)
    
    logger.log_training_end({
        'best_win_rate': best_win_rate,
        'best_avg_reward': best_avg_reward,
        'best_config': best_config
    })
    
    return best_config, best_win_rate, best_avg_reward


def train_minimax_vs_mcts(
    num_matches: int = 100,
    logs_dir: str = 'logs',
    models_dir: str = 'models'
):
    """Train Minimax vs MCTS agents."""
    logger = create_logger(logs_dir=logs_dir)
    model_manager = ModelManager(models_dir=models_dir)
    match_runner = MatchRunner(verbose=False)
    
    logger.log_training_start({
        'agent1_type': 'minimax',
        'agent2_type': 'mcts',
        'num_matches': num_matches
    })
    
    # Load best models or use defaults
    minimax_config = model_manager.load_best_model('minimax')
    if minimax_config is None:
        logger.warning("No best minimax model found, using defaults")
        minimax_config = {
            'type': 'minimax',
            'depth': 4,
            'aggressive_weight': 0.33,
            'exploration_weight': 0.33,
            'safety_weight': 0.34
        }
    
    mcts_config = model_manager.load_best_model('mcts')
    if mcts_config is None:
        logger.warning("No best MCTS model found, using defaults")
        mcts_config = {
            'type': 'mcts',
            'simulation_time_ms': 120,
            'aggressive_weight': 0.33,
            'exploration_weight': 0.33,
            'safety_weight': 0.34
        }
    
    # Create agent configurations
    agent1_config = model_manager.get_model_config_for_agent('best_minimax', 1)
    agent2_config = model_manager.get_model_config_for_agent('best_mcts', 2)
    
    # Run matches
    results = match_runner.run_match_batch(agent1_config, agent2_config, num_matches=num_matches)
    
    logger.log_batch_results(1, results)
    logger.log_training_end(results)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train Case Closed agents')
    parser.add_argument('--agent-type', type=str, choices=['minimax', 'mcts', 'both'],
                       default='minimax', help='Type of agent to train')
    parser.add_argument('--num-matches', type=int, default=100,
                       help='Number of matches per configuration')
    parser.add_argument('--optimization-strategy', type=str,
                       choices=['grid_search', 'random_search'],
                       default='random_search',
                       help='Optimization strategy')
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='Directory for log files')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory for model files')
    
    args = parser.parse_args()
    
    if args.agent_type == 'minimax':
        train_minimax_vs_minimax(
            num_matches=args.num_matches,
            optimization_strategy=args.optimization_strategy,
            logs_dir=args.logs_dir,
            models_dir=args.models_dir
        )
    elif args.agent_type == 'mcts':
        train_mcts_vs_mcts(
            num_matches=args.num_matches,
            optimization_strategy=args.optimization_strategy,
            logs_dir=args.logs_dir,
            models_dir=args.models_dir
        )
    elif args.agent_type == 'both':
        # Train both separately, then pit them against each other
        train_minimax_vs_minimax(
            num_matches=args.num_matches,
            optimization_strategy=args.optimization_strategy,
            logs_dir=args.logs_dir,
            models_dir=args.models_dir
        )
        train_mcts_vs_mcts(
            num_matches=args.num_matches,
            optimization_strategy=args.optimization_strategy,
            logs_dir=args.logs_dir,
            models_dir=args.models_dir
        )
        train_minimax_vs_mcts(
            num_matches=args.num_matches,
            logs_dir=args.logs_dir,
            models_dir=args.models_dir
        )


if __name__ == '__main__':
    main()

