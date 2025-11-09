"""Training script for Case Closed agents with asymmetric evaluation."""

import argparse
import random
from statistics import mean
from typing import Any, Dict, List, Tuple

from case_closed_game import GameResult
from match_runner import MatchRunner
from model_manager import ModelManager
from logger import create_logger
from parameter_optimizer import ParameterOptimizer
from hybrid_training import HybridMinimaxRLTrainer

DEFAULT_BASELINE_WEIGHTS = {
    'aggressive_weight': 0.4,
    'exploration_weight': 0.3,
    'safety_weight': 0.3,
}


def _build_agent_config(agent_type: str, base_config: Dict[str, Any], agent_id: int) -> Dict[str, Any]:
    config = {k: v for k, v in base_config.items() if k != 'metadata'}
    config['type'] = agent_type
    config['agent_id'] = agent_id
    config.setdefault('aggressive_weight', DEFAULT_BASELINE_WEIGHTS['aggressive_weight'])
    config.setdefault('exploration_weight', DEFAULT_BASELINE_WEIGHTS['exploration_weight'])
    config.setdefault('safety_weight', DEFAULT_BASELINE_WEIGHTS['safety_weight'])

    if agent_type == 'minimax':
        config.setdefault('depth', 4)
        config.setdefault('use_transposition_table', True)
        config.setdefault('tt_max_size', 10000)
    elif agent_type == 'mcts':
        config.setdefault('simulation_time_ms', 120)
        config.setdefault('exploration_constant', 1.414)
    elif agent_type == 'hybrid':
        config.setdefault('minimax_depth', 3)
        config.setdefault('minimax_weight', 0.4)
        config.setdefault('risk_weight', 0.5)

    return config


def _prepare_opponent_config(opponent: Dict[str, Any], agent_id: int) -> Dict[str, Any]:
    config = {k: v for k, v in opponent.items() if k != 'metadata'}
    config['agent_id'] = agent_id
    config.setdefault('aggressive_weight', DEFAULT_BASELINE_WEIGHTS['aggressive_weight'])
    config.setdefault('exploration_weight', DEFAULT_BASELINE_WEIGHTS['exploration_weight'])
    config.setdefault('safety_weight', DEFAULT_BASELINE_WEIGHTS['safety_weight'])
    return config


def _combine_weighted_metrics(first: Dict[str, float], second: Dict[str, float],
                              weight_first: float, weight_second: float) -> Dict[str, float]:
    combined: Dict[str, float] = {}
    total_weight = weight_first + weight_second
    if total_weight == 0:
        return combined
    keys = set(first.keys()) | set(second.keys())
    for key in keys:
        combined[key] = (
            first.get(key, 0.0) * weight_first + second.get(key, 0.0) * weight_second
        ) / total_weight
    return combined


def _run_cross_play(
    logger,
    match_runner: MatchRunner,
    agent_type: str,
    candidate_config: Dict[str, Any],
    opponent_config: Dict[str, Any],
    opponent_name: str,
    num_matches: int,
    batch_tag: str,
) -> Dict[str, Any]:
    candidate_as_agent1 = _build_agent_config(agent_type, candidate_config, agent_id=1)
    opponent_as_agent2 = _prepare_opponent_config(opponent_config, agent_id=2)
    def _summary_callback_factory(leg_label: str):
        def _callback(match_index: int, summary: Dict[str, Any]):
            result = summary.get('result')
            if isinstance(result, GameResult):
                result_label = result.name
            else:
                result_label = str(result)
            logger.info(
                "Match summary",
                batch=batch_tag,
                opponent=opponent_name,
                leg=leg_label,
                match_index=match_index,
                result=result_label,
                turns=summary.get('turns'),
                agent1_invalid_moves=summary.get('agent1_invalid_moves'),
                agent2_invalid_moves=summary.get('agent2_invalid_moves'),
                agent1_boosts_used=summary.get('agent1_boosts_used'),
                agent2_boosts_used=summary.get('agent2_boosts_used'),
            )

        return _callback

    forward_results = match_runner.run_match_batch(
        candidate_as_agent1,
        opponent_as_agent2,
        num_matches=num_matches,
        summary_callback=_summary_callback_factory('A'),
    )
    logger.log_batch_results(f"{batch_tag}_{opponent_name}_A", forward_results)

    opponent_as_agent1 = _prepare_opponent_config(opponent_config, agent_id=1)
    candidate_as_agent2 = _build_agent_config(agent_type, candidate_config, agent_id=2)
    reverse_results = match_runner.run_match_batch(
        opponent_as_agent1,
        candidate_as_agent2,
        num_matches=num_matches,
        summary_callback=_summary_callback_factory('B'),
    )
    logger.log_batch_results(f"{batch_tag}_{opponent_name}_B", reverse_results)

    total_matches = forward_results['num_matches'] + reverse_results['num_matches']
    candidate_wins = forward_results['agent1_wins'] + reverse_results['agent2_wins']
    draws = forward_results['draws'] + reverse_results['draws']

    avg_reward = (
        forward_results['agent1_avg_reward'] * forward_results['num_matches'] +
        reverse_results['agent2_avg_reward'] * reverse_results['num_matches']
    ) / total_matches

    avg_turns = (
        forward_results['avg_turns'] * forward_results['num_matches'] +
        reverse_results['avg_turns'] * reverse_results['num_matches']
    ) / total_matches

    avg_invalid_moves = (
        forward_results.get('agent1_avg_invalid_moves', 0.0) * forward_results['num_matches'] +
        reverse_results.get('agent2_avg_invalid_moves', 0.0) * reverse_results['num_matches']
    ) / total_matches

    avg_boosts_used = (
        forward_results.get('agent1_avg_boosts_used', 0.0) * forward_results['num_matches'] +
        reverse_results.get('agent2_avg_boosts_used', 0.0) * reverse_results['num_matches']
    ) / total_matches

    reward_breakdown = _combine_weighted_metrics(
        forward_results.get('agent1_reward_breakdown', {}),
        reverse_results.get('agent2_reward_breakdown', {}),
        forward_results['num_matches'],
        reverse_results['num_matches'],
    )
    stats = _combine_weighted_metrics(
        forward_results.get('agent1_avg_stats', {}),
        reverse_results.get('agent2_avg_stats', {}),
        forward_results['num_matches'],
        reverse_results['num_matches'],
    )

    candidate_win_rate = candidate_wins / total_matches
    draw_rate = draws / total_matches

    return {
        'win_rate': candidate_win_rate,
        'avg_reward': avg_reward,
        'draw_rate': draw_rate,
        'avg_turns': avg_turns,
        'avg_invalid_moves': avg_invalid_moves,
        'avg_boosts_used': avg_boosts_used,
        'reward_breakdown': reward_breakdown,
        'avg_stats': stats,
        'matches_played': total_matches,
    }


def _build_baselines(agent_type: str, model_manager: ModelManager) -> List[Tuple[str, Dict[str, Any]]]:
    baselines: List[Tuple[str, Dict[str, Any]]] = [
        ('random', {'type': 'random'}),
        ('heuristic', {'type': 'heuristic', **DEFAULT_BASELINE_WEIGHTS}),
    ]

    minimax_default = {
        'type': 'minimax',
        **DEFAULT_BASELINE_WEIGHTS,
        'depth': 4,
        'use_transposition_table': True,
        'tt_max_size': 10000,
    }
    mcts_default = {
        'type': 'mcts',
        **DEFAULT_BASELINE_WEIGHTS,
        'simulation_time_ms': 120,
        'exploration_constant': 1.414,
    }

    if agent_type in ('minimax', 'hybrid'):
        baselines.append(('default_minimax', minimax_default))
        baselines.append(('default_mcts', mcts_default))
    else:
        baselines.append(('default_mcts', mcts_default))
        baselines.append(('default_minimax', minimax_default))

    incumbent = model_manager.load_best_model(agent_type)
    if incumbent is not None:
        baselines.append(('incumbent_best', {k: v for k, v in incumbent.items() if k != 'metadata'}))

    return baselines


def _evaluate_configuration(
    logger,
    match_runner: MatchRunner,
    agent_type: str,
    candidate_config: Dict[str, Any],
    baselines: List[Tuple[str, Dict[str, Any]]],
    num_matches: int,
    iteration: int,
    index: int,
) -> Dict[str, Any]:
    baseline_metrics = {}

    for opponent_name, opponent_config in baselines:
        metrics = _run_cross_play(
            logger,
            match_runner,
            agent_type,
            candidate_config,
            opponent_config,
            opponent_name,
            num_matches,
            batch_tag=f"iter{iteration}_cfg{index}",
        )
        baseline_metrics[opponent_name] = metrics
        logger.info(
            f"Evaluation vs {opponent_name}",
            iteration=iteration,
            config_index=index,
            win_rate=metrics['win_rate'],
            avg_reward=metrics['avg_reward'],
            draw_rate=metrics['draw_rate'],
            avg_turns=metrics['avg_turns'],
            avg_invalid_moves=metrics['avg_invalid_moves'],
            avg_boosts_used=metrics['avg_boosts_used'],
            reward_breakdown=metrics['reward_breakdown'],
        )

        if metrics['avg_invalid_moves'] > 0.0:
            logger.warning(
                "Invalid moves detected during evaluation",
                iteration=iteration,
                config_index=index,
                opponent=opponent_name,
                avg_invalid_moves=metrics['avg_invalid_moves'],
            )

    overall_win_rate = mean(metric['win_rate'] for metric in baseline_metrics.values())
    overall_avg_reward = mean(metric['avg_reward'] for metric in baseline_metrics.values())
    overall_avg_invalid_moves = mean(metric['avg_invalid_moves'] for metric in baseline_metrics.values())

    overall_reward_breakdown: Dict[str, float] = {}
    total_reward_weight = sum(metric['matches_played'] for metric in baseline_metrics.values())
    if total_reward_weight > 0:
        for metric in baseline_metrics.values():
            weight = metric['matches_played']
            for key, value in metric['reward_breakdown'].items():
                overall_reward_breakdown[key] = overall_reward_breakdown.get(key, 0.0) + value * weight
        for key in list(overall_reward_breakdown.keys()):
            overall_reward_breakdown[key] /= total_reward_weight

    logger.info(
        "Configuration summary",
        iteration=iteration,
        config_index=index,
        overall_win_rate=overall_win_rate,
        overall_avg_reward=overall_avg_reward,
        overall_avg_invalid_moves=overall_avg_invalid_moves,
    )

    return {
        'config': candidate_config,
        'baseline_metrics': baseline_metrics,
        'overall_win_rate': overall_win_rate,
        'overall_avg_reward': overall_avg_reward,
        'overall_avg_invalid_moves': overall_avg_invalid_moves,
        'overall_reward_breakdown': overall_reward_breakdown,
    }


def _should_promote(metrics: Dict[str, Any]) -> bool:
    baselines = metrics['baseline_metrics']
    threshold_checks = {
        'random': 0.6,
        'heuristic': 0.55,
    }

    for name, threshold in threshold_checks.items():
        if name in baselines and baselines[name]['win_rate'] < threshold:
            return False

    incumbent_metrics = baselines.get('incumbent_best')
    if incumbent_metrics and incumbent_metrics['win_rate'] <= 0.5:
        return False

    if metrics.get('overall_avg_invalid_moves', 0.0) > 0.05:
        return False

    return True


def _train_agent_type(
    agent_type: str,
    num_matches: int,
    num_iterations: int,
    optimization_strategy: str,
    logs_dir: str,
    models_dir: str,
) -> Tuple[Dict[str, Any], float, float]:
    logger = create_logger(logs_dir=logs_dir)
    model_manager = ModelManager(models_dir=models_dir)
    match_runner = MatchRunner(verbose=False)

    logger.log_training_start({
        'agent_type': agent_type,
        'num_matches': num_matches,
        'num_iterations': num_iterations,
        'optimization_strategy': optimization_strategy,
    })

    optimizer = ParameterOptimizer({})
    optimizer.parameter_ranges = optimizer.get_default_ranges(agent_type)

    if optimization_strategy == 'grid_search':
        candidate_pool = optimizer.grid_search()
    elif optimization_strategy == 'random_search':
        candidate_pool = optimizer.random_search(num_samples=50)
    else:
        raise ValueError(f"Unknown optimization strategy: {optimization_strategy}")

    random.shuffle(candidate_pool)
    population_size = min(max(6, num_matches // 5), len(candidate_pool))
    if population_size == 0:
        population_size = len(candidate_pool)
    population = candidate_pool[:population_size]

    baselines = _build_baselines(agent_type, model_manager)

    best_record = {
        'config': None,
        'overall_win_rate': 0.0,
        'overall_avg_reward': 0.0,
        'metadata': {},
    }

    for iteration in range(1, num_iterations + 1):
        logger.info(
            "Iteration start",
            iteration=iteration,
            population_size=len(population),
            baselines=[name for name, _ in baselines],
        )

        iteration_results = []
        for idx, config in enumerate(population, start=1):
            logger.info("Evaluating configuration", iteration=iteration, config_index=idx, config=config)
            metrics = _evaluate_configuration(
                logger,
                match_runner,
                agent_type,
                config,
                baselines,
                num_matches,
                iteration,
                idx,
            )
            iteration_results.append(metrics)

            metadata = {
                'overall_win_rate': metrics['overall_win_rate'],
                'overall_avg_reward': metrics['overall_avg_reward'],
                'overall_avg_invalid_moves': metrics['overall_avg_invalid_moves'],
                'reward_breakdown': metrics['overall_reward_breakdown'],
                'baselines': metrics['baseline_metrics'],
                'iteration': iteration,
                'num_matches_per_baseline': num_matches * 2,
            }

            model_config = {
                'type': agent_type,
                **config,
                'metadata': metadata,
            }
            model_name = f"{agent_type}_iter{iteration}_{idx}"
            model_manager.save_model(model_config, model_name=model_name)

            if metrics['overall_win_rate'] > best_record['overall_win_rate'] or (
                abs(metrics['overall_win_rate'] - best_record['overall_win_rate']) < 1e-6 and
                metrics['overall_avg_reward'] > best_record['overall_avg_reward']
            ):
                best_record = {
                    'config': config,
                    'overall_win_rate': metrics['overall_win_rate'],
                    'overall_avg_reward': metrics['overall_avg_reward'],
                    'overall_avg_invalid_moves': metrics['overall_avg_invalid_moves'],
                    'metadata': metadata,
                }

                if _should_promote(metrics):
                    updated = model_manager.update_best_model(
                        model_config,
                        metrics['overall_win_rate'],
                        metrics['overall_avg_reward'],
                    )
                    if updated:
                        logger.log_best_model_update(model_name, metrics['overall_win_rate'], metrics['overall_avg_reward'])

        iteration_results.sort(key=lambda m: (m['overall_win_rate'], m['overall_avg_reward']), reverse=True)
        top_k = max(2, len(iteration_results) // 2)
        elite_configs = [result['config'] for result in iteration_results[:top_k]]
        fitness_scores = [result['overall_win_rate'] for result in iteration_results[:top_k]]

        if iteration < num_iterations:
            offspring_needed = max(0, population_size - len(elite_configs))
            offspring = optimizer.evolutionary_search(
                elite_configs,
                fitness_scores,
                num_offspring=offspring_needed,
            )
            population = elite_configs + offspring
        else:
            population = []

    logger.log_training_end({
        'best_win_rate': best_record['overall_win_rate'],
        'best_avg_reward': best_record['overall_avg_reward'],
        'best_config': best_record['config'],
    })

    return best_record['config'], best_record['overall_win_rate'], best_record['overall_avg_reward']


def train_minimax_vs_minimax(
    num_matches: int = 40,
    num_iterations: int = 5,
    optimization_strategy: str = 'grid_search',
    logs_dir: str = 'logs',
    models_dir: str = 'models',
):
    """Train Minimax agents with asymmetric evaluation."""
    return _train_agent_type('minimax', num_matches, num_iterations, optimization_strategy, logs_dir, models_dir)


def train_mcts_vs_mcts(
    num_matches: int = 40,
    num_iterations: int = 5,
    optimization_strategy: str = 'grid_search',
    logs_dir: str = 'logs',
    models_dir: str = 'models',
):
    """Train MCTS agents with asymmetric evaluation."""
    return _train_agent_type('mcts', num_matches, num_iterations, optimization_strategy, logs_dir, models_dir)


def train_hybrid_minimax_rl(
    num_matches: int = 20,
    num_iterations: int = 5,
    logs_dir: str = 'logs',
    models_dir: str = 'models',
    teacher_depth: int = 4,
    learner_depth: int = 3,
):
    """Train the hybrid minimax + RL agent."""
    logger = create_logger(logs_dir=logs_dir)
    model_manager = ModelManager(models_dir=models_dir)
    match_runner = MatchRunner(verbose=False)

    logger.log_training_start({
        'agent_type': 'hybrid',
        'num_matches': num_matches,
        'num_iterations': num_iterations,
        'teacher_depth': teacher_depth,
        'learner_depth': learner_depth,
    })

    trainer = HybridMinimaxRLTrainer(
        match_runner,
        agent_id=1,
        teacher_depth=teacher_depth,
        learner_depth=learner_depth,
    )

    baselines = _build_baselines('hybrid', model_manager)

    best_record = {
        'config': None,
        'overall_win_rate': 0.0,
        'overall_avg_reward': 0.0,
        'overall_avg_invalid_moves': 0.0,
        'metadata': {},
    }

    for iteration in range(1, num_iterations + 1):
        logger.info(
            'Hybrid iteration start',
            iteration=iteration,
            baselines=[name for name, _ in baselines],
        )

        rollout_stats = trainer.train_iteration(baselines, matches_per_opponent=num_matches)
        for opponent_name, stats in rollout_stats.items():
            logger.info(
                'Rollout summary',
                iteration=iteration,
                opponent=opponent_name,
                avg_reward=stats.get('avg_reward'),
                win_rate=stats.get('win_rate'),
                draw_rate=stats.get('draw_rate'),
                avg_turns=stats.get('avg_turns'),
            )

        candidate_config = trainer.build_agent_config()
        metrics = _evaluate_configuration(
            logger,
            match_runner,
            'hybrid',
            candidate_config,
            baselines,
            num_matches,
            iteration,
            1,
        )

        metadata = {
            'overall_win_rate': metrics['overall_win_rate'],
            'overall_avg_reward': metrics['overall_avg_reward'],
            'overall_avg_invalid_moves': metrics['overall_avg_invalid_moves'],
            'reward_breakdown': metrics['overall_reward_breakdown'],
            'baselines': metrics['baseline_metrics'],
            'iteration': iteration,
            'num_matches_per_baseline': num_matches * 2,
        }

        model_config = {
            'type': 'hybrid',
            **candidate_config,
            'metadata': metadata,
        }
        model_name = f"hybrid_iter{iteration}"
        model_manager.save_model(model_config, model_name=model_name)

        if metrics['overall_win_rate'] > best_record['overall_win_rate'] or (
            abs(metrics['overall_win_rate'] - best_record['overall_win_rate']) < 1e-6
            and metrics['overall_avg_reward'] > best_record['overall_avg_reward']
        ):
            best_record = {
                'config': candidate_config,
                'overall_win_rate': metrics['overall_win_rate'],
                'overall_avg_reward': metrics['overall_avg_reward'],
                'overall_avg_invalid_moves': metrics['overall_avg_invalid_moves'],
                'metadata': metadata,
            }

            if _should_promote(metrics):
                updated = model_manager.update_best_model(
                    model_config,
                    metrics['overall_win_rate'],
                    metrics['overall_avg_reward'],
                )
                if updated:
                    logger.log_best_model_update(
                        model_name,
                        metrics['overall_win_rate'],
                        metrics['overall_avg_reward'],
                    )

    logger.log_training_end({
        'best_win_rate': best_record['overall_win_rate'],
        'best_avg_reward': best_record['overall_avg_reward'],
        'best_config': best_record['config'],
    })

    return best_record['config'], best_record['overall_win_rate'], best_record['overall_avg_reward']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Case Closed agents with asymmetric evaluation.')
    parser.add_argument('--agent_type', choices=['minimax', 'mcts', 'hybrid'], default='minimax')
    parser.add_argument('--num_matches', type=int, default=40, help='Number of matches per pairing per direction')
    parser.add_argument('--num_iterations', type=int, default=5)
    parser.add_argument('--optimization_strategy', choices=['grid_search', 'random_search'], default='grid_search')
    parser.add_argument('--logs_dir', default='logs')
    parser.add_argument('--models_dir', default='models')
    parser.add_argument('--teacher_depth', type=int, default=4)
    parser.add_argument('--learner_depth', type=int, default=3)

    args = parser.parse_args()

    if args.agent_type == 'minimax':
        train_minimax_vs_minimax(
            num_matches=args.num_matches,
            num_iterations=args.num_iterations,
            optimization_strategy=args.optimization_strategy,
            logs_dir=args.logs_dir,
            models_dir=args.models_dir,
        )
    elif args.agent_type == 'mcts':
        train_mcts_vs_mcts(
            num_matches=args.num_matches,
            num_iterations=args.num_iterations,
            optimization_strategy=args.optimization_strategy,
            logs_dir=args.logs_dir,
            models_dir=args.models_dir,
        )
    else:
        train_hybrid_minimax_rl(
            num_matches=args.num_matches,
            num_iterations=args.num_iterations,
            logs_dir=args.logs_dir,
            models_dir=args.models_dir,
            teacher_depth=args.teacher_depth,
            learner_depth=args.learner_depth,
        )
