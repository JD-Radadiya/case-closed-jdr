"""
Structured logging for training progress and match outcomes.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler


class TrainingLogger:
    """
    Logger for training sessions with structured logging.
    """
    
    def __init__(self, logs_dir: str = "logs", log_level: int = logging.INFO):
        self.logs_dir = logs_dir
        self._ensure_logs_dir()
        
        # Create logger
        self.logger = logging.getLogger('training')
        self.logger.setLevel(log_level)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        log_filename = os.path.join(
            logs_dir,
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        self.log_file = log_filename
    
    def _ensure_logs_dir(self):
        """Ensure logs directory exists."""
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        if kwargs:
            message += f" | {json.dumps(kwargs)}"
        self.logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        if kwargs:
            message += f" | {json.dumps(kwargs)}"
        self.logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        if kwargs:
            message += f" | {json.dumps(kwargs)}"
        self.logger.error(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        if kwargs:
            message += f" | {json.dumps(kwargs)}"
        self.logger.debug(message)
    
    def log_training_start(self, config: Dict[str, Any]):
        """Log training session start."""
        self.info("=" * 80)
        self.info("TRAINING SESSION STARTED", **config)
        self.info("=" * 80)
    
    def log_training_end(self, summary: Dict[str, Any]):
        """Log training session end with summary."""
        self.info("=" * 80)
        self.info("TRAINING SESSION ENDED", **summary)
        self.info("=" * 80)
    
    def log_match_result(self, match_num: int, result: Dict[str, Any]):
        """Log individual match result."""
        self.debug(
            f"Match {match_num}",
            result=result.get('result'),
            turns=result.get('turns'),
            agent1_length=result.get('agent1_length'),
            agent2_length=result.get('agent2_length'),
            agent1_alive=result.get('agent1_alive'),
            agent2_alive=result.get('agent2_alive')
        )
    
    def log_batch_results(self, batch_label: str, results: Dict[str, Any]):
        """Log batch of match results."""
        payload = {
            'agent1_wins': results.get('agent1_wins'),
            'agent2_wins': results.get('agent2_wins'),
            'draws': results.get('draws'),
            'agent1_win_rate': results.get('agent1_win_rate'),
            'agent2_win_rate': results.get('agent2_win_rate'),
            'agent1_avg_reward': results.get('agent1_avg_reward'),
            'agent2_avg_reward': results.get('agent2_avg_reward'),
            'avg_turns': results.get('avg_turns'),
        }

        if 'agent1_reward_breakdown' in results:
            payload['agent1_reward_breakdown'] = results['agent1_reward_breakdown']
        if 'agent2_reward_breakdown' in results:
            payload['agent2_reward_breakdown'] = results['agent2_reward_breakdown']
        if 'agent1_avg_stats' in results:
            payload['agent1_avg_stats'] = results['agent1_avg_stats']
        if 'agent2_avg_stats' in results:
            payload['agent2_avg_stats'] = results['agent2_avg_stats']
        if 'agent1_avg_invalid_moves' in results:
            payload['agent1_avg_invalid_moves'] = results['agent1_avg_invalid_moves']
        if 'agent2_avg_invalid_moves' in results:
            payload['agent2_avg_invalid_moves'] = results['agent2_avg_invalid_moves']
        if 'agent1_avg_boosts_used' in results:
            payload['agent1_avg_boosts_used'] = results['agent1_avg_boosts_used']
        if 'agent2_avg_boosts_used' in results:
            payload['agent2_avg_boosts_used'] = results['agent2_avg_boosts_used']

        self.info(f"Batch {batch_label} Results", **payload)
    
    def log_model_evaluation(self, model_name: str, metrics: Dict[str, Any]):
        """Log model evaluation metrics."""
        self.info(
            f"Model Evaluation: {model_name}",
            **metrics
        )
    
    def log_parameter_update(self, iteration: int, old_params: Dict, new_params: Dict):
        """Log parameter update during optimization."""
        self.info(
            f"Iteration {iteration}: Parameter Update",
            old_aggressive=old_params.get('aggressive_weight'),
            old_exploration=old_params.get('exploration_weight'),
            old_safety=old_params.get('safety_weight'),
            new_aggressive=new_params.get('aggressive_weight'),
            new_exploration=new_params.get('exploration_weight'),
            new_safety=new_params.get('safety_weight')
        )
    
    def log_best_model_update(self, model_name: str, win_rate: float, avg_reward: float):
        """Log when best model is updated."""
        self.info(
            "Best Model Updated",
            model_name=model_name,
            win_rate=win_rate,
            avg_reward=avg_reward
        )
    
    def log_algorithm_stats(self, agent_name: str, stats: Dict[str, Any]):
        """Log algorithm-specific statistics."""
        self.debug(
            f"Algorithm Stats: {agent_name}",
            **stats
        )


def create_logger(logs_dir: str = "logs", log_level: int = logging.INFO) -> TrainingLogger:
    """Create a training logger instance."""
    return TrainingLogger(logs_dir=logs_dir, log_level=log_level)

