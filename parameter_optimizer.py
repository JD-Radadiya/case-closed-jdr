"""
Parameter optimization strategies for agent training.
Supports grid search, random search, and simple evolutionary algorithms.
"""

import random
import itertools
from typing import Dict, List, Tuple, Callable, Any
import numpy as np


class ParameterOptimizer:
    """
    Parameter optimizer for agent configurations.
    """
    
    def __init__(self, parameter_ranges: Dict[str, List[float]]):
        """
        Initialize optimizer with parameter ranges.
        
        Args:
            parameter_ranges: Dictionary mapping parameter names to lists of values
                Example: {
                    'aggressive_weight': [0.2, 0.4, 0.6, 0.8],
                    'exploration_weight': [0.2, 0.4, 0.6, 0.8],
                    'safety_weight': [0.2, 0.4, 0.6, 0.8]
                }
        """
        self.parameter_ranges = parameter_ranges
    
    def grid_search(self) -> List[Dict[str, float]]:
        """
        Generate all combinations of parameters (grid search).
        
        Returns:
            List of parameter configurations
        """
        # Get all parameter names and their values
        param_names = list(self.parameter_ranges.keys())
        param_values = [self.parameter_ranges[name] for name in param_names]
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        configs = []
        for combo in combinations:
            config = {name: value for name, value in zip(param_names, combo)}
            # Normalize weights if they are strategy parameters
            config = self._normalize_weights(config)
            configs.append(config)
        
        return configs
    
    def random_search(self, num_samples: int = 100) -> List[Dict[str, float]]:
        """
        Generate random parameter configurations.
        
        Args:
            num_samples: Number of random configurations to generate
        
        Returns:
            List of parameter configurations
        """
        configs = []
        for _ in range(num_samples):
            config = {}
            for param_name, param_values in self.parameter_ranges.items():
                config[param_name] = random.choice(param_values)
            # Normalize weights
            config = self._normalize_weights(config)
            configs.append(config)
        
        return configs
    
    def evolutionary_search(self, initial_configs: List[Dict[str, float]],
                          fitness_scores: List[float],
                          num_offspring: int = 10,
                          mutation_rate: float = 0.1,
                          mutation_std: float = 0.1) -> List[Dict[str, float]]:
        """
        Generate offspring using evolutionary algorithm.
        
        Args:
            initial_configs: List of parent configurations
            fitness_scores: Fitness scores for each parent (higher is better)
            num_offspring: Number of offspring to generate
            mutation_rate: Probability of mutating each parameter
            mutation_std: Standard deviation for mutation (as fraction of range)
        
        Returns:
            List of offspring configurations
        """
        if not initial_configs or not fitness_scores:
            return self.random_search(num_offspring)
        
        # Normalize fitness scores to probabilities
        fitness_scores = np.array(fitness_scores)
        fitness_scores = fitness_scores - fitness_scores.min() + 1e-6  # Avoid zeros
        probabilities = fitness_scores / fitness_scores.sum()
        
        offspring = []
        for _ in range(num_offspring):
            # Select parent based on fitness
            parent_idx = np.random.choice(len(initial_configs), p=probabilities)
            parent = initial_configs[parent_idx].copy()
            
            # Mutate parameters
            for param_name in parent:
                if param_name in self.parameter_ranges and random.random() < mutation_rate:
                    # Mutate this parameter
                    param_values = self.parameter_ranges[param_name]
                    if isinstance(param_values[0], (int, float)):
                        # Continuous parameter
                        current_value = parent[param_name]
                        param_range = max(param_values) - min(param_values)
                        mutation = random.gauss(0, mutation_std * param_range)
                        new_value = current_value + mutation
                        # Clip to valid range
                        new_value = max(min(param_values), min(max(param_values), new_value))
                        # Snap to nearest valid value
                        parent[param_name] = min(param_values, key=lambda x: abs(x - new_value))
                    else:
                        # Discrete parameter - randomly change
                        parent[param_name] = random.choice(param_values)
            
            # Normalize weights
            parent = self._normalize_weights(parent)
            offspring.append(parent)
        
        return offspring
    
    def _normalize_weights(self, config: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weight parameters to sum to 1.0.
        Applies to aggressive_weight, exploration_weight, safety_weight.
        """
        weight_params = ['aggressive_weight', 'exploration_weight', 'safety_weight']
        if all(param in config for param in weight_params):
            total = sum(config[param] for param in weight_params)
            if total > 0:
                for param in weight_params:
                    config[param] = config[param] / total
        return config
    
    def get_default_ranges(self, agent_type: str) -> Dict[str, List[float]]:
        """
        Get default parameter ranges for an agent type.
        
        Args:
            agent_type: 'minimax' or 'mcts'
        
        Returns:
            Dictionary of parameter ranges
        """
        ranges = {
            'aggressive_weight': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            'exploration_weight': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            'safety_weight': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        }
        
        if agent_type == 'minimax':
            ranges['depth'] = [2, 3, 4, 5]
        elif agent_type == 'mcts':
            ranges['simulation_time_ms'] = [50, 100, 150, 200]
        
        return ranges

