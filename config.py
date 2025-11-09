"""
Central configuration management for Case Closed agents.
"""

import os
from typing import Dict, Any


# Default configuration values
DEFAULT_CONFIG = {
    # Minimax configuration
    'minimax': {
        'depth': 4,
        'aggressive_weight': 0.33,
        'exploration_weight': 0.33,
        'safety_weight': 0.34,
        'use_transposition_table': True,
        'tt_max_size': 10000
    },
    
    # MCTS configuration
    'mcts': {
        'simulation_time_ms': 120,
        'aggressive_weight': 0.33,
        'exploration_weight': 0.33,
        'safety_weight': 0.34,
        'exploration_constant': 1.414
    },
    
    # Training configuration
    'training': {
        'num_matches': 100,
        'optimization_strategy': 'random_search',
        'num_iterations': 10,
        'win_weight': 0.8,
        'survival_weight': 0.1,
        'territory_weight': 0.1
    },
    
    # Paths
    'paths': {
        'models_dir': 'models',
        'logs_dir': 'logs'
    },
    
    # Game configuration
    'game': {
        'max_turns': 200,
        'board_height': 18,
        'board_width': 20
    }
}


def get_config(section: str = None, key: str = None) -> Any:
    """
    Get configuration value.
    
    Args:
        section: Configuration section (e.g., 'minimax', 'mcts', 'training')
        key: Configuration key within section
    
    Returns:
        Configuration value or dictionary
    """
    if section is None:
        return DEFAULT_CONFIG
    
    if key is None:
        return DEFAULT_CONFIG.get(section, {})
    
    return DEFAULT_CONFIG.get(section, {}).get(key)


def update_config(section: str, key: str, value: Any):
    """
    Update configuration value.
    
    Args:
        section: Configuration section
        key: Configuration key
        value: New value
    """
    if section not in DEFAULT_CONFIG:
        DEFAULT_CONFIG[section] = {}
    DEFAULT_CONFIG[section][key] = value


def load_config_from_env():
    """
    Load configuration from environment variables.
    Environment variables should be in format: SECTION_KEY (e.g., MINIMAX_DEPTH)
    """
    for section in DEFAULT_CONFIG:
        for key in DEFAULT_CONFIG[section]:
            env_var = f"{section.upper()}_{key.upper()}"
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Try to convert to appropriate type
                original_value = DEFAULT_CONFIG[section][key]
                if isinstance(original_value, int):
                    DEFAULT_CONFIG[section][key] = int(env_value)
                elif isinstance(original_value, float):
                    DEFAULT_CONFIG[section][key] = float(env_value)
                elif isinstance(original_value, bool):
                    DEFAULT_CONFIG[section][key] = env_value.lower() in ('true', '1', 'yes')
                else:
                    DEFAULT_CONFIG[section][key] = env_value


# Load configuration from environment on import
load_config_from_env()

