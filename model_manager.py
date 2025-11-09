"""
Model manager for saving, loading, and versioning agent models.
"""

import json
import os
from typing import Dict, Optional, List
from datetime import datetime


class ModelManager:
    """
    Manages model saving, loading, and versioning.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self._ensure_models_dir()
    
    def _ensure_models_dir(self):
        """Ensure models directory exists."""
        os.makedirs(self.models_dir, exist_ok=True)
    
    def save_model(self, model_config: Dict, model_name: str = None, 
                  is_best: bool = False) -> str:
        """
        Save a model configuration to a JSON file.
        
        Args:
            model_config: Dictionary containing model configuration
                - 'type': 'minimax' or 'mcts'
                - 'aggressive_weight': float
                - 'exploration_weight': float
                - 'safety_weight': float
                - 'depth': int (for minimax)
                - 'simulation_time_ms': int (for mcts)
                - 'metadata': dict (optional, with win_rate, reward, etc.)
            model_name: Name for the model file (without extension)
            is_best: If True, also save as best model
        
        Returns:
            Path to saved model file
        """
        # Generate model name if not provided
        if model_name is None:
            model_type = model_config.get('type', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_{timestamp}"
        
        # Add metadata if not present
        if 'metadata' not in model_config:
            model_config['metadata'] = {}
        
        model_config['metadata']['saved_at'] = datetime.now().isoformat()
        model_config['metadata']['model_name'] = model_name
        
        # Save model file
        model_path = os.path.join(self.models_dir, f"{model_name}.json")
        with open(model_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Save as best model if requested
        if is_best:
            best_model_name = f"best_{model_config['type']}.json"
            best_model_path = os.path.join(self.models_dir, best_model_name)
            with open(best_model_path, 'w') as f:
                json.dump(model_config, f, indent=2)
        
        return model_path
    
    def load_model(self, model_name: str) -> Optional[Dict]:
        """
        Load a model configuration from a JSON file.
        
        Args:
            model_name: Name of the model file (with or without .json extension)
        
        Returns:
            Model configuration dictionary, or None if not found
        """
        # Add .json extension if not present
        if not model_name.endswith('.json'):
            model_name += '.json'
        
        model_path = os.path.join(self.models_dir, model_name)
        
        if not os.path.exists(model_path):
            return None
        
        with open(model_path, 'r') as f:
            return json.load(f)
    
    def load_best_model(self, model_type: str) -> Optional[Dict]:
        """
        Load the best model of a given type.
        
        Args:
            model_type: 'minimax' or 'mcts'
        
        Returns:
            Model configuration dictionary, or None if not found
        """
        best_model_name = f"best_{model_type}.json"
        return self.load_model(best_model_name)
    
    def list_models(self, model_type: str = None) -> List[str]:
        """
        List all saved models.
        
        Args:
            model_type: Filter by model type ('minimax' or 'mcts'), or None for all
        
        Returns:
            List of model names (without .json extension)
        """
        if not os.path.exists(self.models_dir):
            return []
        
        models = []
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.json') and not filename.startswith('best_'):
                try:
                    model_path = os.path.join(self.models_dir, filename)
                    with open(model_path, 'r') as f:
                        model_config = json.load(f)
                    
                    if model_type is None or model_config.get('type') == model_type:
                        models.append(filename[:-5])  # Remove .json extension
                except:
                    continue
        
        return sorted(models)
    
    def get_model_config_for_agent(self, model_name: str, agent_id: int) -> Dict:
        """
        Get agent configuration from a model file.
        Adds agent_id to the configuration.
        
        Args:
            model_name: Name of the model file
            agent_id: ID of the agent (1 or 2)
        
        Returns:
            Agent configuration dictionary
        """
        model_config = self.load_model(model_name)
        if model_config is None:
            raise ValueError(f"Model not found: {model_name}")
        
        # Create agent configuration
        agent_config = {
            'type': model_config['type'],
            'agent_id': agent_id,
            'aggressive_weight': model_config.get('aggressive_weight', 0.33),
            'exploration_weight': model_config.get('exploration_weight', 0.33),
            'safety_weight': model_config.get('safety_weight', 0.34),
        }
        
        # Add type-specific parameters
        if model_config['type'] == 'minimax':
            agent_config['depth'] = model_config.get('depth', 4)
            agent_config['use_transposition_table'] = model_config.get('use_transposition_table', True)
            agent_config['tt_max_size'] = model_config.get('tt_max_size', 10000)
        elif model_config['type'] == 'mcts':
            agent_config['simulation_time_ms'] = model_config.get('simulation_time_ms', 120)
            agent_config['exploration_constant'] = model_config.get('exploration_constant', 1.414)
        
        return agent_config
    
    def update_best_model(self, model_config: Dict, win_rate: float, avg_reward: float):
        """
        Update best model if the new model performs better.
        
        Args:
            model_config: Model configuration
            win_rate: Win rate of the model
            avg_reward: Average reward of the model
        """
        model_type = model_config['type']
        best_model = self.load_best_model(model_type)
        
        should_update = False
        if best_model is None:
            should_update = True
        else:
            # Compare based on win rate first, then average reward
            best_win_rate = best_model.get('metadata', {}).get('win_rate', 0.0)
            best_avg_reward = best_model.get('metadata', {}).get('avg_reward', 0.0)
            
            if win_rate > best_win_rate:
                should_update = True
            elif win_rate == best_win_rate and avg_reward > best_avg_reward:
                should_update = True
        
        if should_update:
            model_config['metadata']['win_rate'] = win_rate
            model_config['metadata']['avg_reward'] = avg_reward
            self.save_model(model_config, is_best=True)
            return True
        
        return False

