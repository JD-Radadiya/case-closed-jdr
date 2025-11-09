# Training Guide for Case Closed Agents

## Overview

This document explains how to train Minimax and MCTS agents for the Case Closed game, including reward functions, continuous training processes, and parameter tuning guidelines.

## Reward Functions

### Reward Structure

The reward function is designed to optimize agents for winning games while learning good strategic play patterns. It consists of three components:

#### 1. Primary Reward: Win/Loss/Draw (80% weight)

- **Win**: +1.0 (agent survives, opponent dies)
- **Loss**: -1.0 (agent dies, opponent survives)
- **Draw**: 0.0 (both agents die simultaneously)

This is the most important component (80% weight) because winning is the ultimate goal. The high weight ensures agents prioritize winning over other objectives.

#### 2. Survival Time Reward (10% weight)

- Bonus proportional to number of turns survived
- Normalized by maximum game length (200 turns)
- Extra bonus if agent survives longer than opponent
- Helps agents learn to avoid collisions and play safely when needed

#### 3. Territory and Trail Reward (10% weight)

- **Territory control** (60% of this component): Ratio of accessible empty cells
- **Trail length advantage** (40% of this component): How much longer agent's trail is than opponent's
- Encourages agents to control space and build longer trails
- Helps with strategic positioning and area control

### Total Reward Formula

```
total_reward = 0.8 * win_reward + 0.1 * survival_reward + 0.1 * territory_reward
```

### Why These Weights?

- **Win rate (80%)**: Primary objective - agents must learn to win
- **Survival (10%)**: Shaping signal - helps agents learn safe play patterns
- **Territory (10%)**: Shaping signal - helps agents learn strategic positioning

The shaping rewards help agents learn better strategies even when they lose, making training more efficient and leading to better final performance.

## Continuous Training Process

### Step 1: Initial Training

Train agents using self-play:

```bash
# Train Minimax agents
python train.py --agent-type minimax --num-matches 100 --optimization-strategy random_search

# Train MCTS agents
python train.py --agent-type mcts --num-matches 100 --optimization-strategy random_search

# Train both and pit them against each other
python train.py --agent-type both --num-matches 100 --optimization-strategy random_search
```

### Step 2: Evaluate Best Models

The training script automatically saves the best models to `models/best_minimax.json` and `models/best_mcts.json`. These models are loaded automatically by the agents when they run.

### Step 3: Continuous Improvement

1. **Analyze Results**: Check logs in `logs/` directory to see which parameters performed best
2. **Refine Parameter Ranges**: Adjust parameter ranges in `parameter_optimizer.py` based on results
3. **Re-train**: Run training again with refined parameters
4. **Test Against Previous Best**: The new best model will be saved only if it outperforms the previous one

### Step 4: Iterative Refinement

Repeat steps 1-3, gradually:
- Narrowing parameter ranges around best-performing values
- Increasing number of matches for more reliable evaluation
- Testing different optimization strategies (grid search vs random search)

## Parameter Tuning

### Strategy Parameters

All agents have three strategy parameters that control play style:

- **aggressive_weight** (0.0-1.0): Prefers risky moves that corner opponent
- **exploration_weight** (0.0-1.0): Prefers unexplored areas of board
- **safety_weight** (0.0-1.0): Prefers moves that maintain multiple escape routes

These weights are normalized to sum to 1.0, so they represent a probability distribution over strategies.

### Minimax-Specific Parameters

- **depth** (2-5): Search depth for minimax algorithm
  - Higher depth = better play but slower
  - Recommended: 3-4 for training, 4-5 for competition
- **use_transposition_table** (bool): Enable caching of game states
  - Recommended: True (significantly improves performance)
- **tt_max_size** (int): Maximum size of transposition table
  - Recommended: 10000 (balance between memory and cache hit rate)

### MCTS-Specific Parameters

- **simulation_time_ms** (50-200): Time budget for simulations per move
  - Higher time = better play but slower
  - Recommended: 100-150ms for training, 150-200ms for competition
- **exploration_constant** (float): UCB1 exploration constant
  - Recommended: 1.414 (standard value)

### Tuning Guidelines

1. **Start with Defaults**: Use default parameter values as a baseline
2. **Grid Search**: For coarse exploration, use grid search with wide parameter ranges
3. **Random Search**: For fine-tuning, use random search around best-performing values
4. **Evaluate Consistently**: Use same number of matches for fair comparison
5. **Consider Opponent**: Train against different opponent types (minimax vs mcts)

## Logging and Metrics

### Log Files

Training logs are saved to `logs/training_YYYYMMDD_HHMMSS.log` with:
- Training session start/end
- Individual match results
- Batch statistics (win rates, rewards)
- Parameter updates
- Best model updates

### Key Metrics

- **Win Rate**: Percentage of games won (primary metric)
- **Average Reward**: Average reward per game (combines all components)
- **Average Turns**: Average game length (survival metric)
- **Algorithm Stats**: Algorithm-specific statistics (nodes evaluated, simulations run, etc.)

### Interpreting Logs

- **High Win Rate + High Reward**: Good parameter configuration
- **High Win Rate + Low Reward**: Winning but not playing optimally (might be lucky)
- **Low Win Rate + High Reward**: Losing but playing well (opponent might be stronger)
- **Consistent Performance**: Parameter configuration is robust

## Best Practices

### Model Selection

1. **Win Rate First**: Prioritize win rate over average reward
2. **Consistency**: Prefer models with consistent performance across multiple runs
3. **Robustness**: Test best models against different opponents
4. **Trade-offs**: Balance between aggressive and safe play based on opponent

### Training Efficiency

1. **Start Small**: Begin with fewer matches (50-100) for quick iteration
2. **Scale Up**: Increase matches (200-500) for final evaluation
3. **Parallel Training**: Train multiple configurations in parallel if possible
4. **Early Stopping**: Stop training if no improvement after several iterations

### Continuous Training Workflow

1. **Daily Training**: Run training daily to continuously improve
2. **Version Control**: Keep track of model versions and their performance
3. **A/B Testing**: Compare new models against current best before replacing
4. **Documentation**: Document parameter changes and their effects

## Example Training Session

```bash
# 1. Train Minimax agents
python train.py --agent-type minimax --num-matches 100 --optimization-strategy random_search

# 2. Check logs
tail -f logs/training_*.log

# 3. Evaluate best model
python -c "from model_manager import ModelManager; mm = ModelManager(); print(mm.load_best_model('minimax'))"

# 4. Train MCTS agents
python train.py --agent-type mcts --num-matches 100 --optimization-strategy random_search

# 5. Pit best models against each other
python train.py --agent-type both --num-matches 200

# 6. Analyze results and refine parameters
# Edit parameter_optimizer.py to adjust ranges
# Repeat from step 1
```

## Troubleshooting

### Low Win Rates

- Increase search depth (minimax) or simulation time (mcts)
- Adjust strategy weights (try more aggressive or more safe)
- Train against weaker opponents first

### Slow Training

- Reduce number of matches per configuration
- Use smaller parameter search space
- Reduce search depth or simulation time during training

### Overfitting

- Test models against different opponents
- Use more matches for evaluation
- Regularize parameter ranges

### Memory Issues

- Reduce transposition table size (minimax)
- Reduce simulation time (mcts)
- Clear caches between training runs

## Advanced Topics

### Evolutionary Algorithms

The parameter optimizer supports evolutionary algorithms for fine-tuning:

```python
from parameter_optimizer import ParameterOptimizer

optimizer = ParameterOptimizer(param_ranges)
offspring = optimizer.evolutionary_search(
    initial_configs=[best_configs],
    fitness_scores=[win_rates],
    num_offspring=10,
    mutation_rate=0.1
)
```

### Custom Reward Functions

You can customize reward weights in `reward_functions.py`:

```python
reward = calculate_reward(
    game, agent_id, result,
    win_weight=0.9,      # Increase win weight
    survival_weight=0.05, # Decrease survival weight
    territory_weight=0.05 # Decrease territory weight
)
```

### Multi-Agent Training

Train agents against multiple opponent types:

```python
# Train minimax against mcts
train_minimax_vs_mcts(num_matches=100)

# Train mcts against minimax
train_mcts_vs_minimax(num_matches=100)
```

## Conclusion

Continuous training is key to improving agent performance. By systematically exploring parameter space, evaluating results, and iteratively refining, you can develop highly competitive agents for the Case Closed game.

Remember:
- **Win rate is primary**: Focus on winning games
- **Shaping rewards help**: They guide learning toward better strategies
- **Consistency matters**: Robust models perform well across different opponents
- **Iterate continuously**: Regular training leads to continuous improvement

