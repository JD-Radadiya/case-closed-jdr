"""
Reward functions for training agents.
Primary reward: Win/Loss/Draw
Shaping rewards: Survival time, territory control, trail length advantage
"""

from typing import Dict, Tuple
from case_closed_game import Game, GameResult, Agent
from heuristics import evaluate_territory_control, evaluate_trail_length_advantage


def calculate_reward(game: Game, agent_id: int, result: GameResult,
                    win_weight: float = 0.8,
                    survival_weight: float = 0.1,
                    territory_weight: float = 0.1) -> float:
    """
    Calculate reward for an agent based on game outcome and performance.
    
    Reward Ratio:
    - Primary (80%): Win/Loss/Draw outcome
    - Survival (10%): How long the agent survived
    - Territory (10%): Territory control and trail length advantage
    
    Args:
        game: Final game state
        agent_id: ID of agent to calculate reward for (1 or 2)
        result: Game result (AGENT1_WIN, AGENT2_WIN, or DRAW)
        win_weight: Weight for win/loss reward (default 0.8)
        survival_weight: Weight for survival time reward (default 0.1)
        territory_weight: Weight for territory/trail reward (default 0.1)
    
    Returns:
        Total reward score
    """
    agent = game.agent1 if agent_id == 1 else game.agent2
    other_agent = game.agent2 if agent_id == 1 else game.agent1
    
    # Normalize weights to sum to 1.0
    total_weight = win_weight + survival_weight + territory_weight
    if total_weight > 0:
        win_weight /= total_weight
        survival_weight /= total_weight
        territory_weight /= total_weight
    
    # 1. Primary reward: Win/Loss/Draw (80% of total)
    win_reward = _calculate_win_reward(result, agent_id)
    
    # 2. Survival time reward (10% of total)
    survival_reward = _calculate_survival_reward(game, agent, other_agent)
    
    # 3. Territory and trail length reward (10% of total)
    territory_reward = _calculate_territory_reward(game, agent, other_agent)
    
    # Combine rewards
    total_reward = (
        win_weight * win_reward +
        survival_weight * survival_reward +
        territory_weight * territory_reward
    )
    
    return total_reward


def _calculate_win_reward(result: GameResult, agent_id: int) -> float:
    """
    Calculate primary win/loss reward.
    Win: +1.0, Loss: -1.0, Draw: 0.0
    """
    if result == GameResult.DRAW:
        return 0.0
    elif (result == GameResult.AGENT1_WIN and agent_id == 1) or \
         (result == GameResult.AGENT2_WIN and agent_id == 2):
        return 1.0
    else:
        return -1.0


def _calculate_survival_reward(game: Game, agent: Agent, other_agent: Agent) -> float:
    """
    Calculate survival time reward.
    Bonus for surviving longer, normalized by maximum possible turns.
    """
    max_turns = 200  # Maximum game length
    turns_survived = game.turns
    
    # Normalize to [0, 1] range
    survival_ratio = min(turns_survived / max_turns, 1.0)
    
    # Bonus if agent survived longer than opponent
    if agent.alive and not other_agent.alive:
        # Agent won by survival
        return 1.0
    elif not agent.alive and other_agent.alive:
        # Agent died before opponent
        return survival_ratio * 0.5  # Partial credit for survival time
    else:
        # Both alive or both dead
        return survival_ratio


def _calculate_territory_reward(game: Game, agent: Agent, other_agent: Agent) -> float:
    """
    Calculate territory control and trail length reward.
    Combines accessible territory and trail length advantage.
    """
    # Territory control (normalized to [0, 1])
    try:
        territory_score = evaluate_territory_control(game, agent, max_depth=8)
    except:
        territory_score = 0.0
    
    # Trail length advantage (normalized to [-1, 1])
    try:
        trail_advantage = evaluate_trail_length_advantage(game, agent)
        # Normalize to [0, 1] for reward
        trail_score = (trail_advantage + 1.0) / 2.0
    except:
        trail_score = 0.5
    
    # Combine territory and trail scores
    territory_reward = 0.6 * territory_score + 0.4 * trail_score
    
    return territory_reward


def calculate_rewards_for_both_agents(game: Game, result: GameResult,
                                     win_weight: float = 0.8,
                                     survival_weight: float = 0.1,
                                     territory_weight: float = 0.1) -> Tuple[float, float]:
    """
    Calculate rewards for both agents.
    
    Returns:
        Tuple of (agent1_reward, agent2_reward)
    """
    reward1 = calculate_reward(game, 1, result, win_weight, survival_weight, territory_weight)
    reward2 = calculate_reward(game, 2, result, win_weight, survival_weight, territory_weight)
    
    return reward1, reward2


def explain_reward_function() -> str:
    """
    Return a detailed explanation of the reward function.
    """
    explanation = """
    Reward Function Explanation
    ===========================
    
    The reward function optimizes agents to:
    1. Win games (primary objective, 80% weight)
    2. Survive longer (shaping reward, 10% weight)
    3. Control territory and build longer trails (shaping reward, 10% weight)
    
    Primary Reward (Win/Loss/Draw) - 80%:
    - Win: +1.0 (agent survives, opponent dies)
    - Loss: -1.0 (agent dies, opponent survives)
    - Draw: 0.0 (both agents die simultaneously)
    
    This is the most important component because winning is the ultimate goal.
    The high weight (80%) ensures agents prioritize winning over other objectives.
    
    Survival Time Reward - 10%:
    - Bonus proportional to number of turns survived
    - Normalized by maximum game length (200 turns)
    - Extra bonus if agent survives longer than opponent
    - Helps agents learn to avoid collisions and play safely when needed
    
    Territory and Trail Reward - 10%:
    - Territory control: Ratio of accessible empty cells (60% of this component)
    - Trail length advantage: How much longer agent's trail is than opponent's (40% of this component)
    - Encourages agents to control space and build longer trails
    - Helps with strategic positioning and area control
    
    Total Reward Formula:
    total_reward = 0.8 * win_reward + 0.1 * survival_reward + 0.1 * territory_reward
    
    Why These Weights?
    - Win rate (80%): Primary objective - agents must learn to win
    - Survival (10%): Shaping signal - helps agents learn safe play patterns
    - Territory (10%): Shaping signal - helps agents learn strategic positioning
    
    The shaping rewards help agents learn better strategies even when they lose,
    making training more efficient and leading to better final performance.
    """
    return explanation

