"""
Sample agent for Case Closed Challenge - Works with Judge Protocol
This agent runs as a Flask server and responds to judge requests.
"""

import os
from flask import Flask, request, jsonify
from collections import deque
from threading import Lock

from case_closed_game import Game, Direction, GameResult
from mcts_agent import MCTSAgent
from model_manager import ModelManager

app = Flask(__name__)

# Basic identity
PARTICIPANT = os.getenv("PARTICIPANT", "SampleParticipant")
AGENT_NAME = os.getenv("AGENT_NAME", "SampleAgent")

# Track game state
GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
game_lock = Lock()

# Initialize MCTS agent
_mcts_agent = None
_model_manager = ModelManager(models_dir="models")


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge."""
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


def _get_mcts_agent(player_number: int):
    """Get or create MCTS agent for the player."""
    global _mcts_agent
    
    if _mcts_agent is None:
        # Try to load best model, or use defaults
        model_config = _model_manager.load_best_model('mcts')
        if model_config is None:
            # Use default configuration
            _mcts_agent = MCTSAgent(
                agent_id=player_number,
                simulation_time_ms=120,
                aggressive_weight=0.33,
                exploration_weight=0.33,
                safety_weight=0.34,
                exploration_constant=1.414
            )
        else:
            # Load from model configuration
            _mcts_agent = MCTSAgent(
                agent_id=player_number,
                simulation_time_ms=model_config.get('simulation_time_ms', 120),
                aggressive_weight=model_config.get('aggressive_weight', 0.33),
                exploration_weight=model_config.get('exploration_weight', 0.33),
                safety_weight=model_config.get('safety_weight', 0.34),
                exploration_constant=model_config.get('exploration_constant', 1.414)
            )
    
    # Update agent_id if player number changed
    if _mcts_agent.agent_id != player_number:
        _mcts_agent.agent_id = player_number
    
    return _mcts_agent


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    
    # Update our local game state
    _update_local_game_from_post(data)
    
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    """
    player_number = request.args.get("player_number", default=1, type=int)
    
    with game_lock:
        state = dict(LAST_POSTED_STATE)
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining
    
    # -----------------MCTS agent decision-------------------
    try:
        # Get MCTS agent
        mcts_agent = _get_mcts_agent(player_number)
        
        # Update agent_id
        with game_lock:
            mcts_agent.agent_id = player_number
            direction, use_boost = mcts_agent.get_best_move(GLOBAL_GAME)
        
        # Convert Direction to string
        direction_map = {
            Direction.UP: "UP",
            Direction.DOWN: "DOWN",
            Direction.LEFT: "LEFT",
            Direction.RIGHT: "RIGHT"
        }
        move = direction_map.get(direction, "RIGHT")
        
        # Add boost if needed
        if use_boost and boosts_remaining > 0:
            move = f"{move}:BOOST"
    except Exception as e:
        # Fallback to simple strategy if MCTS fails
        print(f"MCTS error: {e}")
        # Use simple decision logic as fallback
        turn_count = state.get("turn_count", 0)
        if player_number == 1:
            my_trail = state.get("agent1_trail", [])
            my_boosts = state.get("agent1_boosts", 3)
            other_trail = state.get("agent2_trail", [])
        else:
            my_trail = state.get("agent2_trail", [])
            my_boosts = state.get("agent2_boosts", 3)
            other_trail = state.get("agent1_trail", [])
        move = decide_move(my_trail, other_trail, turn_count, my_boosts)
    # -----------------end code here--------------------
    
    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state."""
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
        result = data.get("result", "UNKNOWN")
        print(f"\nGame Over! Result: {result}")
    return jsonify({"status": "acknowledged"}), 200


def decide_move(my_trail, other_trail, turn_count, my_boosts):
    """Simple decision logic for the agent.
    
    Strategy:
    - Move in a direction that doesn't immediately hit a trail
    - Use boost if we have them and it's mid-game (turns 30-80)
    """
    if not my_trail:
        return "RIGHT"
    
    # Get current head position and direction
    head = my_trail[-1] if my_trail else (0, 0)
    
    # Calculate current direction if we have at least 2 positions
    current_dir = "RIGHT"
    if len(my_trail) >= 2:
        prev = my_trail[-2]
        dx = head[0] - prev[0]
        dy = head[1] - prev[1]
        
        # Normalize for torus wrapping
        if abs(dx) > 1:
            dx = -1 if dx > 0 else 1
        if abs(dy) > 1:
            dy = -1 if dy > 0 else 1
        
        if dx == 1:
            current_dir = "RIGHT"
        elif dx == -1:
            current_dir = "LEFT"
        elif dy == 1:
            current_dir = "DOWN"
        elif dy == -1:
            current_dir = "UP"
    
    # Simple strategy: try to avoid trails, prefer continuing straight
    # Check available directions (not opposite to current)
    directions = ["UP", "DOWN", "LEFT", "RIGHT"]
    opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    
    # Remove opposite direction
    if current_dir in opposite:
        try:
            directions.remove(opposite[current_dir])
        except ValueError:
            pass
    
    # Prefer current direction if still available
    if current_dir in directions:
        chosen_dir = current_dir
    else:
        # Pick first available
        chosen_dir = directions[0] if directions else "RIGHT"
    
    # Decide whether to use boost
    # Use boost in mid-game when we still have them
    use_boost = my_boosts > 0 and 30 <= turn_count <= 80
    
    if use_boost:
        return f"{chosen_dir}:BOOST"
    else:
        return chosen_dir


if __name__ == "__main__":
    # For development only. Port can be overridden with the PORT env var.
    port = int(os.environ.get("PORT", "5009"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
