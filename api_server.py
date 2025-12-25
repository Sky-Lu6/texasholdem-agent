# api_server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Set, Dict
import sys
import os
from importlib.metadata import version as pkg_version, PackageNotFoundError

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from texasholdem.game.game import TexasHoldEm, GameState, PlayerState
from texasholdem.game.action_type import ActionType
from texasholdem.card import card  # for pretty string conversions

# Import DQN agent
try:
    from texasholdem.rl.dqn_agent import DQNAgent
    from texasholdem.rl.env import HoldemEnv as RLEnv
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False
    print("Warning: DQN agent not available. Using random opponent only.")

# Import hand analyzer
try:
    from texasholdem.rl.hand_analyzer import HandAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    print("Warning: Hand analyzer not available.")


# =========================
# FastAPI app
# =========================
app = FastAPI(title="Texas Hold'em API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Global game instance
# =========================
# Game configuration
BUYIN = 1000
BB = 20
SB = 10
MAX_PLAYERS = 2  # Can be 2-6 for multi-agent

# Game mode: "player" = you play, "spectator" = watch AI vs AI
GAME_MODE = "player"  # Change to "spectator" to watch AI battle

game = TexasHoldEm(buyin=BUYIN, big_blind=BB, small_blind=SB, max_players=MAX_PLAYERS)

# which player is the "human"/React client (only in player mode)
HERO_ID = 0
# In spectator mode, show all cards. In player mode, show only hero's cards
VISIBLE_PLAYERS: Set[int] = {HERO_ID} if GAME_MODE == "player" else set(range(MAX_PLAYERS))

# Track starting chips for current hand to calculate contributions
starting_chips_this_hand: Dict[int, int] = {}

# Hand counter for PGN exports (starts at 1 for first hand)
hand_counter = 1

# Initialize DQN agent if available
dqn_agent = None
rl_env = None
if DQN_AVAILABLE:
    try:
        dqn_agent = DQNAgent()
        rl_env = RLEnv(buyin=BUYIN, big_blind=BB, small_blind=SB, max_players=MAX_PLAYERS)
        print("DQN agent initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize DQN agent: {e}")
        print("Falling back to random opponent.")


# =========================
# Pydantic models
# =========================
class ActionRequest(BaseModel):
    type: str                # "CALL", "FOLD", "CHECK", "RAISE", "ALL_IN"
    total: Optional[int] = None  # amount to "raise to" (your engine semantics)


class RaiseRange(BaseModel):
    min: int
    max: int


class PotView(BaseModel):
    id: int
    amount: int
    eligible_players: List[int]


class PlayerView(BaseModel):
    id: int
    name: str
    state: str
    chips: int
    in_pot: int
    current_bet: int  # Current round bet (for chip stacks on table)
    hand: List[str]
    visible: bool


class AvailableActionsView(BaseModel):
    CALL: bool
    CHECK: bool
    FOLD: bool
    RAISE: bool
    raiseRange: Optional[RaiseRange] = None


class GameStateView(BaseModel):
    players: List[PlayerView]
    board: List[str]
    history: List[str]
    availableActions: AvailableActionsView
    version: str
    isHandRunning: bool
    isGameRunning: bool
    currentPlayer: int
    totalPot: int  # Total chips in all pots
    pots: List[PotView]  # Individual pot breakdown (main + side pots)


# =========================
# Helper: AI opponent (DQN or random fallback)
# =========================
def ai_opponent_action(game: TexasHoldEm):
    """
    Get action from AI opponent (DQN agent if available, else random).

    Returns:
        (ActionType, Optional[int]): action tuple
    """
    # Only use DQN for player 1 in multi-player games (DQN is trained for heads-up)
    current_player_id = game.current_player
    use_dqn = dqn_agent is not None and rl_env is not None and (MAX_PLAYERS == 2 or current_player_id == 1)

    if use_dqn:
        # Use DQN agent
        try:
            # Get state from RL environment
            state = rl_env._obs()
            legal_mask = rl_env.get_legal_actions_mask()

            # Select action
            action_idx = dqn_agent.select_action(state, legal_mask, epsilon=0.0)

            # Map action index to game action
            from texasholdem.rl.env import ACTIONS as RL_ACTIONS
            action = RL_ACTIONS[action_idx]

            # Handle different action types
            if isinstance(action, str):
                # Custom raise sizes - need to cap at available chips
                current_player = game.players[game.current_player]
                player_chips = current_player.chips
                current_bet = game.player_bet_amount(game.current_player)
                max_total = current_bet + player_chips  # Maximum total bet possible

                pot = sum(p.get_total_amount() for p in game.pots)
                to_call = game.chips_to_call(game.current_player)

                if action == "RAISE_MIN":
                    bb = game.big_blind
                    raise_total = max(bb * 2, bb * 4)
                elif action == "RAISE_HALF_POT":
                    raise_total = int(to_call + pot * 0.5)
                elif action == "RAISE_POT":
                    raise_total = int(to_call + pot)
                elif action == "RAISE_2X_POT":
                    raise_total = int(to_call + pot * 2.0)
                else:
                    raise_total = int(to_call + pot)  # Fallback

                # Cap at available chips (convert to all-in if necessary)
                raise_total = min(raise_total, max_total)
                return (ActionType.RAISE, raise_total)
            else:
                # Direct action type (FOLD, CHECK, CALL, ALL_IN)
                # Now CHECK and CALL are separate actions in the 8-action space
                return (action, None)

        except Exception as e:
            print(f"DQN agent error: {e}. Falling back to random.")

    # Fallback: simple random strategy (reliable for all game modes)
    import random

    moves = game.get_available_moves()
    action_types = moves.action_types

    # Simple logic: prefer check > call > fold, occasionally raise
    if ActionType.CHECK in action_types:
        # Free card available
        if ActionType.RAISE in action_types and random.random() < 0.2:
            # 20% of the time, raise instead of checking
            if hasattr(moves, 'raise_range') and moves.raise_range:
                min_raise = moves.raise_range.start
                return (ActionType.RAISE, min_raise)
        return (ActionType.CHECK, None)

    elif ActionType.CALL in action_types:
        # Facing a bet
        rand = random.random()
        if rand < 0.1 and ActionType.RAISE in action_types:
            # 10% raise
            if hasattr(moves, 'raise_range') and moves.raise_range:
                min_raise = moves.raise_range.start
                return (ActionType.RAISE, min_raise)
        elif rand < 0.65:
            # 55% call
            return (ActionType.CALL, None)
        else:
            # 35% fold
            if ActionType.FOLD in action_types:
                return (ActionType.FOLD, None)
            else:
                return (ActionType.CALL, None)

    elif ActionType.FOLD in action_types:
        # Only fold available (shouldn't happen often)
        return (ActionType.FOLD, None)

    else:
        # Fallback to random action
        return moves.sample()


# =========================
# Helper: build JSON state
# =========================
def _card_list_to_strs(cards) -> List[str]:
    # Convert each card to a simple string like "A♠"
    result = []
    for c in cards:
        # Get rank (0-12, where 0=2, 1=3, ..., 12=A)
        rank_int = c.rank
        if rank_int == 12:
            rank_str = "A"
        elif rank_int == 11:
            rank_str = "K"
        elif rank_int == 10:
            rank_str = "Q"
        elif rank_int == 9:
            rank_str = "J"
        elif rank_int == 8:
            rank_str = "10"
        else:
            rank_str = str(rank_int + 2)  # 0->2, 1->3, etc.

        # Get suit (1=spades, 2=hearts, 4=diamonds, 8=clubs)
        suit_int = c.suit
        if suit_int == 1:
            suit_str = "♠"
        elif suit_int == 2:
            suit_str = "♥"
        elif suit_int == 4:
            suit_str = "♦"
        elif suit_int == 8:
            suit_str = "♣"
        else:
            suit_str = "?"

        result.append(rank_str + suit_str)
    return result


def _history_lines() -> List[str]:
    """
    Build a simple textual history from game.hand_history.
    This avoids depending on TextGUI; it's minimal but works for a React UI.
    """
    if game.hand_history is None:
        return []

    # If game is over and no hand is running, don't show history from previous hand
    if not game.is_game_running() or (not game.is_hand_running() and len([p for p in game.players if p.chips > 0]) <= 1):
        return []

    lines: List[str] = []
    h = game.hand_history

    # Preflop + later rounds: preflop, flop, turn, river
    for round_name in ("preflop", "flop", "turn", "river"):
        round_hist = getattr(h, round_name, None)
        if not round_hist:
            continue
        if round_hist.new_cards:
            lines.append(
                f"{round_name.upper()}: {card.card_list_to_pretty_str(round_hist.new_cards)}"
            )
        for action in round_hist.actions:
            val = action.total
            if action.action_type == ActionType.RAISE and val is not None:
                lines.append(
                    f"Player {action.player_id} {action.action_type.name} to {val}"
                )
            else:
                lines.append(
                    f"Player {action.player_id} {action.action_type.name}"
                )

    # Settle info
    if h.settle:
        lines.append("SETTLE:")
        for pot_id, (amount, rank, winners) in h.settle.pot_winners.items():
            lines.append(
                f"Pot {pot_id}: {amount} chips, winners={list(winners)}"
            )

    return lines


def _available_actions_view() -> AvailableActionsView:
    moves = game.get_available_moves()
    ats = set(moves.action_types)

    has_call = ActionType.CALL in ats
    has_check = ActionType.CHECK in ats
    has_fold = ActionType.FOLD in ats
    has_raise = ActionType.RAISE in ats

    raise_range = None
    if has_raise and getattr(moves, "raise_range", None) is not None:
        start = moves.raise_range.start
        stop = moves.raise_range.stop
        raise_range = RaiseRange(min=start, max=stop - 1)

    return AvailableActionsView(
        CALL=has_call,
        CHECK=has_check,
        FOLD=has_fold,
        RAISE=has_raise,
        raiseRange=raise_range,
    )


def _build_state() -> GameStateView:
    players_view: List[PlayerView] = []

    # Calculate total pot from all pots PLUS current round bets
    pot_total = sum(pot.get_total_amount() for pot in game.pots)
    current_round_bets = sum(game.player_bet_amount(p.player_id) for p in game.players)
    total_pot = pot_total + current_round_bets

    for p in game.players:
        hand_cards = game.get_hand(p.player_id)
        # Always send cards to frontend, let frontend decide visibility
        visible = p.player_id in VISIBLE_PLAYERS or not game.is_hand_running()
        hand_strs = _card_list_to_strs(hand_cards)  # Always send hand data

        # Calculate total contribution this hand
        # This is: starting_chips - current_chips + current_round_bets
        starting_chips = starting_chips_this_hand.get(p.player_id, p.chips)
        chips_spent = starting_chips - p.chips
        current_round_bet = game.player_bet_amount(p.player_id)
        total_contribution = chips_spent + current_round_bet

        print(f"DEBUG: Player {p.player_id} - starting={starting_chips}, current={p.chips}, spent={chips_spent}, current_round={current_round_bet}, total={total_contribution}")

        players_view.append(
            PlayerView(
                id=p.player_id,
                name=f"Player {p.player_id}",
                state=p.state.name if isinstance(p.state, PlayerState) else str(p.state),
                chips=p.chips,
                in_pot=total_contribution,  # Total contribution this hand
                current_bet=current_round_bet,  # Current round bet
                hand=hand_strs,
                visible=visible,
            )
        )

    try:
        version_str = pkg_version("texasholdem")
    except PackageNotFoundError:
        version_str = "dev"

    # Build pot information (main pot + side pots)
    pots_view: List[PotView] = []
    for idx, pot in enumerate(game.pots):
        eligible_players = list(pot.players_in_pot())
        pots_view.append(PotView(
            id=idx,  # Use index as pot ID (0=main, 1+=side pots)
            amount=pot.get_total_amount(),
            eligible_players=eligible_players
        ))

    return GameStateView(
        players=players_view,
        board=_card_list_to_strs(game.board),
        history=_history_lines(),
        availableActions=_available_actions_view(),
        version=version_str,
        isHandRunning=game.is_hand_running(),
        isGameRunning=game.is_game_running(),
        currentPlayer=game.current_player,
        totalPot=total_pot,
        pots=pots_view,
    )


def _maybe_start_hand():
    """
    If the game is running and no hand is active, start a new hand.
    Also exports the previous hand's history to a PGN file.
    """
    global starting_chips_this_hand, hand_counter

    # Ensure pgns directory exists
    os.makedirs("./pgns", exist_ok=True)

    print(f"DEBUG _maybe_start_hand: is_game_running={game.is_game_running()}, is_hand_running={game.is_hand_running()}")

    if game.is_game_running() and not game.is_hand_running():
        # Check if there are enough active players to continue
        active_players = [p for p in game.players if p.chips > 0]
        print(f"DEBUG: Active players: {len(active_players)}")

        if len(active_players) <= 1:
            # Game is effectively over, don't start a new hand
            # But still export the final hand if it exists
            print(f"DEBUG: Game over, hand_history is None: {game.hand_history is None}")
            if game.hand_history is not None:
                try:
                    pgn_path = f"./pgns/hand_{hand_counter:04d}.pgn"
                    game.export_history(pgn_path)
                    print(f"Exported final hand history to {pgn_path}")
                    hand_counter += 1
                except Exception as e:
                    print(f"Failed to export final hand history: {e}")
            return

        # Export previous hand history if it exists
        # (On very first call, hand_history will be None)
        print(f"DEBUG: Before starting new hand, hand_history is None: {game.hand_history is None}, hand_counter={hand_counter}")
        if game.hand_history is not None:
            try:
                # Export with the current counter (represents the hand that just finished)
                pgn_path = f"./pgns/hand_{hand_counter:04d}.pgn"
                print(f"DEBUG: Attempting to export to {pgn_path}")
                game.export_history(pgn_path)
                print(f"Exported hand history to {pgn_path}")
                # Increment counter after export so next hand gets next number
                hand_counter += 1
            except Exception as e:
                print(f"Failed to export hand history: {e}")
                import traceback
                traceback.print_exc()
                # Still increment counter even if export failed
                hand_counter += 1

        # Track starting chips before hand starts
        starting_chips_this_hand = {p.player_id: p.chips for p in game.players}
        print(f"DEBUG: Starting hand {hand_counter}")
        game.start_hand()


def _progress_opponents():
    """
    Let AI players act until:
    - In player mode: it's hero's turn or hand is over
    - In spectator mode: hand is over (all players are AI)
    """
    max_actions = 1000  # Safety limit to prevent infinite loops
    action_count = 0

    try:
        if GAME_MODE == "spectator":
            # In spectator mode, play entire hand automatically
            while game.is_game_running() and game.is_hand_running() and action_count < max_actions:
                print(f"[SPECTATOR] Current player: {game.current_player}, Hand running: {game.is_hand_running()}")

                # Sync RL environment state if using DQN
                if rl_env is not None:
                    rl_env.game = game

                action_type, total = ai_opponent_action(game)
                print(f"[SPECTATOR] Player {game.current_player} taking action: {action_type}, total: {total}")
                game.take_action(action_type, total=total)
                action_count += 1
        else:
            # In player mode, let AI act until it's hero's turn
            print(f"[PLAYER MODE] Starting _progress_opponents: current_player={game.current_player}, HERO_ID={HERO_ID}, hand_running={game.is_hand_running()}")

            while game.is_game_running() and game.is_hand_running() and game.current_player != HERO_ID and action_count < max_actions:
                print(f"[AI TURN] Current player: {game.current_player}, HERO_ID: {HERO_ID}")

                # Sync RL environment state if using DQN
                if rl_env is not None:
                    rl_env.game = game

                action_type, total = ai_opponent_action(game)
                print(f"[AI ACTION] Player {game.current_player} taking action: {action_type}, total: {total}")
                game.take_action(action_type, total=total)
                action_count += 1

            print(f"[PLAYER MODE] Exited loop: current_player={game.current_player}, HERO_ID={HERO_ID}, hand_running={game.is_hand_running()}, action_count={action_count}")

        if action_count >= max_actions:
            print(f"WARNING: Hit max action limit ({max_actions}) in _progress_opponents")

    except Exception as e:
        print(f"ERROR in _progress_opponents: {e}")
        import traceback
        traceback.print_exc()


# =========================
# Endpoints
# =========================
@app.get("/state", response_model=GameStateView)
def get_state():
    _maybe_start_hand()
    _progress_opponents()
    state = _build_state()

    # Debug logging
    print(f"[GET /state] Current player: {game.current_player}, HERO_ID: {HERO_ID}, Hand running: {game.is_hand_running()}")
    print(f"[GET /state] Players alive: {[p.player_id for p in game.players if p.chips > 0]}")

    return state


@app.post("/action", response_model=GameStateView)
def post_action(req: ActionRequest):
    if not game.is_game_running():
        raise HTTPException(status_code=400, detail="Game is not running")

    _maybe_start_hand()

    # You’re always acting as HERO_ID here
    if game.current_player != HERO_ID:
        # If it's somehow not your turn, progress opponents first.
        _progress_opponents()
        if game.current_player != HERO_ID:
            raise HTTPException(status_code=400, detail="Not hero's turn")

    # Convert string to ActionType
    try:
        action_type = ActionType[req.type.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown action type: {req.type}")

    # For RAISE / ALL_IN we expect a 'total' amount (raise *to*)
    total = req.total

    if action_type == ActionType.RAISE and total is None:
        raise HTTPException(status_code=400, detail="RAISE requires 'total' amount")

    # Let the engine validate & execute
    try:
        game.take_action(action_type, total=total)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Now opponents may act
    _progress_opponents()

    return _build_state()


@app.post("/new-hand", response_model=GameStateView)
def new_hand():
    """
    Force a new hand (e.g. after the previous one ended).
    """
    if game.is_hand_running():
        raise HTTPException(status_code=400, detail="Hand already running")

    if game.game_state != GameState.RUNNING:
        raise HTTPException(status_code=400, detail="Game is not RUNNING")

    game.start_hand()
    return _build_state()


@app.post("/reset-game", response_model=GameStateView)
def reset_game():
    """
    Reset the entire game (restart with fresh chip stacks).
    """
    global game, rl_env, starting_chips_this_hand, VISIBLE_PLAYERS, hand_counter

    # Clear PGN directory when resetting
    import shutil
    if os.path.exists("./pgns"):
        shutil.rmtree("./pgns")
    os.makedirs("./pgns", exist_ok=True)

    # Reset hand counter
    hand_counter = 1

    # Rebuild the game with fresh chip stacks
    game = TexasHoldEm(buyin=BUYIN, big_blind=BB, small_blind=SB, max_players=MAX_PLAYERS)

    # Reset RL environment if using DQN
    if rl_env is not None:
        rl_env = RLEnv(buyin=BUYIN, big_blind=BB, small_blind=SB, max_players=MAX_PLAYERS)

    # Reset chip tracking
    starting_chips_this_hand = {}

    # Update visible players based on mode
    VISIBLE_PLAYERS = {HERO_ID} if GAME_MODE == "player" else set(range(MAX_PLAYERS))

    # Start a new hand
    _maybe_start_hand()
    _progress_opponents()

    return _build_state()


@app.get("/game-mode")
def get_game_mode():
    """Get current game mode and configuration."""
    return {
        "mode": GAME_MODE,
        "maxPlayers": MAX_PLAYERS,
        "heroId": HERO_ID if GAME_MODE == "player" else None,
        "buyin": BUYIN,
        "bigBlind": BB,
        "smallBlind": SB
    }


@app.post("/set-mode")
def set_game_mode(mode: str, max_players: int = 2):
    """
    Set game mode and number of players.

    Args:
        mode: "player" or "spectator"
        max_players: 2-6 players
    """
    global GAME_MODE, MAX_PLAYERS, game, rl_env, starting_chips_this_hand, VISIBLE_PLAYERS, hand_counter

    if mode not in ["player", "spectator"]:
        raise HTTPException(status_code=400, detail="Mode must be 'player' or 'spectator'")

    if max_players < 2 or max_players > 6:
        raise HTTPException(status_code=400, detail="max_players must be between 2 and 6")

    GAME_MODE = mode
    MAX_PLAYERS = max_players

    # Clear PGN directory when changing modes
    import shutil
    if os.path.exists("./pgns"):
        shutil.rmtree("./pgns")
    os.makedirs("./pgns", exist_ok=True)

    # Reset hand counter
    hand_counter = 1

    # Recreate game with new settings
    game = TexasHoldEm(buyin=BUYIN, big_blind=BB, small_blind=SB, max_players=MAX_PLAYERS)

    if rl_env is not None:
        rl_env = RLEnv(buyin=BUYIN, big_blind=BB, small_blind=SB, max_players=MAX_PLAYERS)

    starting_chips_this_hand = {}

    # Update visible players
    VISIBLE_PLAYERS = {HERO_ID} if GAME_MODE == "player" else set(range(MAX_PLAYERS))

    # Start first hand
    _maybe_start_hand()
    _progress_opponents()

    return {
        "mode": GAME_MODE,
        "maxPlayers": MAX_PLAYERS,
        "message": f"Game mode set to {mode} with {max_players} players"
    }


@app.get("/analytics/hand-history")
def get_hand_history_analytics():
    """
    Get hand history analytics from PGN files.

    Returns statistics including:
    - Total hands played
    - Player statistics (win rates, actions)
    - Action frequency distribution
    - Pot size distribution
    """
    if not ANALYZER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Hand analyzer not available")

    try:
        analyzer = HandAnalyzer("./pgns")
        stats = analyzer.analyze_directory()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")