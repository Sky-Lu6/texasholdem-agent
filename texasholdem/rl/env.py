# texasholdem/rl/env.py
import numpy as np
from texasholdem import TexasHoldEm, ActionType
from texasholdem.game.hand_phase import HandPhase
from texasholdem.card.card import Card
from texasholdem.rl.win_probability import quick_win_probability
import random

# Define multiple raise sizes for richer action space
# Index 0-2: basic actions, 3-7: different raise sizes
ACTIONS = [
    ActionType.FOLD,      # 0
    ActionType.CHECK,     # 1: pass action when no bet (NEW - separated from CALL)
    ActionType.CALL,      # 2: match current bet
    "RAISE_MIN",          # 3: minimum legal raise
    "RAISE_HALF_POT",     # 4: raise to 0.5x pot
    "RAISE_POT",          # 5: raise to 1x pot
    "RAISE_2X_POT",       # 6: raise to 2x pot
    ActionType.ALL_IN     # 7: all-in
]

# Card encoding constants
NUM_RANKS = 13  # 2-A
NUM_SUITS = 4   # spades, hearts, diamonds, clubs
NUM_CARDS = 52  # 13 * 4

class HoldemEnv:
    HERO_ID = 0

    def __init__(self, **kw):
        defaults = dict(buyin=200, big_blind=5, small_blind=2, max_players=2)
        defaults.update(kw)
        self.game = TexasHoldEm(**defaults)

    def _card_to_index(self, card: Card) -> int:
        """Convert a Card to an index 0-51.

        Card representation: rank (0-12) * 4 + suit_index (0-3)
        where rank 0=2, 1=3, ..., 12=A
        and suit 1=spades(0), 2=hearts(1), 4=diamonds(2), 8=clubs(3)
        """
        # Map suit bit flags (1,2,4,8) to indices (0,1,2,3)
        suit_map = {1: 0, 2: 1, 4: 2, 8: 3}  # spades, hearts, diamonds, clubs
        suit_idx = suit_map.get(card.suit, 0)
        return card.rank * NUM_SUITS + suit_idx

    def _encode_cards(self, cards) -> np.ndarray:
        """Encode a list of cards as a 52-dimensional binary vector.

        Args:
            cards: List of Card objects

        Returns:
            52-dimensional numpy array with 1s at positions of present cards
        """
        encoding = np.zeros(NUM_CARDS, dtype=np.float32)
        for card in cards:
            idx = self._card_to_index(card)
            encoding[idx] = 1.0
        return encoding

    def _encode_hand_phase(self) -> np.ndarray:
        """Encode current hand phase as one-hot vector.

        Returns:
            4-dimensional one-hot vector [preflop, flop, turn, river]
        """
        phase_encoding = np.zeros(4, dtype=np.float32)
        phase = self.game.hand_phase

        if phase == HandPhase.PREFLOP:
            phase_encoding[0] = 1.0
        elif phase == HandPhase.FLOP:
            phase_encoding[1] = 1.0
        elif phase == HandPhase.TURN:
            phase_encoding[2] = 1.0
        elif phase == HandPhase.RIVER:
            phase_encoding[3] = 1.0
        # PREHAND and SETTLE don't happen during agent decisions

        return phase_encoding

    def _advance_to_hero_or_terminal(self):
        """让非我方玩家一直行动，直到轮到我方或终局。"""
        while self.game.is_hand_running() and self.game.current_player != self.HERO_ID:
            try:
                to_call = float(self.game.chips_to_call(self.game.current_player))
            except Exception:
                to_call = 0.0

            try:
                pot = float(sum(p.get_total_amount() for p in self.game.pots))
            except Exception:
                pot = 0.0

            bb = float(self.game.big_blind)

            if to_call > 0 and self._is_legal(ActionType.CALL):
                # 面对下注：大多跟注，少量加注（使用不同加注大小）
                if self._is_legal(ActionType.RAISE) and random.random() < 0.15:
                    try:
                        # Randomly choose raise size
                        choice = random.random()
                        if choice < 0.4:
                            # Min raise
                            raise_total = max(bb * 2, bb * 4)
                        elif choice < 0.7:
                            # Half pot raise
                            raise_total = to_call + pot * 0.5
                        else:
                            # Pot raise
                            raise_total = to_call + pot
                        self.game.take_action(ActionType.RAISE, total=int(raise_total))
                    except Exception:
                        self.game.take_action(ActionType.CALL)
                else:
                    self.game.take_action(ActionType.CALL)
            elif self._is_legal(ActionType.RAISE) and random.random() < 0.25:
                # 无需跟注：偶尔先手加注
                try:
                    # Randomly choose raise size
                    choice = random.random()
                    if choice < 0.5:
                        # Min raise
                        raise_total = max(bb * 2, bb * 4)
                    else:
                        # Half pot raise
                        raise_total = pot * 0.5
                    self.game.take_action(ActionType.RAISE, total=int(raise_total))
                except Exception:
                    if self._is_legal(ActionType.CHECK):
                        self.game.take_action(ActionType.CHECK)
            elif self._is_legal(ActionType.CHECK):
                self.game.take_action(ActionType.CHECK)
            else:
                break

    def reset(self):
        # Always rebuild the game to reset chip stacks
        # This prevents the issue where a player runs out of chips
        self.game = type(self.game)(
            buyin=self.game.buyin,
            big_blind=self.game.big_blind,
            small_blind=self.game.small_blind,
            max_players=self.game.max_players
        )
        self.game.start_hand()
        self._advance_to_hero_or_terminal()

        # 记录"本手开局"时我方筹码（用于终局奖励）
        self._stack_start = float(self.game.players[self.HERO_ID].chips)

        # Initialize win probability tracking
        self._prev_win_prob = self._calculate_win_probability()

        return self._obs()

    def _is_legal(self, action_type):
        """Return True if current player can legally perform this action."""
        try:
            self.game.validate_move(action=action_type, throws=True)
            return True
        except Exception:
            return False

    def get_legal_actions_mask(self):
        """Get a binary mask indicating which actions are legal.

        Returns:
            numpy array of shape (8,) with 1.0 for legal actions, 0.0 for illegal
        """
        mask = np.zeros(len(ACTIONS), dtype=np.float32)

        # Check basic actions (FOLD, CHECK, CALL, ALL_IN)
        for i, action in enumerate(ACTIONS):
            if action in [ActionType.FOLD, ActionType.CHECK, ActionType.CALL, ActionType.ALL_IN]:
                if self._is_legal(action):
                    mask[i] = 1.0
            else:
                # For custom raise sizes, check if raising is possible
                # by validating a minimal raise amount
                try:
                    bb = int(self.game.big_blind)
                    min_raise = bb * 2  # Minimum raise is typically 2*BB
                    # Try to validate a minimal raise
                    self.game.validate_move(action=ActionType.RAISE, total=min_raise, throws=True)
                    mask[i] = 1.0  # If it doesn't throw, raising is legal
                except Exception:
                    mask[i] = 0.0  # Raising not allowed

        # Ensure at least one action is legal (fallback safety check)
        if mask.sum() == 0:
            if self._is_legal(ActionType.FOLD):
                mask[0] = 1.0
            elif self._is_legal(ActionType.CHECK):
                mask[1] = 1.0
            elif self._is_legal(ActionType.CALL):
                mask[2] = 1.0

        return mask

    def _calculate_win_probability(self) -> float:
        """Calculate current win probability for the hero.

        Returns:
            Win probability between 0.0 and 1.0
        """
        hero_hand = self.game.get_hand(self.HERO_ID)
        if not hero_hand or len(hero_hand) != 2:
            return 0.0

        board = self.game.board
        num_opponents = len([p for p in self.game.players
                            if p.player_id != self.HERO_ID
                            and p.state not in (self.game.players[0].state.__class__.OUT,
                                               self.game.players[0].state.__class__.SKIP)])

        # Use quick estimation (100 simulations)
        return quick_win_probability(hero_hand, board, num_opponents=max(1, num_opponents))

    def step(self, a_idx: int):
        # Hand already ended - this shouldn't happen in normal training
        # If called after done=True, just return terminal state with 0 reward
        if not self.game.is_hand_running():
            return self._obs(), 0.0, True, {"terminal": True}

        action = ACTIONS[a_idx]

        # Save win probability BEFORE action (for reward calculation)
        prev_win_prob = self._prev_win_prob

        # Save chips before action for penalty calculation
        chips_before = float(self.game.players[self.HERO_ID].chips)

        # Execute the action with proper raise sizing
        if action in [ActionType.FOLD, ActionType.CALL, ActionType.CHECK, ActionType.ALL_IN]:
            # Direct actions without sizing - just execute them
            self.game.take_action(action)
        else:
            # Custom raise sizes (RAISE_MIN, RAISE_HALF_POT, RAISE_POT, RAISE_2X_POT)
            try:
                pot = float(sum(p.get_total_amount() for p in self.game.pots))
            except Exception:
                pot = 0.0

            hero_chips = float(self.game.players[self.HERO_ID].chips)
            current_bet = float(self.game.player_bet_amount(self.HERO_ID))
            to_call = float(self.game.chips_to_call(self.HERO_ID))

            # Use game's min_raise() method to get the correct minimum raise amount
            min_raise_value = float(self.game.min_raise())

            # Calculate raise VALUE (how much we're raising by), then convert to TOTAL
            if action == "RAISE_MIN":
                # Minimum legal raise
                raise_value = min_raise_value
            elif action == "RAISE_HALF_POT":
                # Raise by half pot (after calling)
                raise_value = max(pot * 0.5, min_raise_value)
            elif action == "RAISE_POT":
                # Raise by pot size (after calling)
                raise_value = max(pot, min_raise_value)
            elif action == "RAISE_2X_POT":
                # Raise by 2x pot (after calling)
                raise_value = max(pot * 2.0, min_raise_value)
            else:
                # Fallback to min raise
                raise_value = min_raise_value

            # Convert raise VALUE to TOTAL using game's method
            raise_total = self.game.value_to_total(int(raise_value), self.HERO_ID)

            # Cap at hero's stack (convert to all-in if necessary)
            max_total = current_bet + hero_chips
            raise_total = min(raise_total, max_total)

            # Execute raise
            self.game.take_action(ActionType.RAISE, total=int(raise_total))

        # 推进到我方回合/或牌局结束
        self._advance_to_hero_or_terminal()

        # Calculate reward as change in win probability
        done = not self.game.is_hand_running()

        if done:
            # Terminal state: use actual outcome based on chip change
            chips_now = float(self.game.players[self.HERO_ID].chips)
            chips_start = getattr(self, "_stack_start", chips_now)
            chip_delta = chips_now - chips_start

            # Use normalized chip change as terminal reward
            # Positive if we won, negative if we lost
            bb = max(1.0, float(self.game.big_blind))
            reward = chip_delta / (bb * 10.0)  # Normalize by 10BB

            # Bonus/penalty based on outcome
            if chip_delta > 0:
                # Won the hand
                current_win_prob = 1.0
                # Small bonus for winning
                reward += 0.1
            elif chip_delta < 0:
                # Lost the hand
                current_win_prob = 0.0
                # Penalty for losing, scaled by how much we lost
                # Bigger losses get bigger penalties (discourage reckless all-ins)
                loss_ratio = abs(chip_delta) / max(1.0, chips_start)
                if loss_ratio > 0.5:  # Lost more than 50% of stack
                    reward -= 0.3  # Heavy penalty for big losses
                else:
                    reward -= 0.1  # Mild penalty for small losses
            else:
                # Tied (rare, but possible)
                current_win_prob = 0.5
                reward = 0.0
        else:
            # Non-terminal state: estimate win probability
            current_win_prob = self._calculate_win_probability()

            # Base reward is change in win probability
            reward = (current_win_prob - prev_win_prob) * 0.5  # Scale down to 0.5

            # Add penalty for reckless all-in behavior
            if action == ActionType.ALL_IN:
                # All-in penalty based on win probability
                # If win prob is low (<30%), heavily penalize all-in
                if current_win_prob < 0.3:
                    reward -= 1.0  # VERY heavy penalty for all-in with weak hand
                elif current_win_prob < 0.5:
                    reward -= 0.5  # Heavy penalty for all-in with marginal hand
                elif current_win_prob < 0.7:
                    reward -= 0.2  # Moderate penalty for all-in with decent hand
                # Only allow all-in with very strong hands (>70% win prob)

            # Reward good raise sizing with appropriate hands
            # DRAMATICALLY INCREASED to overcome check/call passivity
            if action in ["RAISE_MIN", "RAISE_HALF_POT"]:
                # Small raises are good with medium-strength hands
                if 0.35 <= current_win_prob < 0.7:
                    reward += 0.6  # MASSIVE reward for small raises (was 0.3)
                elif current_win_prob >= 0.7:
                    reward += 0.5  # Strong reward with strong hands (was 0.25)
                elif 0.25 <= current_win_prob < 0.35:
                    reward += 0.3  # Even weak raises are rewarded (was 0.1)
            elif action in ["RAISE_POT"]:
                # Pot-sized raises are good with strong hands
                if current_win_prob >= 0.55:
                    reward += 0.7  # MASSIVE reward for pot raises with good hands (was 0.35)
                elif current_win_prob >= 0.4:
                    reward += 0.4  # Strong reward for pot raises with ok hands (was 0.2)
                elif current_win_prob < 0.25:
                    reward -= 0.15  # Penalty for pot raises with very weak hands
            elif action in ["RAISE_2X_POT"]:
                # Big raises should only be with very strong hands
                if current_win_prob >= 0.65:
                    reward += 0.6  # MASSIVE reward for big raises with strong hands (was 0.3)
                elif current_win_prob >= 0.5:
                    reward += 0.3  # Strong reward with medium hands (was 0.15)
                elif current_win_prob < 0.3:
                    reward -= 0.3  # Penalty for big raises with weak hands

            # Minimal check reward to encourage more aggressive play
            if action == ActionType.CHECK:
                # Drastically reduced to make raises more attractive
                if current_win_prob < 0.3:
                    reward += 0.05  # Small reward for checking weak hands
                elif current_win_prob >= 0.6:
                    reward -= 0.15  # PENALTY for checking strong hands (missing value!)
                else:
                    reward += 0.02  # Tiny reward for checking medium hands (was 0.1)
            elif action == ActionType.CALL:
                # Reward calling based on pot odds and hand strength
                try:
                    to_call = float(self.game.chips_to_call(self.HERO_ID))
                    pot = float(sum(p.get_total_amount() for p in self.game.pots))
                    pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0

                    # If pot odds are good, reward calling
                    if current_win_prob > pot_odds:
                        reward += 0.15  # Good call based on pot odds
                    elif current_win_prob > pot_odds * 0.7:
                        reward += 0.05  # Marginal call
                except:
                    if current_win_prob > 0.3:
                        reward += 0.1  # Fallback: reward calls with decent hands

            # Reduce fold rewards (only reward very good folds)
            if action == ActionType.FOLD:
                try:
                    to_call = float(self.game.chips_to_call(self.HERO_ID))
                    # Only reward fold if hand is very weak AND call is expensive
                    if current_win_prob < 0.2 and to_call > chips_before * 0.2:
                        reward += 0.05  # Small bonus for very good fold
                    elif current_win_prob > 0.35:
                        # Penalize folding decent hands
                        reward -= 0.1
                except:
                    pass

        # Update previous win probability for next step
        self._prev_win_prob = current_win_prob

        return self._obs(), reward, done, {}

    def _obs(self):
        """Build observation state.

        State components:
        - Hero's hole cards: 52-dim binary vector
        - Board cards: 52-dim binary vector
        - Hand phase: 4-dim one-hot vector
        - Pot size (normalized by BB): 1 scalar
        - Chips to call (normalized by BB): 1 scalar
        - Hero stack (normalized by BB): 1 scalar
        - Opponent stack (normalized by BB): 1 scalar
        - Pot odds: 1 scalar (chips_to_call / (pot + chips_to_call))
        - Stack-to-pot ratio (SPR): 1 scalar (hero_stack / pot)
        - Position indicator: 1 scalar (1.0 if in position, 0.0 otherwise)
        - Number of raises this round: 1 scalar (normalized)

        Total: 52 + 52 + 4 + 8 = 116 dimensions
        """
        # Get hero's hole cards
        hero_hand = self.game.get_hand(self.HERO_ID)
        hero_cards_encoded = self._encode_cards(hero_hand)

        # Get board cards
        board_cards_encoded = self._encode_cards(self.game.board)

        # Get hand phase
        phase_encoded = self._encode_hand_phase()

        # Get basic numerical features
        bb = max(1.0, float(self.game.big_blind))

        try:
            pot = float(sum(p.get_total_amount() for p in self.game.pots))
        except Exception:
            pot = 0.0

        try:
            to_call = float(self.game.chips_to_call(self.game.current_player))
        except Exception:
            to_call = 0.0

        try:
            hero_stack = float(self.game.players[self.HERO_ID].chips)
            opp_stack = float(self.game.players[1 - self.HERO_ID].chips)
        except Exception:
            hero_stack = 0.0
            opp_stack = 0.0

        # Normalize by big blind
        pot_bb = pot / bb
        to_call_bb = to_call / bb
        hero_stack_bb = hero_stack / bb
        opp_stack_bb = opp_stack / bb

        # Calculate pot odds
        if to_call > 0 and pot + to_call > 0:
            pot_odds = to_call / (pot + to_call)
        else:
            pot_odds = 0.0

        # Calculate stack-to-pot ratio (SPR)
        if pot > 0:
            spr = hero_stack / pot
        else:
            spr = 100.0  # Large value when pot is empty
        spr = min(spr, 100.0)  # Cap at 100 to prevent huge values

        # Position indicator (are we acting last this round?)
        # In heads-up: button acts last postflop, BB acts last preflop
        in_position = 0.0
        if self.game.hand_phase == HandPhase.PREFLOP:
            # BB is in position preflop
            in_position = 1.0 if self.game.current_player == self.game.bb_loc else 0.0
        else:
            # Button is in position postflop
            in_position = 1.0 if self.game.current_player == self.game.btn_loc else 0.0

        # Count raises this round (use betting history)
        num_raises = 0.0
        if self.game.hand_history and self.game.hand_phase != HandPhase.PREHAND:
            # Get the betting history for current phase
            phase_attr_map = {
                HandPhase.PREFLOP: 'preflop',
                HandPhase.FLOP: 'flop',
                HandPhase.TURN: 'turn',
                HandPhase.RIVER: 'river'
            }
            phase_attr = phase_attr_map.get(self.game.hand_phase)
            if phase_attr:
                betting_history = getattr(self.game.hand_history, phase_attr, None)
                if betting_history and hasattr(betting_history, 'actions'):
                    num_raises = sum(1 for action in betting_history.actions
                                   if action.action_type == ActionType.RAISE)
        num_raises_norm = min(num_raises / 5.0, 1.0)  # Normalize, cap at 5 raises

        # Combine all features
        poker_features = np.array([
            pot_bb,
            to_call_bb,
            hero_stack_bb,
            opp_stack_bb,
            pot_odds,
            spr,
            in_position,
            num_raises_norm
        ], dtype=np.float32)

        # Concatenate all features
        obs = np.concatenate([
            hero_cards_encoded,    # 52
            board_cards_encoded,   # 52
            phase_encoded,         # 4
            poker_features         # 8
        ])

        return obs

