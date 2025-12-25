# texasholdem/rl/win_probability.py
"""
Monte Carlo win probability calculator for Texas Hold'em.

This module estimates the probability that a player will win given their hole cards,
the current board, and the current game state.
"""
import random
from typing import List
from texasholdem.card.card import Card
from texasholdem.card.deck import Deck
from texasholdem.evaluator import evaluator


def calculate_win_probability(
    hero_hand: List[Card],
    board: List[Card],
    num_opponents: int = 1,
    num_simulations: int = 1000
) -> float:
    """Calculate win probability using Monte Carlo simulation.

    Args:
        hero_hand: Hero's two hole cards
        board: Community cards (0-5 cards)
        num_opponents: Number of opponents (default 1 for heads-up)
        num_simulations: Number of Monte Carlo simulations to run

    Returns:
        Win probability as a float between 0.0 and 1.0
    """
    if not hero_hand or len(hero_hand) != 2:
        # If no hand, return 0
        return 0.0

    wins = 0
    ties = 0

    # Get cards already dealt (hero's hand + board)
    dealt_cards = set(hero_hand + board)

    for _ in range(num_simulations):
        # Create a deck without the dealt cards
        deck = Deck()
        deck.cards = [c for c in deck.cards if c not in dealt_cards]
        random.shuffle(deck.cards)

        # Complete the board if needed (need 5 total)
        sim_board = board.copy()
        cards_needed = 5 - len(sim_board)
        if cards_needed > 0:
            new_cards = deck.draw(num=cards_needed)
            sim_board.extend(new_cards)

        # Evaluate hero's hand
        hero_rank = evaluator.evaluate(hero_hand, sim_board)

        # Simulate opponent hands
        opponent_ranks = []
        for _ in range(num_opponents):
            if len(deck.cards) >= 2:
                opp_hand = deck.draw(num=2)
                opp_rank = evaluator.evaluate(opp_hand, sim_board)
                opponent_ranks.append(opp_rank)

        # Check if hero wins (lower rank is better)
        if not opponent_ranks:
            # No valid opponent hands, hero wins
            wins += 1
        elif hero_rank < min(opponent_ranks):
            wins += 1
        elif hero_rank == min(opponent_ranks):
            ties += 1

    # Win probability = (wins + ties/2) / total_simulations
    win_prob = (wins + ties * 0.5) / num_simulations
    return win_prob


def quick_win_probability(
    hero_hand: List[Card],
    board: List[Card],
    num_opponents: int = 1
) -> float:
    """Fast win probability estimation with fewer simulations.

    Uses 100 simulations for speed during training.

    Args:
        hero_hand: Hero's two hole cards
        board: Community cards (0-5 cards)
        num_opponents: Number of opponents (default 1)

    Returns:
        Win probability as a float between 0.0 and 1.0
    """
    return calculate_win_probability(
        hero_hand=hero_hand,
        board=board,
        num_opponents=num_opponents,
        num_simulations=100  # Reduced for speed
    )
