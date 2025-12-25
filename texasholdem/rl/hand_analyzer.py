"""
Hand History Analyzer

Parses PGN files and extracts statistics for analysis dashboard.
"""
import os
import re
from typing import Dict, List, Any
from collections import defaultdict


class HandAnalyzer:
    """Analyzes poker hand histories from PGN files."""

    def __init__(self, pgn_directory: str = "./pgns"):
        self.pgn_directory = pgn_directory
        self.hands = []

    def parse_pgn_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Parse a single PGN file and extract hand data."""
        hands = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Each PGN file contains one hand (sections are separated by blank lines
            # but they all belong to the same hand)
            if content.strip():
                hand_data = self._parse_hand(content)
                if hand_data:
                    hands.append(hand_data)

        except Exception as e:
            print(f"Error parsing {filepath}: {e}")

        return hands

    def _parse_hand(self, hand_text: str) -> Dict[str, Any]:
        """Parse a single hand from PGN text."""
        hand_data = {
            'players': {},
            'winner': None,
            'pot_size': 0,
            'actions': [],
            'board': [],
            'hand_phase': None
        }

        lines = hand_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse player chips: "Player Chips: 1000,1000"
            if line.startswith('Player Chips:'):
                chips_str = line.split(':', 1)[1].strip()
                chips_list = [int(c.strip()) for c in chips_str.split(',')]
                for idx, chips in enumerate(chips_list):
                    hand_data['players'][idx] = {'chips': chips, 'actions': 0, 'invested': 0}
                continue

            # Parse actions: "1. (0,CALL);(1,RAISE,40)"
            if re.match(r'\d+\.\s+\(', line):
                # Extract all (player,action,amount?) tuples
                actions = re.findall(r'\((\d+),(FOLD|CALL|CHECK|RAISE|ALL_IN)(?:,(\d+))?\)', line)
                for action_tuple in actions:
                    player_id = int(action_tuple[0])
                    action_type = action_tuple[1]
                    amount = int(action_tuple[2]) if action_tuple[2] else 0

                    hand_data['actions'].append({
                        'player': player_id,
                        'action': action_type,
                        'amount': amount
                    })

                    if player_id in hand_data['players']:
                        hand_data['players'][player_id]['actions'] += 1
                        if amount > 0:
                            hand_data['players'][player_id]['invested'] = amount
                continue

            # Parse phase headers: "PREFLOP", "FLOP", "TURN", "RIVER"
            if line in ['PREFLOP', 'FLOP', 'TURN', 'RIVER']:
                hand_data['hand_phase'] = line
                continue

            # Parse new cards: "New Cards: [3s,3h,9c]"
            if line.startswith('New Cards:'):
                cards_str = line.split(':', 1)[1].strip()
                # Extract cards from brackets [card1,card2,...]
                cards_match = re.findall(r'\[([^\]]+)\]', cards_str)
                if cards_match:
                    cards = [c.strip() for c in cards_match[0].split(',') if c.strip()]
                    hand_data['board'].extend(cards)
                continue

            # Parse winner: "Winners: (Pot 0,2000,3241,[1])"
            if line.startswith('Winners:'):
                # Format: (Pot ID, amount, rank, [winners])
                winner_match = re.search(r'\(Pot \d+,(\d+),\d+,\[([^\]]+)\]\)', line)
                if winner_match:
                    pot_amount = int(winner_match.group(1))
                    winners_str = winner_match.group(2)
                    winners = [int(w.strip()) for w in winners_str.split(',')]

                    hand_data['pot_size'] += pot_amount
                    if not hand_data['winner']:
                        hand_data['winner'] = winners[0] if len(winners) == 1 else winners
                continue

        return hand_data if hand_data['players'] else None

    def analyze_directory(self) -> Dict[str, Any]:
        """Analyze all PGN files in the directory."""
        all_hands = []

        if not os.path.exists(self.pgn_directory):
            return self._empty_stats()

        for filename in os.listdir(self.pgn_directory):
            if filename.endswith('.pgn'):
                filepath = os.path.join(self.pgn_directory, filename)
                hands = self.parse_pgn_file(filepath)
                all_hands.extend(hands)

        self.hands = all_hands
        return self.calculate_statistics()

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate statistics from parsed hands."""
        if not self.hands:
            return self._empty_stats()

        stats = {
            'total_hands': len(self.hands),
            'player_stats': defaultdict(lambda: {
                'hands_played': 0,
                'hands_won': 0,
                'total_invested': 0,
                'total_won': 0,
                'win_rate': 0.0,
                'actions': defaultdict(int),
                'avg_pot_size': 0.0
            }),
            'action_frequency': defaultdict(int),
            'phase_distribution': defaultdict(int),
            'pot_size_distribution': [],
            'winner_distribution': defaultdict(int)
        }

        total_pot = 0

        for hand in self.hands:
            # Track phase
            if hand['hand_phase']:
                stats['phase_distribution'][hand['hand_phase']] += 1

            # Track pot sizes
            if hand['pot_size'] > 0:
                stats['pot_size_distribution'].append(hand['pot_size'])
                total_pot += hand['pot_size']

            # Track player stats
            for player_id, player_data in hand['players'].items():
                player_stats = stats['player_stats'][player_id]
                player_stats['hands_played'] += 1
                player_stats['total_invested'] += player_data.get('invested', 0)

            # Track actions
            for action in hand['actions']:
                stats['action_frequency'][action['action']] += 1

                player_id = action['player']
                player_stats = stats['player_stats'][player_id]
                player_stats['actions'][action['action']] += 1

            # Track winners
            winner = hand['winner']
            if winner is not None:
                if isinstance(winner, list):
                    for w in winner:
                        stats['winner_distribution'][w] += 1
                        stats['player_stats'][w]['hands_won'] += 1
                        stats['player_stats'][w]['total_won'] += hand['pot_size'] / len(winner)
                else:
                    stats['winner_distribution'][winner] += 1
                    stats['player_stats'][winner]['hands_won'] += 1
                    stats['player_stats'][winner]['total_won'] += hand['pot_size']

        # Calculate derived stats
        for player_id, player_stats in stats['player_stats'].items():
            if player_stats['hands_played'] > 0:
                player_stats['win_rate'] = (player_stats['hands_won'] / player_stats['hands_played']) * 100
                player_stats['avg_pot_size'] = player_stats['total_won'] / player_stats['hands_played'] if player_stats['hands_played'] > 0 else 0

        # Convert defaultdicts to regular dicts for JSON serialization
        stats['player_stats'] = dict(stats['player_stats'])
        for pid in stats['player_stats']:
            stats['player_stats'][pid]['actions'] = dict(stats['player_stats'][pid]['actions'])

        stats['action_frequency'] = dict(stats['action_frequency'])
        stats['phase_distribution'] = dict(stats['phase_distribution'])
        stats['winner_distribution'] = dict(stats['winner_distribution'])

        # Add summary stats
        stats['avg_pot_size'] = total_pot / len(self.hands) if self.hands else 0

        return stats

    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics structure."""
        return {
            'total_hands': 0,
            'player_stats': {},
            'action_frequency': {},
            'phase_distribution': {},
            'pot_size_distribution': [],
            'winner_distribution': {},
            'avg_pot_size': 0
        }


# Example usage
if __name__ == "__main__":
    analyzer = HandAnalyzer("./pgns")
    stats = analyzer.analyze_directory()

    print(f"Total hands: {stats['total_hands']}")
    print(f"Average pot size: {stats['avg_pot_size']:.2f}")
    print(f"\nPlayer statistics:")
    for player_id, pstats in stats['player_stats'].items():
        print(f"  Player {player_id}: {pstats['hands_won']}/{pstats['hands_played']} wins ({pstats['win_rate']:.1f}%)")
