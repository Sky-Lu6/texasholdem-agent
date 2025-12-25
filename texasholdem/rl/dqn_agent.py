# texasholdem/rl/dqn_agent.py
"""
DQN Agent wrapper for inference (playing games without training).
This agent can be used in the API server or for evaluation.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from texasholdem.rl.env import HoldemEnv, ACTIONS


class DQNAgent:
    """DQN Agent that can play poker using a trained model."""

    def __init__(self, model_path=None, obs_dim=116, act_dim=8):
        """Initialize the DQN agent.

        Args:
            model_path: Path to trained model weights. If None, uses default checkpoint.
            obs_dim: State dimension (default 116)
            act_dim: Action space size (default 8 - now includes separate CHECK and CALL)
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Build network (same architecture as training)
        self.net = self._build_network()

        # Load weights if available
        if model_path is None:
            # Try to load best model first, fall back to latest
            ckpt_dir = "checkpoints"
            best_path = os.path.join(ckpt_dir, "holdem_dqn_best.pt")
            latest_path = os.path.join(ckpt_dir, "holdem_dqn.pt")

            if os.path.exists(best_path):
                model_path = best_path
                print(f"Loading best model from {best_path}")
            elif os.path.exists(latest_path):
                model_path = latest_path
                print(f"Loading latest model from {latest_path}")
            else:
                print("Warning: No trained model found. Using random weights.")
                return

        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location="cpu")

                # Handle different state_dict formats
                # Training saves with "net." prefix, but inference doesn't use it
                if any(k.startswith("net.") for k in state_dict.keys()):
                    # Remove "net." prefix from keys
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith("net."):
                            new_state_dict[k[4:]] = v  # Remove "net." prefix
                        else:
                            new_state_dict[k] = v
                    state_dict = new_state_dict

                self.net.load_state_dict(state_dict)
                self.net.eval()  # Set to evaluation mode
                print(f"Successfully loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using random weights instead.")

    def _build_network(self):
        """Build the Q-network (same as training)."""
        return nn.Sequential(
            nn.Linear(self.obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, self.act_dim)
        )

    def select_action(self, state, legal_mask, epsilon=0.0):
        """Select an action given state and legal action mask.

        Args:
            state: numpy array of shape (obs_dim,)
            legal_mask: numpy array of shape (act_dim,) with 1.0 for legal actions
            epsilon: epsilon for epsilon-greedy (default 0.0 = greedy)

        Returns:
            action_idx: int, index of selected action
        """
        # Epsilon-greedy
        if np.random.random() < epsilon:
            # Random legal action
            legal_indices = [i for i in range(self.act_dim) if legal_mask[i] == 1.0]
            return np.random.choice(legal_indices)

        # Greedy action
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.net(state_tensor)[0]  # shape (act_dim,)

            # Mask illegal actions
            masked_q = q_values.clone()
            for i in range(len(legal_mask)):
                if legal_mask[i] == 0.0:
                    masked_q[i] = -float('inf')

            return int(torch.argmax(masked_q).item())

    def get_action_name(self, action_idx):
        """Get human-readable action name.

        Args:
            action_idx: int, action index

        Returns:
            str: action name
        """
        action = ACTIONS[action_idx]
        if isinstance(action, str):
            return action
        else:
            return action.name


def play_game_with_agent(agent, env=None, verbose=False):
    """Play a single game using the DQN agent.

    Args:
        agent: DQNAgent instance
        env: HoldemEnv instance (creates new one if None)
        verbose: bool, whether to print game progress

    Returns:
        total_reward: float, cumulative reward for the game
        won: bool, whether agent won the hand
    """
    if env is None:
        env = HoldemEnv()

    state = env.reset()
    total_reward = 0.0
    done = False
    step = 0

    while not done:
        legal_mask = env.get_legal_actions_mask()
        action_idx = agent.select_action(state, legal_mask, epsilon=0.0)  # Greedy

        if verbose:
            action_name = agent.get_action_name(action_idx)
            print(f"Step {step + 1}: {action_name}")

        state, reward, done, _ = env.step(action_idx)
        total_reward += reward
        step += 1

    won = total_reward > 0

    if verbose:
        result = "WON" if won else "LOST"
        print(f"Game finished: {result} | Reward: {total_reward:.4f}")

    return total_reward, won


if __name__ == "__main__":
    # Demo: Load agent and play 10 games
    print("=" * 60)
    print("DQN Agent Demo")
    print("=" * 60)

    agent = DQNAgent()
    env = HoldemEnv()

    wins = 0
    total_reward = 0.0
    num_games = 10

    print(f"\nPlaying {num_games} games...\n")

    for i in range(num_games):
        reward, won = play_game_with_agent(agent, env, verbose=False)
        total_reward += reward
        if won:
            wins += 1
        print(f"Game {i+1:2d}: {'WIN' if won else 'LOSS'} | Reward: {reward:+7.4f}")

    print("\n" + "=" * 60)
    print(f"Results: {wins}/{num_games} wins ({100*wins/num_games:.1f}%)")
    print(f"Average reward: {total_reward/num_games:.4f}")
    print("=" * 60)
