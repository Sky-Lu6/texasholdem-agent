# texasholdem/rl/train_multi_agent.py
"""
Multi-agent parallel training for faster learning.
Runs multiple environments in parallel with vectorized experience collection.
"""
import pickle
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from texasholdem.rl.env import HoldemEnv
import os
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

def select_legal_action(q_values, legal_mask):
    """Select action from Q-values using legal action mask."""
    masked_q = q_values.clone()
    for i in range(len(legal_mask)):
        if legal_mask[i] == 0.0:
            masked_q[i] = -float('inf')
    return int(torch.argmax(masked_q).item())

# === checkpoints & paths ===
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(CKPT_DIR, "holdem_dqn_multi.pt")
BEST_MODEL_PATH = os.path.join(CKPT_DIR, "holdem_dqn_best.pt")
REPLAY_PATH = os.path.join(CKPT_DIR, "replay_multi.pkl")
STATE_PATH = os.path.join(CKPT_DIR, "train_state_multi.pkl")

class Net(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, act_dim)
        )
    def forward(self, x):
        return self.net(x)

class ParallelEnvRunner:
    """Manages multiple environments running in parallel."""

    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.envs = [HoldemEnv() for _ in range(num_envs)]
        self.states = [env.reset() for env in self.envs]
        self.dones = [False] * num_envs
        self.episode_rewards = [0.0] * num_envs
        self.episode_lengths = [0] * num_envs

    def reset_env(self, idx):
        """Reset a specific environment."""
        self.states[idx] = self.envs[idx].reset()
        self.dones[idx] = False
        reward = self.episode_rewards[idx]
        length = self.episode_lengths[idx]
        self.episode_rewards[idx] = 0.0
        self.episode_lengths[idx] = 0
        return reward, length

    def step(self, actions):
        """
        Take a step in all environments.

        Args:
            actions: list of action indices, one per environment

        Returns:
            states: list of next states
            rewards: list of rewards
            dones: list of done flags
            infos: list of info dicts
            finished_episodes: list of (reward, length) for completed episodes
        """
        finished_episodes = []

        for i in range(self.num_envs):
            if not self.dones[i]:
                state, reward, done, info = self.envs[i].step(actions[i])
                self.states[i] = state
                self.dones[i] = done
                self.episode_rewards[i] += reward
                self.episode_lengths[i] += 1

                if done:
                    ep_reward, ep_length = self.reset_env(i)
                    finished_episodes.append((ep_reward, ep_length))

        return self.states.copy(), self.dones.copy(), finished_episodes

    def get_legal_masks(self):
        """Get legal action masks for all environments."""
        return [env.get_legal_actions_mask() for env in self.envs]

def train_multi_agent(episodes=1000, num_parallel_envs=4, save_interval=100, reset_epsilon=False):
    """
    Train with multiple parallel environments.

    Args:
        episodes: Total number of episodes to train (across all envs)
        num_parallel_envs: Number of environments to run in parallel
        save_interval: Save checkpoint every N episodes
        reset_epsilon: If True, reset epsilon to initial value (useful when changing reward function)
    """
    print("=" * 80)
    print("MULTI-AGENT PARALLEL TRAINING")
    print("=" * 80)
    print(f"Parallel environments: {num_parallel_envs}")
    print(f"Target total episodes: {episodes}")
    print(f"Effective speedup: ~{num_parallel_envs}x")
    print("=" * 80)

    # Initialize environment to get dimensions
    env_sample = HoldemEnv()
    s = env_sample.reset()
    from texasholdem.rl.env import ACTIONS
    obs_dim, act_dim = s.shape[0], len(ACTIONS)
    print(f"State dimension: {obs_dim}")
    print(f"Action dimension: {act_dim}")

    # Create parallel environments
    env_runner = ParallelEnvRunner(num_envs=num_parallel_envs)

    # Initialize networks
    q = Net(obs_dim, act_dim)
    tgt = Net(obs_dim, act_dim)
    tgt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=1e-3)

    # Load existing model if available
    load_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
    if os.path.exists(load_path):
        try:
            state = torch.load(load_path, map_location="cpu")
            q.load_state_dict(state, strict=False)
            tgt.load_state_dict(q.state_dict())
            print(f"Loaded model from {load_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")

    # Initialize replay buffer
    buf = deque(maxlen=100000)  # Larger buffer for multi-agent
    if os.path.exists(REPLAY_PATH):
        try:
            with open(REPLAY_PATH, "rb") as f:
                buf = pickle.load(f)
            print(f"Loaded replay buffer with {len(buf)} entries")
        except:
            print("Could not load replay buffer, starting fresh")

    # Hyperparameters
    gamma = 0.99
    eps_init, eps_min, eps_decay = 0.9, 0.05, 0.9995
    eps = eps_init
    batch_size = 128  # Larger batch for multi-agent

    # Training state
    global_steps = 0
    best_avg_reward = float('-inf')
    episode_rewards = []
    win_count = 0
    total_hands = 0
    start_time = time.time()

    # Load training state if exists
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "rb") as f:
                st = pickle.load(f)

            # Load epsilon from checkpoint, unless reset_epsilon flag is set
            if reset_epsilon:
                print(f"RESET EPSILON FLAG SET - Using initial epsilon={eps_init:.3f}")
                eps = eps_init
            else:
                eps = st.get("eps", eps)
                eps = max(eps, 0.03)

            global_steps = st.get("global_steps", 0)
            best_avg_reward = st.get("best_avg_reward", float('-inf'))
            episode_rewards = st.get("episode_rewards", [])
            win_count = st.get("win_count", 0)
            total_hands = st.get("total_hands", 0)

            print(f"Resuming: eps={eps:.3f}, global_steps={global_steps}")
            print(f"Best avg reward: {best_avg_reward:.4f}")
            if total_hands > 0:
                print(f"Win rate: {win_count}/{total_hands} = {100*win_count/total_hands:.1f}%")
        except Exception as e:
            print(f"Could not load training state: {e}")

    print("\nStarting training...")
    print("=" * 80)

    # Training loop
    # Calculate how many episodes we've already completed
    completed_episodes = len(episode_rewards)

    # If we've already completed the target, inform user
    if completed_episodes >= episodes:
        print(f"Already completed {completed_episodes} episodes (target: {episodes})")
        print("To train more, increase --episodes or delete checkpoints/train_state_multi.pkl")
        return

    print(f"Resuming from episode {completed_episodes + 1}")

    while completed_episodes < episodes:
        # Get current states and legal masks
        states = env_runner.states
        legal_masks = env_runner.get_legal_masks()

        # Select actions for all environments
        actions = []
        for i in range(num_parallel_envs):
            if random.random() < eps:
                # Epsilon-greedy: random legal action
                legal_indices = [j for j in range(act_dim) if legal_masks[i][j] == 1.0]
                a = random.choice(legal_indices) if legal_indices else 0
            else:
                # Greedy: best legal action
                with torch.no_grad():
                    s_tensor = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0)
                    q_values = q(s_tensor)[0]
                    a = select_legal_action(q_values, legal_masks[i])
            actions.append(a)

        # Step all environments
        old_states = states.copy()
        next_states, dones, finished_episodes = env_runner.step(actions)

        # Collect experiences
        for i in range(num_parallel_envs):
            if not dones[i]:  # Only add if episode is still running
                reward = env_runner.episode_rewards[i] - (
                    episode_rewards[-1] if episode_rewards else 0.0
                )
                buf.append((old_states[i], actions[i], reward, next_states[i], dones[i]))
                global_steps += 1

        # Process finished episodes
        for ep_reward, ep_length in finished_episodes:
            episode_rewards.append(ep_reward)
            total_hands += 1
            if ep_reward > 0:
                win_count += 1

            completed_episodes += 1

            # Print progress every 10 episodes
            if completed_episodes % 10 == 0:
                recent_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                win_rate = 100 * win_count / total_hands if total_hands > 0 else 0.0
                elapsed = time.time() - start_time
                eps_per_sec = completed_episodes / max(1, elapsed)

                print(f"Ep {completed_episodes:5d}/{episodes} | "
                      f"Reward: {ep_reward:8.3f} | Avg100: {recent_avg:7.3f} | "
                      f"WinRate: {win_rate:5.1f}% | Eps: {eps:5.3f} | "
                      f"Speed: {eps_per_sec:.1f} ep/s")

        # Train on batch
        if len(buf) >= batch_size:
            batch_samples = random.sample(buf, batch_size)
            batch_s, batch_a, batch_r, batch_s2, batch_d = zip(*batch_samples)

            S = torch.tensor(np.array(batch_s)).float()
            A = torch.tensor(batch_a).long().unsqueeze(1)
            R = torch.tensor(batch_r).float().unsqueeze(1)
            S2 = torch.tensor(np.array(batch_s2)).float()
            D = torch.tensor(batch_d).float().unsqueeze(1)

            q_sa = q(S).gather(1, A)
            with torch.no_grad():
                max_next = tgt(S2).max(1, keepdim=True)[0]
                y = R + gamma * (1 - D) * max_next

            loss = nn.functional.smooth_l1_loss(q_sa, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 5.0)
            opt.step()

        # Soft target update
        if global_steps % 10 == 0:
            for t, s_ in zip(tgt.parameters(), q.parameters()):
                t.data.copy_(0.995 * t.data + 0.005 * s_.data)

        # Decay epsilon
        if completed_episodes > len(episode_rewards) - num_parallel_envs:
            eps = max(eps_min, eps * eps_decay)

        # Save best model
        if len(episode_rewards) >= 20:
            recent_20_avg = np.mean(episode_rewards[-20:])
            if recent_20_avg > best_avg_reward:
                best_avg_reward = recent_20_avg
                torch.save(q.state_dict(), BEST_MODEL_PATH)
                print(f"  >>> New best model! Avg reward (last 20): {best_avg_reward:.4f}")

        # Periodic checkpoint
        if completed_episodes % save_interval == 0 and completed_episodes > 0:
            torch.save(q.state_dict(), MODEL_PATH)
            with open(REPLAY_PATH, "wb") as f:
                pickle.dump(buf, f)
            with open(STATE_PATH, "wb") as f:
                pickle.dump({
                    "eps": eps,
                    "global_steps": global_steps,
                    "best_avg_reward": best_avg_reward,
                    "episode_rewards": episode_rewards,
                    "win_count": win_count,
                    "total_hands": total_hands
                }, f)
            print(f"  >>> Checkpoint saved at episode {completed_episodes}")

    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Total episodes: {completed_episodes}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Average speed: {completed_episodes/elapsed:.1f} episodes/second")
    print(f"Speedup vs single-env: ~{num_parallel_envs:.1f}x")
    print(f"Final win rate: {win_count}/{total_hands} = {100*win_count/total_hands:.1f}%")
    print(f"Best avg reward (last 20): {best_avg_reward:.4f}")
    print(f"Final epsilon: {eps:.4f}")
    print(f"Replay buffer size: {len(buf)}")
    print("=" * 80)

    # Save final checkpoint
    torch.save(q.state_dict(), MODEL_PATH)
    with open(REPLAY_PATH, "wb") as f:
        pickle.dump(buf, f)
    with open(STATE_PATH, "wb") as f:
        pickle.dump({
            "eps": eps,
            "global_steps": global_steps,
            "best_avg_reward": best_avg_reward,
            "episode_rewards": episode_rewards,
            "win_count": win_count,
            "total_hands": total_hands
        }, f)
    print(f"\nSaved final checkpoint to {MODEL_PATH}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-agent DQN training for Texas Hold'em")
    parser.add_argument("--episodes", type=int, default=5000, help="Total episodes to train")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--save-interval", type=int, default=100, help="Save checkpoint every N episodes")
    args = parser.parse_args()

    train_multi_agent(
        episodes=args.episodes,
        num_parallel_envs=args.parallel,
        save_interval=args.save_interval
    )
