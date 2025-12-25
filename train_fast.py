#!/usr/bin/env python
"""
Quick script to run fast multi-agent training.
Usage:
    python train_fast.py              # Default: 5000 episodes, 4 parallel envs
    python train_fast.py --episodes 10000 --parallel 8
"""
from texasholdem.rl.train_multi_agent import train_multi_agent
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast multi-agent DQN training")
    parser.add_argument("--episodes", type=int, default=5000,
                        help="Total episodes to train (default: 5000)")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="Save checkpoint every N episodes (default: 100)")
    parser.add_argument("--continue", dest="continue_training", action="store_true",
                        help="Continue training beyond checkpoint episodes")
    parser.add_argument("--reset-epsilon", action="store_true",
                        help="Reset epsilon to initial value (useful when changing reward function)")
    args = parser.parse_args()

    # If --continue flag is set, add more episodes to existing count
    if args.continue_training:
        import os
        import pickle
        STATE_PATH = os.path.join("checkpoints", "train_state_multi.pkl")
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "rb") as f:
                st = pickle.load(f)
            existing_episodes = len(st.get("episode_rewards", []))
            args.episodes = existing_episodes + args.episodes
            print(f"Continue mode: Training from {existing_episodes} to {args.episodes} episodes")

    print(f"""
================================================================
           FAST MULTI-AGENT POKER TRAINING
================================================================

Configuration:
  * Total episodes: {args.episodes}
  * Parallel envs: {args.parallel}
  * Speedup: ~{args.parallel}x faster than single-env training
  * Save interval: every {args.save_interval} episodes

Expected training time:
  * Single-env baseline: ~{args.episodes * 5 / 60:.0f} minutes
  * Multi-agent ({args.parallel} envs): ~{args.episodes * 5 / 60 / args.parallel:.0f} minutes

Starting training...
""")

    train_multi_agent(
        episodes=args.episodes,
        num_parallel_envs=args.parallel,
        save_interval=args.save_interval,
        reset_epsilon=args.reset_epsilon
    )
