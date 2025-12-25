# texasholdem/rl/train_dqn.py
import pickle
import random
import numpy as np
from collections import deque
import torch, torch.nn as nn, torch.optim as optim
from texasholdem.rl.env import HoldemEnv
import os

def select_legal_action(q_values, legal_mask):
    """Select action from Q-values using legal action mask.

    Args:
        q_values: torch.Tensor of shape (7,) with Q-values for each action
        legal_mask: numpy array of shape (7,) with 1.0 for legal, 0.0 for illegal

    Returns:
        int: index of selected action
    """
    # Mask out illegal actions by setting their Q-values to -inf
    masked_q = q_values.clone()
    for i in range(len(legal_mask)):
        if legal_mask[i] == 0.0:
            masked_q[i] = -float('inf')

    return int(torch.argmax(masked_q).item())

# === checkpoints & paths ===
CKPT_DIR    = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
MODEL_PATH  = os.path.join(CKPT_DIR, "holdem_dqn.pt")
REPLAY_PATH = os.path.join(CKPT_DIR, "replay.pkl")
STATE_PATH  = os.path.join(CKPT_DIR, "train_state.pkl")

class Net(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Larger network for richer state representation (112 dims with cards)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, act_dim)
        )
    def forward(self, x):
        return self.net(x)

def train(episodes=100):
    env = HoldemEnv()
    s = env.reset()
    obs_dim, act_dim = s.shape[0], 7  # Changed from 3 to 7 actions

    print(f"State dimension: {obs_dim} (expected 116 with enhanced features)")
    print(f"Action dimension: {act_dim} (FOLD, CALL, RAISE_MIN, RAISE_HALF_POT, RAISE_POT, RAISE_2X_POT, ALL_IN)")

    q = Net(obs_dim, act_dim)
    tgt = Net(obs_dim, act_dim)
    tgt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=1e-3)

    # ---- load model if exists ----
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location="cpu")
            q.load_state_dict(state, strict=False)  # allow shape mismatches

            # If the output head size is wrong, reinit just the last layer
            out_dim = q.net[-1].out_features
            if out_dim != act_dim:
                print(f"Warning: checkpoint head={out_dim}, expected={act_dim}. Reinitializing head.")
                in_dim = q.net[-1].in_features
                q.net[-1] = nn.Linear(in_dim, act_dim)
            # mirror into target
            tgt.load_state_dict(q.state_dict())
            print(f"Loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint (state size mismatch?): {e}")
            print("Starting with fresh model. Consider deleting old checkpoints if state dimension changed.")

    buf = deque(maxlen=50000)
    if os.path.exists(REPLAY_PATH):
        with open(REPLAY_PATH, "rb") as f:
            buf = pickle.load(f)
        print(f"Loaded replay buffer with {len(buf)} entries")

    gamma = 0.99
    eps_init, eps_min, eps_decay = 0.9, 0.05, 0.9995
    eps = eps_init
    batch = 64

    global_steps = 0
    best_avg_reward = float('-inf')

    # Training metrics tracking
    episode_rewards = []
    episode_losses = []
    win_count = 0
    total_hands = 0

    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "rb") as f:
            st = pickle.load(f)
        eps = st.get("eps", eps)
        global_steps = st.get("global_steps", 0)
        best_avg_reward = st.get("best_avg_reward", float('-inf'))
        episode_rewards = st.get("episode_rewards", [])
        episode_losses = st.get("episode_losses", [])
        win_count = st.get("win_count", 0)
        total_hands = st.get("total_hands", 0)
        eps = max(eps, 0.03)   # warm-start floor
        print(f"Resuming eps={eps:.3f}, global_steps={global_steps}")
        print(f"Best avg reward so far: {best_avg_reward:.4f}")
        if total_hands > 0:
            print(f"Win rate so far: {win_count}/{total_hands} = {100*win_count/total_hands:.1f}%")

    for ep in range(episodes):
        s = env.reset()
        done = False
        total = 0.0
        episode_loss = 0.0
        loss_count = 0

        while not done:
            # Get legal actions mask
            legal_mask = env.get_legal_actions_mask()

            if random.random() < eps:
                # Epsilon-greedy: random action from legal actions
                legal_indices = [i for i in range(act_dim) if legal_mask[i] == 1.0]
                a = random.choice(legal_indices)
            else:
                # Greedy: select best legal action
                with torch.no_grad():
                    q_t = q(torch.tensor(s, dtype=torch.float32).unsqueeze(0))[0]  # shape (act_dim,)
                    a = select_legal_action(q_t, legal_mask)

            s2, r, done, _ = env.step(a)
            buf.append((s, a, r, s2, done))
            s = s2
            total += r
            global_steps += 1

            if len(buf) >= batch:
                batch_s, batch_a, batch_r, batch_s2, batch_d = zip(*random.sample(buf, batch))
                S  = torch.tensor(np.array(batch_s)).float()
                A  = torch.tensor(batch_a).long().unsqueeze(1)
                R  = torch.tensor(batch_r).float().unsqueeze(1)
                S2 = torch.tensor(np.array(batch_s2)).float()
                D  = torch.tensor(batch_d).float().unsqueeze(1)

                q_sa = q(S).gather(1, A)
                with torch.no_grad():
                    max_next = tgt(S2).max(1, keepdim=True)[0]
                    y = R + gamma * (1 - D) * max_next

                loss = nn.functional.smooth_l1_loss(q_sa, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                # Track loss
                episode_loss += loss.item()
                loss_count += 1

        # Track episode metrics
        episode_rewards.append(total)
        avg_loss = episode_loss / max(1, loss_count)
        episode_losses.append(avg_loss)

        # Track wins (positive reward indicates won hand)
        total_hands += 1
        if total > 0:
            win_count += 1

        # Print episode summary with metrics
        recent_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        win_rate = 100 * win_count / total_hands if total_hands > 0 else 0.0

        print(f"Episode {ep+1:3d} | Reward: {total:8.3f} | Avg100: {recent_avg:7.3f} | "
              f"Loss: {avg_loss:7.4f} | WinRate: {win_rate:5.1f}% | Eps: {eps:5.3f}")

        # Soft target network update
        for t, s_ in zip(tgt.parameters(), q.parameters()):
            t.data.copy_(0.995 * t.data + 0.005 * s_.data)

        # Decay epsilon
        eps = max(eps_min, eps * eps_decay)

        # Save best model based on recent average reward
        if len(episode_rewards) >= 20:
            recent_20_avg = np.mean(episode_rewards[-20:])
            if recent_20_avg > best_avg_reward:
                best_avg_reward = recent_20_avg
                best_model_path = os.path.join(CKPT_DIR, "holdem_dqn_best.pt")
                torch.save(q.state_dict(), best_model_path)
                print(f"  >>> New best model! Avg reward (last 20): {best_avg_reward:.4f}")

    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Final win rate: {win_count}/{total_hands} = {100*win_count/total_hands:.1f}%")
    print(f"Best avg reward (last 20): {best_avg_reward:.4f}")
    print(f"Final epsilon: {eps:.4f}")
    print("=" * 60)

    # Save checkpoint and replay buffer
    torch.save(q.state_dict(), MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    with open(REPLAY_PATH, "wb") as f:
        pickle.dump(buf, f)
    print(f"Saved replay buffer with {len(buf)} entries")

    with open(STATE_PATH, "wb") as f:
        pickle.dump({
            "eps": eps,
            "global_steps": global_steps,
            "best_avg_reward": best_avg_reward,
            "episode_rewards": episode_rewards,
            "episode_losses": episode_losses,
            "win_count": win_count,
            "total_hands": total_hands
        }, f)
    print(f"Saved training state: eps={eps:.3f}, steps={global_steps}")


if __name__ == "__main__":
    train(episodes=100)