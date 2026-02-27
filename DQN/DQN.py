import random
from collections import deque
from utils import set_seed, linear_epsilon
from config import Config
from plots import plot_training
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym

cfg = Config()
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = map(np.array, zip(*batch))
        return s, a, r, s2, done

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


def train_dqn():
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    # render_mode="human" abre ventana (pyglet) y se actualiza en cada step
    env = gym.make(cfg.env_id, render_mode="human")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Seeds Gymnasium (nuevo API)
    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)
    env.observation_space.seed(cfg.seed)

    q = QNetwork(obs_dim, n_actions).to(device)
    q_target = QNetwork(obs_dim, n_actions).to(device)
    q_target.load_state_dict(q.state_dict())
    q_target.eval()

    optimizer = optim.Adam(q.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.buffer_size)
    loss_fn = nn.SmoothL1Loss()  # Función de pérdida Huber loss

    # Historial para gráficas
    episode_rewards = []
    episode_epsilons = []
    losses = []
    global_step = 0

    for ep in range(1, cfg.max_episodes + 1):
        s, _ = env.reset()
        ep_reward = 0.0
        ep_loss_vals = []

        for _ in range(cfg.max_steps_per_episode):
            global_step += 1

            eps = linear_epsilon(
                global_step, cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps
            )

            # Política epsilon-greedy
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    qvals = q(st)
                    a = int(torch.argmax(qvals, dim=1).item())

            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            replay.push(s, a, r, s2, done)
            s = s2
            ep_reward += r

            # Entrenamiento
            if len(replay) >= cfg.min_buffer_size and (global_step % cfg.train_freq == 0):
                sb, ab, rb, s2b, doneb = replay.sample(cfg.batch_size)

                sb_t = torch.tensor(sb, dtype=torch.float32, device=device)
                ab_t = torch.tensor(ab, dtype=torch.int64, device=device).unsqueeze(1)
                rb_t = torch.tensor(rb, dtype=torch.float32, device=device).unsqueeze(1)
                s2b_t = torch.tensor(s2b, dtype=torch.float32, device=device)
                doneb_t = torch.tensor(doneb.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(1)

                # Q(s,a)
                q_sa = q(sb_t).gather(1, ab_t)

                # Target: r + gamma * max_a' Q_target(s',a') * (1-done)
                with torch.no_grad():
                    max_q_next = q_target(s2b_t).max(dim=1, keepdim=True)[0]
                    target = rb_t + cfg.gamma * max_q_next * (1.0 - doneb_t)

                loss = loss_fn(q_sa, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), max_norm=10.0)  # estabilidad
                optimizer.step()

                losses.append(loss.item())
                ep_loss_vals.append(loss.item())

            # Actualizar red objetivo
            if global_step % cfg.target_update_freq == 0:
                q_target.load_state_dict(q.state_dict())

            if done:
                break

        episode_rewards.append(ep_reward)
        episode_epsilons.append(eps)

        # Logging
        avg_last = np.mean(episode_rewards[-cfg.solved_window:]) if len(episode_rewards) >= 2 else ep_reward
        ep_loss_mean = float(np.mean(ep_loss_vals)) if ep_loss_vals else 0.0
        print(
            f"Ep {ep:04d} | reward={ep_reward:6.1f} | avg({cfg.solved_window})={avg_last:6.1f} "
            f"| eps={eps:.3f} | loss~={ep_loss_mean:.4f} | steps={global_step}"
        )

        # Criterio de “resuelto”
        if len(episode_rewards) >= cfg.solved_window and avg_last >= cfg.solved_avg_reward:
            print(f"\n✅ Entorno resuelto: promedio {cfg.solved_window} eps = {avg_last:.1f} (>= {cfg.solved_avg_reward}).")
            break

    env.close()
    return episode_rewards, episode_epsilons, losses



