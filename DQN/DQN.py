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
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py



cfg = Config()
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, done = map(np.array, zip(*batch))
        return s, a, r, s2, done

    def __len__(self):
        return len(self.buf)

class QNetworkMLP(nn.Module):
    """Para entornos con estado vectorial (CartPole)."""
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class DQNCNN(nn.Module):
    """Para Atari: entrada (N,C,84,84) con C=frame_stack."""
    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            n_flat = self.features(dummy).view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flat, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


# =========================
# Environments
# =========================
def is_atari_env(env_id: str) -> bool:
    return env_id.startswith("ALE/")


def make_env(env_id: str, seed: int, render: bool):
    render_mode = "human" if render else None

    if not is_atari_env(env_id):
        env = gym.make(env_id, render_mode=render_mode)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    # Atari
    env = gym.make(env_id, render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,  # mantenemos 0..255; normalizamos nosotros
    )
    env = FrameStackObservation(env, stack_size=cfg.atari_frame_stack)

    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def obs_to_tensor(obs, device, atari: bool):
    """
    CartPole: obs (obs_dim,) -> (1,obs_dim)
    Atari: obs (C,84,84) uint8 -> (1,C,84,84) float [0,1]
    """
    if not atari:
        return torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    x = np.array(obs, dtype=np.float32) / 255.0
    return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)


# =========================
# Entrenamiento: CartPole
# =========================
def train_cartpole():
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CartPole] device: {device}")

    env = make_env(cfg.env_id, cfg.seed, cfg.render)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q = QNetworkMLP(obs_dim, n_actions).to(device)
    q_target = QNetworkMLP(obs_dim, n_actions).to(device)
    q_target.load_state_dict(q.state_dict())
    q_target.eval()

    optimizer = optim.Adam(q.parameters(), lr=cfg.lr)
    loss_fn = nn.SmoothL1Loss()
    replay = ReplayBuffer(cfg.buffer_size)

    episode_rewards, episode_epsilons, losses = [], [], []
    global_step = 0

    for ep in range(1, cfg.max_episodes + 1):
        s, _ = env.reset()
        ep_reward = 0.0
        ep_loss_vals = []

        for _ in range(cfg.max_steps_per_episode):
            global_step += 1
            eps = linear_epsilon(global_step, cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps)

            if random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    st = obs_to_tensor(s, device, atari=False)
                    qvals = q(st)
                    a = int(torch.argmax(qvals, dim=1).item())

            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            replay.push(s, a, float(r), s2, done)
            s = s2
            ep_reward += float(r)

            # Entrenar
            if len(replay) >= cfg.min_buffer_size and (global_step % cfg.train_freq == 0):
                sb, ab, rb, s2b, doneb = replay.sample(cfg.batch_size)

                sb_t = torch.tensor(sb, dtype=torch.float32, device=device)
                ab_t = torch.tensor(ab, dtype=torch.int64, device=device).unsqueeze(1)
                rb_t = torch.tensor(rb, dtype=torch.float32, device=device).unsqueeze(1)
                s2b_t = torch.tensor(s2b, dtype=torch.float32, device=device)
                doneb_t = torch.tensor(doneb.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(1)

                q_sa = q(sb_t).gather(1, ab_t)
                with torch.no_grad():
                    max_q_next = q_target(s2b_t).max(dim=1, keepdim=True)[0]
                    target = rb_t + cfg.gamma * max_q_next * (1.0 - doneb_t)

                loss = loss_fn(q_sa, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                optimizer.step()

                losses.append(loss.item())
                ep_loss_vals.append(loss.item())

            if global_step % cfg.target_update_freq == 0:
                q_target.load_state_dict(q.state_dict())

            if done:
                break

        episode_rewards.append(ep_reward)
        episode_epsilons.append(eps)
        avg_50 = np.mean(episode_rewards[-50:])
        print(f"Ep {ep:04d} | R={ep_reward:6.1f} | avg50={avg_50:6.1f} | eps={eps:.3f} | loss~={np.mean(ep_loss_vals) if ep_loss_vals else 0:.4f}")

    env.close()
    return episode_rewards, episode_epsilons, losses


# =========================
# Entrenamiento: Atari
# =========================
def train_atari():
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Atari] device: {device}")

    env = make_env(cfg.env_id, cfg.seed, cfg.render)
    n_actions = env.action_space.n

    q = DQNCNN(n_actions=n_actions, in_channels=cfg.atari_frame_stack).to(device)
    q_target = DQNCNN(n_actions=n_actions, in_channels=cfg.atari_frame_stack).to(device)
    q_target.load_state_dict(q.state_dict())
    q_target.eval()

    optimizer = optim.Adam(q.parameters(), lr=cfg.atari_lr)
    loss_fn = nn.SmoothL1Loss()
    replay = ReplayBuffer(cfg.buffer_size)

    # Historial (por episodio) + loss (por update)
    episode_rewards, episode_epsilons, losses = [], [], []

    step = 0
    episode = 0

    obs, _ = env.reset()
    ep_reward = 0.0

    while step < cfg.atari_max_steps_total:
        step += 1
        eps = linear_epsilon(step, cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps)

        # acción
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                st = obs_to_tensor(obs, device, atari=True)
                qvals = q(st)
                a = int(torch.argmax(qvals, dim=1).item())

        obs2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        ep_reward += float(r)

        replay.push(np.array(obs), a, float(r), np.array(obs2), done)
        obs = obs2

        # Entrenar
        if len(replay) >= cfg.atari_min_buffer_size and (step % cfg.atari_train_freq == 0):
            sb, ab, rb, s2b, doneb = replay.sample(cfg.atari_batch_size)

            sb_t = torch.tensor(sb, dtype=torch.float32, device=device) / 255.0
            s2b_t = torch.tensor(s2b, dtype=torch.float32, device=device) / 255.0
            ab_t = torch.tensor(ab, dtype=torch.int64, device=device).unsqueeze(1)
            rb_t = torch.tensor(rb, dtype=torch.float32, device=device).unsqueeze(1)
            doneb_t = torch.tensor(doneb.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(1)

            q_sa = q(sb_t).gather(1, ab_t)
            with torch.no_grad():
                max_q_next = q_target(s2b_t).max(dim=1, keepdim=True)[0]
                target = rb_t + cfg.gamma * max_q_next * (1.0 - doneb_t)

            loss = loss_fn(q_sa, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            optimizer.step()
            losses.append(loss.item())

        if step % cfg.atari_target_update_freq == 0:
            q_target.load_state_dict(q.state_dict())

        # fin de episodio
        if done:
            episode += 1
            episode_rewards.append(ep_reward)
            episode_epsilons.append(eps)
            avg_20 = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
            print(f"Ep {episode:04d} | step={step:7d}/{cfg.atari_max_steps_total} | R={ep_reward:7.1f} | avg20={avg_20:7.1f} | eps={eps:.3f} | replay={len(replay)}")

            obs, _ = env.reset()
            ep_reward = 0.0

    env.close()
    return episode_rewards, episode_epsilons, losses



