from dataclasses import dataclass

@dataclass
class Config:
    env_id: str = "CartPole-v1" # gym.make("ALE/NombreDelJuego-v5", render_mode="human") para juegos de Atari
    seed: int = 42

    # Entrenamiento
    max_episodes: int = 800
    max_steps_per_episode: int = 500
    gamma: float = 0.99

    # Replay Buffer
    buffer_size: int = 100_000
    batch_size: int = 64
    min_buffer_size: int = 2_000

    # Optimizador
    lr: float = 1e-3
    train_freq: int = 1                 # entrenar cada N pasos
    target_update_freq: int = 1_000     # copiar pesos a target cada N pasos

    # Epsilon-greedy
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000       # decaimiento lineal en pasos

    # Terminar antes si se "resuelve"
    solved_avg_reward: float = 475.0
    solved_window: int = 100


