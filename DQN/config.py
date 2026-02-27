from dataclasses import dataclass


@dataclass
class Config:
    env_id: str = "ALE/Asteroids-v5" # gym.make("ALE/NombreDelJuego-v5", render_mode="human") para juegos de Atari
    seed: int = 666
    render: bool = True

    # Entrenamiento
    max_episodes: int = 100
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
    
    # Atari extras (se usan solo si env_id inicia con "ALE/")
    atari_frame_stack: int = 4
    atari_min_buffer_size: int = 20_000
    atari_batch_size: int = 32
    atari_lr: float = 1e-4
    atari_train_freq: int = 4
    atari_target_update_freq: int = 10_000
    atari_max_steps_total: int = 500_000

    # Terminar antes si se "resuelve"
    solved_avg_reward: float = 475.0
    solved_window: int = 100


