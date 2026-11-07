from dataclasses import dataclass


@dataclass
class TrainState:
    """Per-run training counters; each instance owns its own state."""
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


def learning_rate(step: int, model_size: int, factor: float, warmup: int) -> float:
    """Noam learning-rate schedule from the original Transformer paper."""
    step = max(step, 1)
    return factor * (model_size ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)
