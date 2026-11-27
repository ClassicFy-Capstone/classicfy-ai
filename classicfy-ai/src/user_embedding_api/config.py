"""Configuration shared by HTTP and numerical calculation layers."""

from dataclasses import dataclass, field
import math
import os


@dataclass(frozen=True)
class Settings:
    decayAlpha: float = 0.95
    apiKey: str | None = field(default=None, repr=False)

    def __post_init__(self):
        if not math.isfinite(self.decayAlpha) or not 0 < self.decayAlpha <= 1:
            raise ValueError("USER_EMBEDDING_DECAY_ALPHA must be in (0, 1]")
        if self.apiKey is not None and not self.apiKey.strip():
            raise ValueError("AI_API_KEY must not be empty when configured")

    @classmethod
    def fromEnv(cls):
        return cls(
            decayAlpha=float(os.environ.get("USER_EMBEDDING_DECAY_ALPHA", "0.95")),
            apiKey=os.environ.get("AI_API_KEY"),
        )
