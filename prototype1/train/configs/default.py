from dataclasses import dataclass

@dataclass
class TrainConfig:
    epochs: int = 100
    