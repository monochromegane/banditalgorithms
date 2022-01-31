from typing import Protocol


class BanditType(Protocol):
    def select(self) -> int:
        ...

    def update(self, idx_arm: int, reward: float) -> None:
        ...
