from typing import List, Protocol


class BanditType(Protocol):
    def select(self) -> int:
        ...

    def update(self, idx_arm: int, reward: float) -> None:
        ...


class ContextualBanditType(Protocol):
    def select(self, ctx: List[float]) -> int:
        ...

    def update(self, idx_arm: int, reward: float, ctx: List[float]) -> None:
        ...
