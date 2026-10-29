"""Shared embedding guards for interfaces with engine-owned coupling."""
from __future__ import annotations

from typing import ClassVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydft_qmmm.potentials import ElectronicPotential


class EngineEmbeddingMixin:
    """Manage embedding state while adapters supply backend requirements.

    Fields stay on the concrete interfaces to preserve their dataclass
    constructors. Error messages describe each engine's embedding backend.
    """

    embedding: bool
    potentials: list[ElectronicPotential]
    _embedding_conflict_message: ClassVar[str]
    _embedding_unavailable_message: ClassVar[str]

    def configure_electrostatic_embedding(self, enabled: bool) -> None:
        """Enable QM electrostatics or reject conflicting manual embedding."""
        if enabled:
            # Interfaces are frozen, but coupling configuration is finalized
            # while the composite calculator is being built.
            object.__setattr__(self, "embedding", True)
        elif self.embedding:
            raise ValueError(self._embedding_conflict_message)

    def add_electronic_potential(self, potential: ElectronicPotential) -> None:
        """Register a potential only when the engine embedding is enabled."""
        if not self.embedding:
            raise NotImplementedError(self._embedding_unavailable_message)
        self.potentials.append(potential)
