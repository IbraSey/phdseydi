"""Common inference contracts and mutable Gibbs state."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from ..config import ETASParameters, GPParameters
from ..data.catalog import EventCatalog


class InferenceMethod(ABC):
    """Common interface for inference algorithms."""

    @abstractmethod
    def fit(self, catalog: EventCatalog):
        """Fit the configured model to an event catalog."""



@dataclass
class GibbsState:
    eps: np.ndarray
    f_data: np.ndarray
    gp_parameters: GPParameters
    branching: np.ndarray | None = None
    etas_parameters: ETASParameters | None = None
    beta: float | None = None
    gp_latent_state: object | None = None

