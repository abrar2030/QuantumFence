from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .types import RawDetection


class BaseDetector(ABC):
    name: str = "detector"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def predict(
        self,
        frame: np.ndarray,
        classes: Optional[Sequence[int]] = None,
        conf: Optional[float] = None,
    ) -> List[RawDetection]: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abstractmethod
    def class_names(self) -> Dict[int, str]: ...

    def unload(self) -> None:
        return None

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "loaded": self.is_loaded}
