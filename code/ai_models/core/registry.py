import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..constants import WEIGHT_SUFFIXES


@dataclass(frozen=True)
class ModelInfo:
    name: str
    path: str
    size_bytes: int
    modified: float
    sha256: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "modified": self.modified,
            "sha256": self.sha256,
        }


class ModelRegistry:
    def __init__(self, weights_dir: Path):
        self.weights_dir = Path(weights_dir)
        self._cache: Dict[str, Tuple[Tuple[int, float], str]] = {}
        self._lock = threading.Lock()

    def _digest(self, path: Path, stat_key: Tuple[int, float]) -> str:
        key = str(path)
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached[0] == stat_key:
                return cached[1]
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        with self._lock:
            self._cache[key] = (stat_key, digest)
        return digest

    def describe(self, path: Path) -> Optional[ModelInfo]:
        path = Path(path)
        if not path.is_file():
            return None
        st = path.stat()
        return ModelInfo(
            name=path.name,
            path=str(path),
            size_bytes=st.st_size,
            modified=st.st_mtime,
            sha256=self._digest(path, (st.st_size, st.st_mtime)),
        )

    def list_models(self) -> List[ModelInfo]:
        if not self.weights_dir.is_dir():
            return []
        infos = []
        for p in sorted(self.weights_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in WEIGHT_SUFFIXES:
                info = self.describe(p)
                if info:
                    infos.append(info)
        return infos
