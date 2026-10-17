from typing import Tuple

import numpy as np

from ..core.types import BoundingBox


class KalmanBoxFilter:
    def __init__(
        self,
        bbox: BoundingBox,
        process_noise: float = 4.0,
        measurement_noise: float = 4.0,
    ):
        self.x = np.array(
            [bbox.cx, bbox.cy, bbox.width, bbox.height, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0])
        self.q = process_noise
        self.R = np.eye(4) * measurement_noise
        self.H = np.hstack([np.eye(4), np.zeros((4, 4))])

    def predict(self, dt: float) -> None:
        if dt <= 0.0:
            return
        F = np.eye(8)
        for i in range(4):
            F[i, i + 4] = dt
        Q = np.diag([self.q * dt * dt] * 4 + [self.q * dt] * 4)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)

    def update(self, bbox: BoundingBox) -> None:
        z = np.array([bbox.cx, bbox.cy, bbox.width, bbox.height], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)

    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox.from_center(*self.x[:4])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[4]), float(self.x[5])
