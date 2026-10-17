from typing import List, Tuple

import numpy as np

Tile = Tuple[int, int, int, int]


def _starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def generate_tiles(
    frame_shape: Tuple[int, ...], tile_size: int, overlap: float
) -> List[Tile]:
    height, width = frame_shape[:2]
    stride = max(1, int(tile_size * (1.0 - overlap)))
    tile_w = min(tile_size, width)
    tile_h = min(tile_size, height)
    tiles: List[Tile] = []
    for y in _starts(height, tile_h, stride):
        for x in _starts(width, tile_w, stride):
            tiles.append((x, y, x + tile_w, y + tile_h))
    return tiles


def crop(frame: np.ndarray, tile: Tile) -> np.ndarray:
    x1, y1, x2, y2 = tile
    return np.ascontiguousarray(frame[y1:y2, x1:x2])
