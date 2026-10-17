from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..core.types import BoundingBox


@dataclass(frozen=True)
class ScoredBox:
    bbox: BoundingBox
    confidence: float
    label: str


@dataclass(frozen=True)
class EvaluationResult:
    precision: float
    recall: float
    f1: float
    average_precision: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "average_precision": round(self.average_precision, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


def match_predictions(
    predictions: Sequence[ScoredBox],
    ground_truth: Sequence[BoundingBox],
    iou_threshold: float = 0.5,
) -> List[Tuple[float, bool]]:
    ordered = sorted(predictions, key=lambda p: p.confidence, reverse=True)
    taken = set()
    outcomes: List[Tuple[float, bool]] = []
    for pred in ordered:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(ground_truth):
            if idx in taken:
                continue
            iou = pred.bbox.iou(gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            taken.add(best_idx)
            outcomes.append((pred.confidence, True))
        else:
            outcomes.append((pred.confidence, False))
    return outcomes


def average_precision(outcomes: Sequence[Tuple[float, bool]], total_gt: int) -> float:
    if total_gt == 0 or not outcomes:
        return 0.0
    ordered = sorted(outcomes, key=lambda o: o[0], reverse=True)
    tp = np.cumsum([1 if hit else 0 for _, hit in ordered])
    fp = np.cumsum([0 if hit else 1 for _, hit in ordered])
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    idx = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1]))


def evaluate_detections(
    predictions_per_image: Sequence[Sequence[ScoredBox]],
    ground_truth_per_image: Sequence[Sequence[BoundingBox]],
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.0,
) -> EvaluationResult:
    if len(predictions_per_image) != len(ground_truth_per_image):
        raise ValueError("predictions and ground truth must align per image")
    all_outcomes: List[Tuple[float, bool]] = []
    total_gt = 0
    tp = fp = 0
    for preds, gts in zip(predictions_per_image, ground_truth_per_image, strict=True):
        total_gt += len(gts)
        outcomes = match_predictions(preds, gts, iou_threshold)
        all_outcomes.extend(outcomes)
        for conf, hit in outcomes:
            if conf >= confidence_threshold:
                if hit:
                    tp += 1
                else:
                    fp += 1
    fn = total_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EvaluationResult(
        precision=precision,
        recall=recall,
        f1=f1,
        average_precision=average_precision(all_outcomes, total_gt),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )
