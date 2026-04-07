"""
Evaluation module for Perception Chains.
Metrics: Acc, GA@IoU≥0.5, HR, ANLS, F1, Recall@1.
"""

from __future__ import annotations
import json
import logging
import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from tqdm import tqdm
from PIL import Image

from models.ave import parse_anchor_chain, build_ave_prompt
from inference.prl import PerceptualRestoringLoop, ConsistencyChecker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------
def compute_accuracy(predictions: List[str], references: List[str]) -> float:
    correct = sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references))
    return correct / max(len(predictions), 1)


def compute_relaxed_accuracy(predictions: List[str], references: List[str], tolerance: float = 0.05) -> float:
    """ChartQA relaxed accuracy: correct if within ±tolerance of numeric answer."""
    correct = 0
    for p, r in zip(predictions, references):
        try:
            pv, rv = float(p), float(r)
            if abs(pv - rv) <= tolerance * abs(rv + 1e-8):
                correct += 1
        except ValueError:
            if p.strip().lower() == r.strip().lower():
                correct += 1
    return correct / max(len(predictions), 1)


def compute_anls(predictions: List[str], references: List[str]) -> float:
    """Average Normalised Levenshtein Similarity for DocVQA."""
    def _nls(pred, ref):
        if not pred or not ref:
            return 0.0
        pred, ref = pred.lower(), ref.lower()
        m, n = len(pred), len(ref)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = min(
                    dp[i-1][j] + 1, dp[i][j-1] + 1,
                    dp[i-1][j-1] + (0 if pred[i-1] == ref[j-1] else 1)
                )
        dist = dp[m][n]
        nls = 1.0 - dist / max(m, n)
        return nls if nls >= 0.5 else 0.0

    scores = [_nls(p, r) for p, r in zip(predictions, references)]
    return sum(scores) / max(len(scores), 1)


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-8)


def compute_grounding_accuracy(
    pred_boxes: List[List[float]],
    gt_boxes: List[List[float]],
    iou_threshold: float = 0.5,
) -> float:
    correct = sum(compute_iou(p, g) >= iou_threshold for p, g in zip(pred_boxes, gt_boxes))
    return correct / max(len(pred_boxes), 1)


def compute_hallucination_rate(predictions: List[str], references: List[str]) -> float:
    """HR = 1 - F1 (for POPE-style binary yes/no)."""
    tp = fp = fn = 0
    for p, r in zip(predictions, references):
        p, r = p.strip().lower(), r.strip().lower()
        if r == "yes":
            if p == "yes":
                tp += 1
            else:
                fn += 1
        else:
            if p == "yes":
                fp += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return 1.0 - f1


METRIC_FNS = {
    "ChartQA": lambda p, r: compute_relaxed_accuracy(p, r),
    "DocVQA": compute_anls,
    "RSVQA": compute_accuracy,
    "DIOR-RSVG": None,  # special: grounding accuracy
    "MuMuQA": compute_accuracy,
    "MMIU": compute_accuracy,
    "POPE": compute_hallucination_rate,
    "HallusionBench": compute_hallucination_rate,
    "GQA": compute_accuracy,
    "TextVQA": compute_accuracy,
    "AI2D": compute_accuracy,
    "ObjHalBench": None,  # F1
    "Hallucinogen": compute_accuracy,
    "RefCOCO": None,  # grounding
    "Flickr30k": None,  # recall@1
}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class Evaluator:
    """Run evaluation across benchmarks."""

    def __init__(
        self,
        model,
        processor,
        use_prl: bool = True,
        prl_config: Optional[dict] = None,
    ):
        self.model = model
        self.processor = processor
        self.use_prl = use_prl

        if use_prl:
            checker = ConsistencyChecker(nli_threshold=prl_config.get("nli_threshold", 0.5) if prl_config else 0.5)
            self.prl = PerceptualRestoringLoop(
                model=model,
                processor=processor,
                consistency_checker=checker,
                n_vote=prl_config.get("n_vote", 4) if prl_config else 4,
                tau_samp=prl_config.get("tau_samp", 0.7) if prl_config else 0.7,
                t_max=prl_config.get("t_max", 2) if prl_config else 2,
                tau_prl=prl_config.get("tau_prl", 0.0) if prl_config else 0.0,
            )

    @torch.no_grad()
    def evaluate_benchmark(
        self,
        benchmark: str,
        samples: List[dict],
    ) -> Dict[str, float]:
        """Evaluate a single benchmark."""
        predictions = []
        references = []
        pred_boxes = []
        gt_boxes = []
        prl_stats = {"triggered": 0, "total": 0}

        for sample in tqdm(samples, desc=f"Eval {benchmark}"):
            image = Image.open(sample["image_path"]).convert("RGB")
            question = sample["question"]
            gt = sample["answer"]
            task = sample.get("task", "default")
            task_type = "closed" if benchmark in ("ChartQA", "POPE", "RSVQA") else "open"

            if self.use_prl:
                answer, chain, meta = self.prl(image, question, task, task_type)
                prl_stats["total"] += 1
                if meta.get("prl_triggered"):
                    prl_stats["triggered"] += 1
            else:
                prompt = build_ave_prompt(question, task)
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]}]
                text_input = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(text=[text_input], images=[image], return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                gen_ids = self.model.generate(**inputs, max_new_tokens=1024)
                text = self.processor.batch_decode(
                    gen_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )[0]
                chain = parse_anchor_chain(text, task)
                answer = chain.answer or ""

            predictions.append(answer)
            references.append(gt)

            # Collect boxes for grounding metrics
            if "gt_bbox" in sample and chain.anchors:
                pred_boxes.append(chain.anchors[0].bbox)
                gt_boxes.append(sample["gt_bbox"])

        # Compute metrics
        results = {}
        metric_fn = METRIC_FNS.get(benchmark)
        if metric_fn:
            score = metric_fn(predictions, references)
            metric_name = "HR" if "hallucin" in benchmark.lower() or benchmark == "POPE" else "Acc"
            results[metric_name] = score

        if pred_boxes and gt_boxes:
            results["GA@IoU>=0.5"] = compute_grounding_accuracy(pred_boxes, gt_boxes)

        if prl_stats["total"] > 0:
            results["PRL_trigger_rate"] = prl_stats["triggered"] / prl_stats["total"]

        return results

    def evaluate_all(
        self,
        benchmark_data: Dict[str, List[dict]],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate across all benchmarks."""
        all_results = {}
        for bm, samples in benchmark_data.items():
            logger.info(f"Evaluating {bm} ({len(samples)} samples)")
            results = self.evaluate_benchmark(bm, samples)
            all_results[bm] = results
            logger.info(f"  {bm}: {results}")

        return all_results
