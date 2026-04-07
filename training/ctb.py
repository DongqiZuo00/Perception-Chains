"""
Confidence-Weighted Trajectory Bootstrapping (CTB) — §3.3

Phase 1 (Cold start): Frozen 72B teacher generates k=8 candidates per sample;
  retain only if all k agree on final answer (exact match).
Phase 2 (Self-bootstrap): 7B student generates candidates, scored by conf(C_j).
  conf(C_j) = a_bar(C_j) * crr(C_j) * tsc(C_j)
"""

from __future__ import annotations
import os
import json
import torch
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

from models.ave import (
    AnchorChain, AnchorNode, parse_anchor_chain,
    anchor_agreement, build_ave_prompt, K_SLOTS,
)

logger = logging.getLogger(__name__)


@dataclass
class CachedTrajectory:
    """A cached teacher trajectory with hidden states."""
    sample_id: str
    chain: AnchorChain
    answer: str
    hidden_states: Optional[Dict[int, Dict[int, torch.Tensor]]] = None  # {layer: {pos: tensor}}
    confidence: float = 1.0


@dataclass
class CTBCache:
    """Collection of cached reference trajectories T*."""
    trajectories: List[CachedTrajectory] = field(default_factory=list)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = []
        for t in self.trajectories:
            entry = {
                "sample_id": t.sample_id,
                "answer": t.answer,
                "confidence": t.confidence,
                "anchors": [
                    {"bbox": a.bbox, "attribute": a.attribute,
                     "slot": a.slot, "slot_idx": a.slot_idx}
                    for a in t.chain.anchors
                ],
            }
            data.append(entry)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        # Save hidden states separately as a torch file
        hs_path = path.replace(".json", "_hidden_states.pt")
        hs_data = {}
        for i, t in enumerate(self.trajectories):
            if t.hidden_states is not None:
                hs_data[i] = t.hidden_states
        if hs_data:
            torch.save(hs_data, hs_path)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        hs_path = path.replace(".json", "_hidden_states.pt")
        hs_data = {}
        if os.path.exists(hs_path):
            hs_data = torch.load(hs_path, map_location="cpu")

        self.trajectories = []
        for i, entry in enumerate(data):
            chain = AnchorChain(
                anchors=[
                    AnchorNode(**a) for a in entry["anchors"]
                ],
                answer=entry["answer"],
            )
            self.trajectories.append(CachedTrajectory(
                sample_id=entry["sample_id"],
                chain=chain,
                answer=entry["answer"],
                hidden_states=hs_data.get(i),
                confidence=entry.get("confidence", 1.0),
            ))


def phase1_cold_start(
    teacher_model,
    dataset,
    k: int = 8,
    tau: float = 0.7,
    top_p: float = 0.9,
    max_samples: int = 5000,
) -> CTBCache:
    """Phase 1: Generate k chains per sample from frozen 72B teacher.
    Retain only chains where all k samples agree on the final answer."""
    cache = CTBCache()
    processor = teacher_model.processor

    for idx, sample in enumerate(tqdm(dataset[:max_samples], desc="CTB Phase 1")):
        image = sample["image"]
        question = sample["question"]
        task = sample.get("task", "default")
        gt_answer = sample.get("answer", None)

        prompt = build_ave_prompt(question, task)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_input], images=[image], return_tensors="pt")
        inputs = {k: v.to(teacher_model.model.device) for k, v in inputs.items()}

        # Generate k candidates
        chain_texts = teacher_model.generate_chains(inputs, k=k, tau=tau, top_p=top_p)
        chains = [parse_anchor_chain(t, task) for t in chain_texts]
        answers = [c.answer for c in chains if c.answer]

        # Retain only if all k agree via exact match
        if len(answers) == k and len(set(answers)) == 1:
            best_chain = chains[0]
            cache.trajectories.append(CachedTrajectory(
                sample_id=str(idx),
                chain=best_chain,
                answer=answers[0],
                confidence=1.0,
            ))

    logger.info(f"Phase 1: retained {len(cache.trajectories)}/{max_samples} trajectories")
    return cache


def compute_confidence(
    chain: AnchorChain,
    slot_values_across_samples: List[List[str]],
    gt_answer: Optional[str],
    teacher_score: float,
) -> float:
    """Compute conf(C_j) = a_bar * crr * tsc (Eq. 4)."""
    a_bar = anchor_agreement(slot_values_across_samples)
    crr = 1.0 if gt_answer is None else float(chain.answer == gt_answer)
    tsc = teacher_score
    return a_bar * crr * tsc


def phase2_self_bootstrap(
    student_model,
    teacher_model,
    dataset,
    phase1_cache: CTBCache,
    k: int = 8,
    tau: float = 0.7,
    tau_conf: float = 0.5,
    rounds: int = 2,
) -> CTBCache:
    """Phase 2: Student generates candidates, scored by conf(C_j).
    Two rounds of self-bootstrap."""
    cache = CTBCache(trajectories=list(phase1_cache.trajectories))
    processor = student_model.processor

    for round_idx in range(rounds):
        logger.info(f"Phase 2, round {round_idx + 1}/{rounds}")
        new_trajectories = []

        for idx, sample in enumerate(tqdm(dataset, desc=f"CTB Phase 2 round {round_idx+1}")):
            image = sample["image"]
            question = sample["question"]
            task = sample.get("task", "default")
            gt_answer = sample.get("answer", None)

            prompt = build_ave_prompt(question, task)
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]}
            ]
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_input], images=[image], return_tensors="pt")
            inputs = {k_: v.to(student_model.device) for k_, v in inputs.items()}

            # Generate k chains from student
            chain_texts = []
            for _ in range(k):
                gen_ids = student_model.generate(
                    **inputs, max_new_tokens=1024,
                    do_sample=True, temperature=tau, top_p=0.9,
                )
                text = processor.batch_decode(
                    gen_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )[0]
                chain_texts.append(text)

            chains = [parse_anchor_chain(t, task) for t in chain_texts]

            # Collect slot values across samples for anchor agreement
            slot_values = [[] for _ in range(K_SLOTS)]
            for c in chains:
                for a in c.anchors[:K_SLOTS]:
                    slot_values[a.slot_idx].append(a.attribute)

            # Score each chain
            for c in chains:
                if c.answer is None:
                    continue
                # Teacher score: binary correctness (simplified)
                tsc = 1.0  # In practice: query 72B verifier
                conf = compute_confidence(c, slot_values, gt_answer, tsc)
                if conf >= tau_conf:
                    new_trajectories.append(CachedTrajectory(
                        sample_id=f"{idx}_r{round_idx}",
                        chain=c,
                        answer=c.answer,
                        confidence=conf,
                    ))

        cache.trajectories.extend(new_trajectories)
        logger.info(
            f"Round {round_idx+1}: added {len(new_trajectories)} trajectories, "
            f"total {len(cache.trajectories)}"
        )

    return cache
