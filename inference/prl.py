"""
Perceptual Restoring Loop (PRL) — §3.4

Inference-time consistency check and targeted anchor re-localisation.
1. Vote: sample N_vote=4 chains, majority-vote answer ŷ^(0)
2. Select: choose chain with highest a_bar consistent with ŷ^(0)
3. Identify: check consistency; if inconsistent, find conflicting anchors I
4. Repair: re-localise only anchors in I; repeat ≤ T_max=2 times
"""

from __future__ import annotations
import torch
import logging
from typing import List, Optional, Tuple, Dict
from collections import Counter

from models.ave import (
    AnchorChain, AnchorNode, parse_anchor_chain,
    anchor_agreement, build_ave_prompt, K_SLOTS,
)

logger = logging.getLogger(__name__)


class ConsistencyChecker:
    """Check whether an anchor chain is consistent with a provisional answer.
    
    Closed-ended tasks: exact string match.
    Open-ended tasks: NLI entailment via DeBERTa-v3-large.
    """

    def __init__(self, nli_model=None, nli_tokenizer=None, nli_threshold: float = 0.5):
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer
        self.nli_threshold = nli_threshold

    def check_closed(self, chain: AnchorChain, answer: str) -> Tuple[bool, List[int]]:
        """Exact string match for closed-ended tasks."""
        inconsistent_slots = []
        for a in chain.anchors:
            # Check if the anchor's attribute is compatible with the answer
            if a.attribute and answer:
                # Simple heuristic: if the attribute directly contradicts
                if a.slot == "answer_slot" and a.attribute.lower() != answer.lower():
                    inconsistent_slots.append(a.slot_idx)
        return len(inconsistent_slots) == 0, inconsistent_slots

    def check_open(self, chain: AnchorChain, answer: str) -> Tuple[bool, List[int]]:
        """NLI-based consistency for open-ended tasks."""
        if self.nli_model is None:
            return self.check_closed(chain, answer)

        inconsistent_slots = []
        for a in chain.anchors:
            if not a.attribute:
                continue
            premise = f"The answer is {answer}."
            hypothesis = f"Based on region {a.bbox}, the {a.slot} is {a.attribute}."

            inputs = self.nli_tokenizer(
                premise, hypothesis, return_tensors="pt", truncation=True,
            ).to(next(self.nli_model.parameters()).device)

            with torch.no_grad():
                logits = self.nli_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                # Assuming label 2 = contradiction for DeBERTa NLI
                contradiction_prob = probs[0, 2].item()

            if contradiction_prob > self.nli_threshold:
                inconsistent_slots.append(a.slot_idx)

        return len(inconsistent_slots) == 0, inconsistent_slots


class PerceptualRestoringLoop:
    """PRL inference-time correction mechanism."""

    def __init__(
        self,
        model,
        processor,
        consistency_checker: ConsistencyChecker,
        n_vote: int = 4,
        tau_samp: float = 0.7,
        t_max: int = 2,
        tau_prl: float = 0.0,
    ):
        self.model = model
        self.processor = processor
        self.checker = consistency_checker
        self.n_vote = n_vote
        self.tau_samp = tau_samp
        self.t_max = t_max
        self.tau_prl = tau_prl

    def _generate_chain(self, inputs: dict, task: str, do_sample: bool = True) -> AnchorChain:
        """Generate a single anchor chain."""
        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = self.tau_samp
            gen_kwargs["top_p"] = 0.9
        else:
            gen_kwargs["do_sample"] = False

        gen_ids = self.model.generate(**inputs, **gen_kwargs)
        text = self.processor.batch_decode(
            gen_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]
        return parse_anchor_chain(text, task)

    def _build_repair_prompt(
        self,
        question: str,
        chain: AnchorChain,
        inconsistent_indices: List[int],
        answer: str,
        task: str,
    ) -> str:
        """Build a repair prompt that re-localises only the inconsistent anchors."""
        prompt = (
            f"You previously answered the question and found some inconsistencies. "
            f"Re-examine only the following slots, using the provisional answer "
            f"'{answer}' as a guide.\n\n"
            f"Question: {question}\n\n"
        )
        for idx in inconsistent_indices:
            if idx < len(chain.anchors):
                a = chain.anchors[idx]
                prompt += (
                    f"Slot {idx+1} ({a.slot}): Previous region {a.bbox}, "
                    f"attribute '{a.attribute}' — INCONSISTENT. "
                    f"Re-localise this slot:\n"
                )
        prompt += "\nProvide corrected slots with <box> regions and updated attributes:\n"
        return prompt

    @torch.no_grad()
    def __call__(
        self,
        image,
        question: str,
        task: str = "default",
        task_type: str = "closed",
    ) -> Tuple[str, AnchorChain, Dict]:
        """Run PRL inference pipeline.
        
        Returns: (final_answer, final_chain, metadata)
        """
        prompt = build_ave_prompt(question, task)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}
        ]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text_input], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Step 1: Vote — sample N_vote chains
        chains = []
        for _ in range(self.n_vote):
            c = self._generate_chain(inputs, task, do_sample=True)
            chains.append(c)

        # Majority vote
        answers = [c.answer for c in chains if c.answer]
        if not answers:
            # Fallback to greedy
            c = self._generate_chain(inputs, task, do_sample=False)
            return c.answer or "", c, {"prl_triggered": False, "iterations": 0}

        vote_counts = Counter(answers)
        y_hat = vote_counts.most_common(1)[0][0]

        # Step 2: Select — chain with highest a_bar consistent with ŷ^(0)
        best_chain = None
        best_score = -1
        for c in chains:
            if c.answer == y_hat:
                slot_vals = [[a.attribute] for a in c.anchors]
                score = anchor_agreement(slot_vals)
                if score > best_score:
                    best_score = score
                    best_chain = c

        if best_chain is None:
            best_chain = chains[0]

        metadata = {"prl_triggered": False, "iterations": 0, "inconsistent_slots": []}

        # Steps 3-4: Identify and Repair (up to T_max iterations)
        current_chain = best_chain
        current_answer = y_hat

        for t in range(self.t_max):
            # Check consistency
            check_fn = self.checker.check_closed if task_type == "closed" else self.checker.check_open
            consistent, inconsistent_ids = check_fn(current_chain, current_answer)

            if consistent:
                break

            # Apply PRL threshold: skip if confidence is above threshold
            if self.tau_prl > 0:
                inconsistency_ratio = len(inconsistent_ids) / max(len(current_chain.anchors), 1)
                if inconsistency_ratio < self.tau_prl:
                    break

            metadata["prl_triggered"] = True
            metadata["iterations"] = t + 1
            metadata["inconsistent_slots"] = inconsistent_ids

            # Repair: re-localise only inconsistent anchors
            repair_prompt = self._build_repair_prompt(
                question, current_chain, inconsistent_ids, current_answer, task
            )
            repair_messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": repair_prompt},
                ]}
            ]
            repair_text = self.processor.apply_chat_template(
                repair_messages, tokenize=False, add_generation_prompt=True
            )
            repair_inputs = self.processor(
                text=[repair_text], images=[image], return_tensors="pt"
            )
            repair_inputs = {k: v.to(self.model.device) for k, v in repair_inputs.items()}

            repaired = self._generate_chain(repair_inputs, task, do_sample=False)

            # Merge: update only inconsistent slots
            for idx in inconsistent_ids:
                if idx < len(repaired.anchors) and idx < len(current_chain.anchors):
                    current_chain.anchors[idx] = repaired.anchors[idx]

            current_answer = repaired.answer or current_answer

        return current_answer, current_chain, metadata
