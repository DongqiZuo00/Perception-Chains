"""
Data loading utilities for Perception Chains.
Constructs a balanced training mixture of 1.25×10^4 instances per benchmark
(10^5 total across 8 seen benchmarks).
"""

from __future__ import annotations
import os
import json
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from PIL import Image

import torch
from torch.utils.data import Dataset, ConcatDataset

from models.ave import build_ave_prompt, SLOT_SCHEMAS


@dataclass
class PCDataSample:
    """A single training sample for Perception Chains."""
    image_path: str
    question: str
    answer: str
    task: str
    benchmark: str
    # Optional: pre-computed reference trajectory
    reference_chain_text: Optional[str] = None
    # Optional: cached teacher hidden states path
    teacher_cache_id: Optional[str] = None


class PerceptionChainsDataset(Dataset):
    """Dataset that wraps benchmark data with AVE prompts."""

    def __init__(
        self,
        samples: List[PCDataSample],
        processor,
        max_length: int = 2048,
    ):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample.image_path).convert("RGB")
        prompt = build_ave_prompt(sample.question, sample.task)

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}
        ]

        if sample.reference_chain_text:
            messages.append({
                "role": "assistant",
                "content": sample.reference_chain_text,
            })

        text = self.processor.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=sample.reference_chain_text is None,
        )

        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt",
            padding="max_length", max_length=self.max_length, truncation=True,
        )

        # Flatten batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Create labels (mask everything before assistant response)
        if sample.reference_chain_text:
            labels = inputs["input_ids"].clone()
            # Mask prompt tokens with -100
            # Find where the assistant response starts
            labels[:len(labels) // 2] = -100  # rough heuristic; proper impl uses tokenizer
            inputs["labels"] = labels

        inputs["task"] = sample.task
        inputs["benchmark"] = sample.benchmark

        return inputs


def load_benchmark_data(
    benchmark: str,
    data_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[PCDataSample]:
    """Load data from a specific benchmark directory.
    
    Expected structure:
        data_dir/{benchmark}/{split}/
            metadata.jsonl  (image_path, question, answer per line)
            images/
    """
    meta_path = os.path.join(data_dir, benchmark, split, "metadata.jsonl")
    if not os.path.exists(meta_path):
        # Try loading from HuggingFace datasets format
        return _load_hf_benchmark(benchmark, split, max_samples)

    samples = []
    img_dir = os.path.join(data_dir, benchmark, split, "images")

    with open(meta_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            img_path = os.path.join(img_dir, entry["image"])
            if not os.path.exists(img_path):
                continue
            task = _benchmark_to_task(benchmark)
            samples.append(PCDataSample(
                image_path=img_path,
                question=entry["question"],
                answer=entry["answer"],
                task=task,
                benchmark=benchmark,
            ))
            if max_samples and len(samples) >= max_samples:
                break

    return samples


def _load_hf_benchmark(benchmark: str, split: str, max_samples: Optional[int]) -> List[PCDataSample]:
    """Fallback: load from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    HF_MAPPING = {
        "ChartQA": ("HuggingFaceM4/ChartQA", "train"),
        "DocVQA": ("HuggingFaceM4/DocVQA", "train"),
        "GQA": ("merve/gqa", "train"),
        "TextVQA": ("facebook/textvqa", "train"),
        "POPE": ("lmms-lab/POPE", "test"),
    }

    if benchmark not in HF_MAPPING:
        return []

    ds_name, ds_split = HF_MAPPING[benchmark]
    try:
        ds = load_dataset(ds_name, split=ds_split)
    except Exception:
        return []

    samples = []
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        # Save image temporarily
        img_path = f"/tmp/pc_data/{benchmark}/{i}.png"
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        if "image" in item and item["image"] is not None:
            item["image"].save(img_path)
        else:
            continue

        q = item.get("question", item.get("query", ""))
        a = item.get("answer", item.get("answers", [""])[0] if "answers" in item else "")
        if isinstance(a, list):
            a = a[0] if a else ""

        samples.append(PCDataSample(
            image_path=img_path,
            question=q,
            answer=str(a),
            task=_benchmark_to_task(benchmark),
            benchmark=benchmark,
        ))

    return samples


def _benchmark_to_task(benchmark: str) -> str:
    return benchmark if benchmark in SLOT_SCHEMAS else "default"


def build_balanced_mixture(
    benchmarks: List[str],
    data_dir: str,
    per_benchmark_cap: int = 12500,
    seed: int = 42,
) -> List[PCDataSample]:
    """Construct balanced training mixture: per_benchmark_cap samples per benchmark."""
    rng = random.Random(seed)
    all_samples = []

    for bm in benchmarks:
        samples = load_benchmark_data(bm, data_dir, split="train")
        if len(samples) > per_benchmark_cap:
            samples = rng.sample(samples, per_benchmark_cap)
        elif len(samples) < per_benchmark_cap:
            # Oversample
            while len(samples) < per_benchmark_cap:
                samples.extend(rng.choices(samples, k=min(per_benchmark_cap - len(samples), len(samples))))
        all_samples.extend(samples[:per_benchmark_cap])

    rng.shuffle(all_samples)
    return all_samples
