"""
Anchored Visual Evidence (AVE) — §3.1
Each reasoning step emits a triple a_i = (b_i, v_i, s_i):
  b_i: bounding box [x1, y1, x2, y2] in [0, 1000]
  v_i: structured attribute tuple
  s_i: reasoning slot assignment
"""

from __future__ import annotations
import re
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Slot schemas (task-specific; Table S2 in supplementary)
# ---------------------------------------------------------------------------
SLOT_SCHEMAS = {
    "ChartQA": ["data_region", "axis_label", "value_read", "comparison"],
    "DocVQA": ["field_region", "field_label", "field_value", "answer_slot"],
    "RSVQA": ["object_region", "object_class", "attribute", "answer_slot"],
    "DIOR-RSVG": ["target_region", "target_class", "spatial_relation", "grounding"],
    "MuMuQA": ["image_region", "entity", "relation", "answer_slot"],
    "MMIU": ["region_1", "region_2", "comparison_attr", "answer_slot"],
    "POPE": ["object_region", "object_class", "presence", "answer_slot"],
    "HallusionBench": ["region", "attribute", "consistency", "answer_slot"],
    # generic fallback
    "default": ["region", "attribute", "reasoning", "answer_slot"],
}

K_SLOTS = 4  # fixed across all tasks (§3.1)


@dataclass
class AnchorNode:
    """Visual anchor node a_i = (b_i, v_i, s_i)."""
    bbox: List[float]          # [x1, y1, x2, y2] normalised to [0,1000]
    attribute: str             # v_i — extracted attribute value
    slot: str                  # s_i — which reasoning slot this fills
    slot_idx: int = 0          # position in the chain (0..K-1)


@dataclass
class AnchorChain:
    """Ordered sequence C = (a_1, ..., a_K)."""
    anchors: List[AnchorNode] = field(default_factory=list)
    answer: Optional[str] = None

    @property
    def K(self) -> int:
        return len(self.anchors)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_ave_prompt(question: str, task: str = "default") -> str:
    """Build the instruction prefix that decomposes `question` into K sub-questions,
    each requiring a bounding box before any attribute extraction."""
    schema = SLOT_SCHEMAS.get(task, SLOT_SCHEMAS["default"])
    slots_desc = "\n".join(
        f"  Slot {i+1} ({s}): First output <box>[x1, y1, x2, y2]</box>, "
        f"then extract the attribute for this slot."
        for i, s in enumerate(schema)
    )
    prompt = (
        f"You are given an image and a question. Decompose the question into "
        f"exactly {K_SLOTS} sub-questions. For each sub-question, you MUST:\n"
        f"1. Name a spatial region as a bounding box <box>[x1, y1, x2, y2]</box> "
        f"   (coordinates in [0, 1000]).\n"
        f"2. Extract a visual attribute from that region.\n"
        f"3. Assign the attribute to the corresponding reasoning slot.\n\n"
        f"Slots:\n{slots_desc}\n\n"
        f"After filling all {K_SLOTS} slots, provide the final answer.\n\n"
        f"Question: {question}\n\n"
        f"Begin your step-by-step grounded reasoning:\n"
    )
    return prompt


# ---------------------------------------------------------------------------
# Parsing model outputs into AnchorChain
# ---------------------------------------------------------------------------
_BOX_RE = re.compile(
    r"<box>\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*</box>"
)
_SLOT_RE = re.compile(r"Slot\s+(\d+)\s*\(([^)]+)\)\s*:")
_ANSWER_RE = re.compile(r"(?:Final [Aa]nswer|ANSWER)\s*:\s*(.+?)(?:\n|$)")


def parse_anchor_chain(text: str, task: str = "default") -> AnchorChain:
    """Parse a model's textual output into a structured AnchorChain."""
    schema = SLOT_SCHEMAS.get(task, SLOT_SCHEMAS["default"])
    anchors: List[AnchorNode] = []
    boxes = _BOX_RE.findall(text)
    slots_found = _SLOT_RE.findall(text)

    # Split text by slot markers
    parts = _SLOT_RE.split(text)

    # Pair up: each slot section should contain a box and an attribute
    slot_sections = []
    i = 1
    while i < len(parts) - 1:
        slot_idx_str, slot_name, content = parts[i], parts[i+1], parts[i+2] if i+2 < len(parts) else ""
        slot_sections.append((int(slot_idx_str), slot_name.strip(), content))
        i += 3

    for idx, (slot_idx, slot_name, content) in enumerate(slot_sections):
        box_match = _BOX_RE.search(content)
        bbox = [float(x) for x in box_match.groups()] if box_match else [0, 0, 0, 0]
        # attribute: text between box close and next slot/answer marker
        attr_text = content
        if box_match:
            attr_text = content[box_match.end():].strip()
        # Take first line as attribute
        attr = attr_text.split("\n")[0].strip().rstrip(".")
        anchors.append(AnchorNode(
            bbox=bbox, attribute=attr, slot=slot_name, slot_idx=idx
        ))

    # Fallback: if slot-based parsing failed, try sequential box extraction
    if not anchors and boxes:
        for idx, (x1, y1, x2, y2) in enumerate(boxes[:K_SLOTS]):
            anchors.append(AnchorNode(
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                attribute="",
                slot=schema[idx] if idx < len(schema) else f"slot_{idx}",
                slot_idx=idx,
            ))

    answer_match = _ANSWER_RE.search(text)
    answer = answer_match.group(1).strip() if answer_match else None

    return AnchorChain(anchors=anchors, answer=answer)


# ---------------------------------------------------------------------------
# Anchor-level agreement score  a_bar(C) — Eq. in §3.3
# ---------------------------------------------------------------------------
import math

def anchor_agreement(slot_values_across_samples: List[List[str]]) -> float:
    """Compute normalised slot-level anchor agreement a_bar.
    
    Args:
        slot_values_across_samples: shape [K][k_samples] — sampled values per slot.
    Returns:
        a_bar in [0, 1].
    """
    K = len(slot_values_across_samples)
    if K == 0:
        return 0.0
    scores = []
    for slot_vals in slot_values_across_samples:
        # empirical entropy H_i
        counts: dict = {}
        for v in slot_vals:
            counts[v] = counts.get(v, 0) + 1
        n = len(slot_vals)
        if n <= 1:
            scores.append(1.0)
            continue
        H = -sum((c / n) * math.log(c / n + 1e-12) for c in counts.values())
        V_i = len(counts)
        normalised = 1.0 - H / (math.log(V_i) + 1e-12) if V_i > 1 else 1.0
        scores.append(max(0.0, normalised))
    return sum(scores) / K
