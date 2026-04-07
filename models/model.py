"""
Model wrapper around Qwen2.5-VL for Perception Chains.
Provides hooks to extract hidden states at anchor positions for the stability loss.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from dataclasses import dataclass


@dataclass
class AnchorHiddenStates:
    """Hidden states collected at anchor positions across layers."""
    # shape: [num_anchors, num_active_layers, hidden_dim]
    states: torch.Tensor
    # which layers are active for each anchor
    active_layers: List[List[int]]
    # token positions of the first region-coordinate token per anchor
    anchor_positions: List[int]


class PerceptionChainsModel(nn.Module):
    """Wraps Qwen2.5-VL-7B with hidden-state hooks for stability loss."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map: str = "auto",
        torch_dtype=torch.bfloat16,
    ):
        super().__init__()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, device_map=device_map, torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size  # 3584 for 7B
        self.num_layers = self.config.num_hidden_layers

        # Storage for hooked hidden states
        self._anchor_hooks: List = []
        self._captured_states: Dict[int, Dict[int, torch.Tensor]] = {}

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def enable_gradient_checkpointing(self):
        self.model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    # Hidden-state capture at anchor positions
    # ------------------------------------------------------------------
    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # output[0] is hidden states: [batch, seq, hidden]
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            self._captured_states[layer_idx] = hs.detach()
        return hook_fn

    def register_anchor_hooks(self, layer_indices: Optional[List[int]] = None):
        """Register forward hooks on specified layers (or all)."""
        self.remove_anchor_hooks()
        if layer_indices is None:
            layer_indices = list(range(self.num_layers))
        for idx in layer_indices:
            layer = self.model.model.layers[idx]
            h = layer.register_forward_hook(self._make_hook(idx))
            self._anchor_hooks.append(h)

    def remove_anchor_hooks(self):
        for h in self._anchor_hooks:
            h.remove()
        self._anchor_hooks.clear()
        self._captured_states.clear()

    def get_anchor_hidden_states(
        self,
        anchor_token_positions: List[int],
        active_layer_map: Dict[int, List[int]],
    ) -> Dict[int, Dict[int, torch.Tensor]]:
        """Retrieve captured hidden states at anchor positions.
        
        Returns:
            {anchor_idx: {layer_idx: hidden_state [hidden_dim]}}
        """
        result = {}
        for anchor_idx, pos in enumerate(anchor_token_positions):
            result[anchor_idx] = {}
            layers = active_layer_map.get(anchor_idx, [])
            for l in layers:
                if l in self._captured_states:
                    # [batch=0, pos, hidden]
                    result[anchor_idx][l] = self._captured_states[l][0, pos]
        return result

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)


class TeacherModel:
    """Frozen 72B teacher for CTB trajectory generation."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        device_map: str = "auto",
        torch_dtype=torch.bfloat16,
    ):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, device_map=device_map, torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size  # 7168 for 72B

    @torch.no_grad()
    def generate_chains(
        self,
        inputs: dict,
        k: int = 8,
        tau: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 1024,
    ) -> List[str]:
        """Generate k candidate anchor chains via sampling."""
        outputs = []
        for _ in range(k):
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=tau,
                top_p=top_p,
            )
            text = self.processor.batch_decode(
                gen_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )[0]
            outputs.append(text)
        return outputs

    @torch.no_grad()
    def extract_hidden_states(
        self, inputs: dict, token_positions: List[int]
    ) -> Dict[int, torch.Tensor]:
        """Extract hidden states at given positions across all layers."""
        out = self.model(
            **inputs, output_hidden_states=True,
        )
        # out.hidden_states: tuple of [batch, seq, hidden] per layer
        result = {}
        for l, hs in enumerate(out.hidden_states):
            for pos in token_positions:
                if l not in result:
                    result[l] = {}
                result[l][pos] = hs[0, pos].cpu()
        return result
