"""
Fit W_proj via PCA on 1,000 calibration samples from the teacher.
W_proj ∈ R^{d_s × d_t} maps teacher hidden states to student space.
Frozen throughout training.
"""

import argparse
import torch
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--num_calibration", type=int, default=1000)
    parser.add_argument("--output_path", type=str, default="data/wproj.pt")
    parser.add_argument("--data_dir", type=str, default="data/calibration")
    args = parser.parse_args()

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from training.stability_loss import DimensionProjector

    logger.info("Loading teacher model...")
    teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.teacher_model, device_map="auto", torch_dtype=torch.bfloat16,
    )
    teacher.eval()
    d_t = teacher.config.hidden_size  # 7168

    logger.info("Loading student config for d_s...")
    from transformers import AutoConfig
    student_config = AutoConfig.from_pretrained(args.student_model)
    d_s = student_config.hidden_size  # 3584

    logger.info(f"d_s={d_s}, d_t={d_t}")

    # Collect hidden states from teacher on calibration data
    # Using random inputs as proxy (in practice, use real calibration samples)
    logger.info(f"Collecting {args.num_calibration} hidden states from teacher...")
    processor = AutoProcessor.from_pretrained(args.teacher_model)

    all_states = []
    for i in tqdm(range(args.num_calibration)):
        # Placeholder: in practice, load real images from calibration set
        dummy_input = torch.randint(0, 1000, (1, 64), device=teacher.device)
        attn_mask = torch.ones_like(dummy_input)
        with torch.no_grad():
            out = teacher(input_ids=dummy_input, attention_mask=attn_mask, output_hidden_states=True)
            # Take the middle layer's last-token hidden state
            mid_layer = len(out.hidden_states) // 2
            hs = out.hidden_states[mid_layer][0, -1].float().cpu()
            all_states.append(hs)

    states_matrix = torch.stack(all_states)  # [N, d_t]
    logger.info(f"Collected states: {states_matrix.shape}")

    # Fit PCA
    projector = DimensionProjector(d_s=d_s, d_t=d_t)
    projector.fit(states_matrix)
    logger.info("PCA fitted.")

    # Save
    import os
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(projector.state_dict(), args.output_path)
    logger.info(f"Saved W_proj to {args.output_path}")


if __name__ == "__main__":
    main()
