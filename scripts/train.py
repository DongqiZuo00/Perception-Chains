"""
Main training script for Perception Chains.
Usage: torchrun --nproc_per_node=8 scripts/train.py --config configs/perception_chains.yaml
"""

import argparse
import logging
import os
import yaml
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/perception_chains.yaml")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--ctb_cache", type=str, default="data/ctb_cache/phase1_cache.json")
    parser.add_argument("--wproj_path", type=str, default="data/wproj.pt")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Load model ----
    from models.model import PerceptionChainsModel
    logger.info(f"Loading student model: {config['model']['student']}")
    model = PerceptionChainsModel(
        model_name=config["model"]["student"],
        torch_dtype=torch.bfloat16,
    )
    if config["training"]["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()

    # ---- Load W_proj ----
    from training.stability_loss import DimensionProjector
    wproj = DimensionProjector(d_s=config["model"]["d_s"], d_t=config["model"]["d_t"])
    if os.path.exists(args.wproj_path):
        wproj.load_state_dict(torch.load(args.wproj_path, map_location="cpu"))
        logger.info(f"Loaded W_proj from {args.wproj_path}")
    else:
        logger.warning(f"W_proj not found at {args.wproj_path}; using random init (fit it first!)")

    # ---- Load CTB cache ----
    from training.ctb import CTBCache
    teacher_cache = CTBCache()
    if os.path.exists(args.ctb_cache):
        teacher_cache.load(args.ctb_cache)
        logger.info(f"Loaded {len(teacher_cache.trajectories)} cached trajectories")
    else:
        logger.warning(f"CTB cache not found at {args.ctb_cache}; training without stability targets")

    # ---- Build dataset ----
    from data.dataset import build_balanced_mixture, PerceptionChainsDataset
    logger.info("Building balanced training mixture...")
    seen_benchmarks = config["evaluation"]["seen_benchmarks"]
    samples = build_balanced_mixture(
        benchmarks=seen_benchmarks,
        data_dir=args.data_dir,
        per_benchmark_cap=config["training"]["per_benchmark_cap"],
        seed=args.seed,
    )

    # Attach reference trajectories from CTB cache where available
    cache_lookup = {t.sample_id: t for t in teacher_cache.trajectories}
    for i, s in enumerate(samples):
        traj = cache_lookup.get(str(i))
        if traj and traj.chain.answer:
            # Reconstruct the reference chain text (simplified serialisation)
            chain_text = _serialize_chain(traj.chain)
            s.reference_chain_text = chain_text

    train_dataset = PerceptionChainsDataset(samples, model.processor)
    logger.info(f"Training dataset: {len(train_dataset)} samples")

    # ---- Train ----
    from training.trainer import PerceptionChainsTrainer
    trainer = PerceptionChainsTrainer(
        model=model,
        train_dataset=train_dataset,
        teacher_cache=teacher_cache,
        wproj=wproj,
        config=config,
        output_dir=args.output_dir,
    )
    trainer.train()


def _serialize_chain(chain) -> str:
    """Convert an AnchorChain to the text format expected by the model."""
    from models.ave import SLOT_SCHEMAS
    lines = []
    for a in chain.anchors:
        bbox_str = f"[{int(a.bbox[0])}, {int(a.bbox[1])}, {int(a.bbox[2])}, {int(a.bbox[3])}]"
        lines.append(
            f"Slot {a.slot_idx + 1} ({a.slot}): <box>{bbox_str}</box> {a.attribute}"
        )
    if chain.answer:
        lines.append(f"Final Answer: {chain.answer}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
