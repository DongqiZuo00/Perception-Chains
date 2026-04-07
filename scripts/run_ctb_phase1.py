"""CTB Phase 1: Cold start — generate and filter teacher trajectories."""

import argparse
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--output_dir", default="data/ctb_cache")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    from models.model import TeacherModel
    from training.ctb import phase1_cold_start
    from data.dataset import load_benchmark_data

    logger.info("Loading teacher model...")
    teacher = TeacherModel(args.teacher_model)

    # Load cold-start data (sample from training benchmarks)
    logger.info("Loading dataset...")
    benchmarks = ["ChartQA", "DocVQA", "RSVQA", "DIOR-RSVG",
                   "MuMuQA", "MMIU", "POPE", "HallusionBench"]
    per_bm = args.num_samples // len(benchmarks)
    dataset = []
    for bm in benchmarks:
        samples = load_benchmark_data(bm, args.data_dir, max_samples=per_bm)
        for s in samples:
            from PIL import Image
            dataset.append({
                "image": Image.open(s.image_path).convert("RGB"),
                "question": s.question,
                "answer": s.answer,
                "task": s.task,
            })

    logger.info(f"Phase 1 with {len(dataset)} samples, k={args.k}")
    cache = phase1_cold_start(
        teacher, dataset, k=args.k, tau=args.tau, top_p=args.top_p,
        max_samples=args.num_samples,
    )

    output_path = f"{args.output_dir}/phase1_cache.json"
    cache.save(output_path)
    logger.info(f"Saved {len(cache.trajectories)} trajectories to {output_path}")


if __name__ == "__main__":
    main()
