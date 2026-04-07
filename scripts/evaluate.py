"""
Evaluation script for Perception Chains.
Usage: python scripts/evaluate.py --model_path outputs/final --benchmarks seen --use_prl
"""

import argparse
import json
import logging
import os
import yaml
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/perception_chains.yaml")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--benchmarks", type=str, default="seen", choices=["seen", "unseen", "all"])
    parser.add_argument("--use_prl", action="store_true")
    parser.add_argument("--tau_prl", type=float, default=0.0)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model
    from models.model import PerceptionChainsModel
    logger.info(f"Loading model from {args.model_path}")
    model = PerceptionChainsModel(args.model_path, torch_dtype=torch.bfloat16)
    model.model.eval()

    # Select benchmarks
    if args.benchmarks == "seen":
        bm_list = config["evaluation"]["seen_benchmarks"]
    elif args.benchmarks == "unseen":
        bm_list = config["evaluation"]["unseen_benchmarks"]
    else:
        bm_list = config["evaluation"]["seen_benchmarks"] + config["evaluation"]["unseen_benchmarks"]

    # Load evaluation data
    from data.dataset import load_benchmark_data
    benchmark_data = {}
    for bm in bm_list:
        split = "test" if bm in config["evaluation"].get("unseen_benchmarks", []) else "val"
        samples = load_benchmark_data(bm, args.data_dir, split=split, max_samples=args.max_samples)
        if samples:
            benchmark_data[bm] = [
                {"image_path": s.image_path, "question": s.question,
                 "answer": s.answer, "task": s.task}
                for s in samples
            ]
            logger.info(f"  {bm}: {len(samples)} samples")
        else:
            logger.warning(f"  {bm}: no data found, skipping")

    # Evaluate
    from evaluation.evaluator import Evaluator
    prl_config = config.get("prl", {})
    prl_config["tau_prl"] = args.tau_prl

    evaluator = Evaluator(
        model=model,
        processor=model.processor,
        use_prl=args.use_prl,
        prl_config=prl_config,
    )

    results = evaluator.evaluate_all(benchmark_data)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for bm, metrics in results.items():
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  {bm:20s}: {metrics_str}")
    print("=" * 60)

    # Save
    output_file = args.output_file or os.path.join(
        os.path.dirname(args.model_path), "eval_results.json"
    )
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()
