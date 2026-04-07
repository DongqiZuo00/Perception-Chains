"""CTB Phase 2: Self-bootstrap from student model."""

import argparse
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_ckpt", required=True)
    parser.add_argument("--teacher_model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--phase1_cache", default="data/ctb_cache/phase1_cache.json")
    parser.add_argument("--output_dir", default="data/ctb_cache")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--config", default="configs/perception_chains.yaml")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    from models.model import PerceptionChainsModel, TeacherModel
    from training.ctb import CTBCache, phase2_self_bootstrap
    from data.dataset import load_benchmark_data
    from PIL import Image

    logger.info("Loading student model...")
    student = PerceptionChainsModel(args.student_ckpt)

    logger.info("Loading teacher model...")
    teacher = TeacherModel(args.teacher_model)

    logger.info("Loading Phase 1 cache...")
    phase1_cache = CTBCache()
    phase1_cache.load(args.phase1_cache)

    # Load dataset
    benchmarks = config["evaluation"]["seen_benchmarks"]
    dataset = []
    for bm in benchmarks:
        samples = load_benchmark_data(bm, args.data_dir, max_samples=5000)
        for s in samples:
            dataset.append({
                "image": Image.open(s.image_path).convert("RGB"),
                "question": s.question,
                "answer": s.answer,
                "task": s.task,
            })

    ctb_cfg = config["ctb"]["phase2"]
    cache = phase2_self_bootstrap(
        student, teacher, dataset, phase1_cache,
        k=ctb_cfg["k_samples"], tau=ctb_cfg["tau_samp"],
        tau_conf=ctb_cfg["tau_conf"], rounds=args.rounds,
    )

    output_path = f"{args.output_dir}/phase2_cache.json"
    cache.save(output_path)
    logger.info(f"Saved {len(cache.trajectories)} trajectories to {output_path}")


if __name__ == "__main__":
    main()
