import os

import hydra
import joblib
import torch
from loguru import logger
from tqdm import tqdm

from holomotion.src.training.lmdb_motion_lib import LmdbMotionLib


@hydra.main(
    config_path="../../config",
    config_name="misc/export_single_motion.yaml",
    version_base="1.1",
)
def main(cfg):
    cache_device = torch.device("cpu")
    ml = LmdbMotionLib(cfg.robot.motion, cache_device)
    motion_keys: list[str] = cfg.motion_keys
    if len(motion_keys) == 0:
        motion_keys = ml.all_motion_keys
    for motion_key in tqdm(motion_keys):
        motion_res = ml.export_motion_clip(motion_key)
        dump_dir = cfg.dump_dir
        os.makedirs(dump_dir, exist_ok=True)
        joblib.dump(
            {f"{motion_key}": motion_res},
            os.path.join(
                dump_dir,
                f"{motion_key}.pkl",
            ),
        )
        logger.info(f"Exported motion {motion_key} to {dump_dir}")


if __name__ == "__main__":
    main()
