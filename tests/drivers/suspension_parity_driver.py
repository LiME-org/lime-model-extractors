#!/usr/bin/env python3
"""Emit Python reference suspension models at successive job prefixes."""

import json
import random
import subprocess
import sys
from pathlib import Path

from rt_model_inference import (
    basic_from_segmented_observation,
    infer_bos_suspension_model,
    infer_max_suspension_model,
    infer_min_suspension_model,
    infer_segmented_suspension_model,
)


def random_segment(rng: random.Random) -> tuple[int, int]:
    suspension = rng.randrange(101)
    return suspension, rng.randint(1 if suspension == 0 else 0, 100)


def main() -> None:
    reference_dir = Path(sys.argv[1])
    basic_cases = []
    segmented_cases = []
    bos_cases = []
    for seed in ("Self-suspensions", "BOS", "streaming"):
        rng = random.Random(seed)
        basic = [(rng.randrange(11), rng.randrange(101)) for _ in range(129)]
        jobs = [
            [random_segment(rng) for _ in range(rng.randrange(1, 9))]
            for _ in range(129)
        ]
        for nmax in (None, 1, 7, 32):
            basic_cases.append(
                {
                    "seed": seed,
                    "observations": basic,
                    "nmax": nmax,
                    "snapshots": [
                        {
                            "end": end,
                            "maximum": infer_max_suspension_model(basic[:end], nmax),
                            "minimum": infer_min_suspension_model(basic[:end], nmax),
                        }
                        for end in (0, 1, 7, 31, len(basic))
                    ],
                }
            )
        for max_segments in (None, 4, 8):
            segmented_cases.append(
                {
                    "seed": seed,
                    "observations": jobs,
                    "max_segments": max_segments,
                    "basic": [basic_from_segmented_observation(job) for job in jobs],
                    "snapshots": [
                        {
                            "end": end,
                            "model": infer_segmented_suspension_model(
                                jobs[:end], max_segments
                            ),
                        }
                        for end in (0, 1, 7, 31, len(jobs))
                    ],
                }
            )
            for max_models in (None, 1, 2, 5):
                bos_cases.append(
                    {
                        "seed": seed,
                        "observations": jobs,
                        "max_segments": max_segments,
                        "max_models": max_models,
                        "snapshots": [
                            {
                                "end": end,
                                "model": infer_bos_suspension_model(
                                    jobs[:end], max_segments, max_models
                                ),
                            }
                            for end in (0, 1, 7, 31, len(jobs))
                        ],
                    }
                )
    revision = subprocess.check_output(
        ["git", "-C", str(reference_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        json.dumps(
            {
                "revision": revision,
                "basic": basic_cases,
                "segmented": segmented_cases,
                "bos": bos_cases,
            }
        )
    )


if __name__ == "__main__":
    main()
