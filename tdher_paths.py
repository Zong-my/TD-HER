"""Paths for the source-only TD-HER distribution.
All generated files belong to an external workspace, never to this checkout.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
WORK_ROOT = Path(os.environ.get("TDHER_WORKDIR", str(REPO_ROOT.parent / "TD-HER-work"))).expanduser().resolve()
if WORK_ROOT == REPO_ROOT or REPO_ROOT in WORK_ROOT.parents:
    raise ValueError("TDHER_WORKDIR must be outside the source checkout")
DATA_ROOT = Path(os.environ.get("TDHER_DATA_ROOT", str(WORK_ROOT / "data"))).expanduser().resolve()
if DATA_ROOT == REPO_ROOT or REPO_ROOT in DATA_ROOT.parents:
    raise ValueError("TDHER_DATA_ROOT must be outside the source checkout")
RESULTS_ROOT = WORK_ROOT / "results"

def data_path(relative=""):
    aliases = {"ieee39_v8_80_10_10": "ieee39", "ieee300_v2_posttrigger": "ieee300",
               "ieee39_v8": "raw/ieee39", "ieee300_v2": "raw/ieee300"}
    parts = str(relative).split("/")
    if len(parts) > 1 and parts[1].startswith("adjacency") and parts[0] in ("ieee39_v8", "ieee300_v2"):
        parts[0] = "ieee39" if parts[0] == "ieee39_v8" else "ieee300"
        return DATA_ROOT.joinpath(*parts)
    if parts and parts[0] in aliases:
        parts[0] = aliases[parts[0]]
    return DATA_ROOT.joinpath(*parts)

def work_path(relative=""):
    relative = str(relative)
    if relative.startswith("TD-HER/data/"):
        return data_path(relative[len("TD-HER/data/"):])
    if relative.startswith("data/"):
        return data_path(relative[5:])
    return WORK_ROOT / relative

def external_output(path):
    path = Path(path).expanduser().resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise ValueError("Output path must be outside the source checkout: " + str(path))
    return path
