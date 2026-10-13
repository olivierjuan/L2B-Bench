"""Build the data/ strate directories and their signed manifests (R3.1/R3.2).

Usage::

    python scripts/build_strate_data.py \
        --miplib-s-from /path/to/branch-search-trees-dataset \
        --milp-l-from /path/to/miplib2017/collection

``--miplib-s-from`` points at the branch-search-trees dataset (Zarpellon et
al., AAAI 2021; https://osf.io/uvg5f/) whose ``train_instances/`` (19) and
``test_instances/`` (8) form the 27-instance MIPLIB-S strate with its original
split.

``--milp-l-from`` points at a directory containing MIPLIB 2017 instance files
(e.g. an extracted collection.zip from https://miplib.zib.de). The 73 MILP-L
instances (57 easy / 16 hard) and their easy/hard split follow the Mamba
evaluation (github.com/doctor-watson626/Mamba-Branching, logs_miplib_large).

The instance files stay on disk (gitignored); the signed manifests and their
provenance are committed.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gyozas.strates import build_strate

# MILP-L easy split (57 instances) — Mamba-Branching, logs_miplib_large/easy.
MILP_L_EASY = [
    "aflow40b",
    "app1-2",
    "bc1",
    "bell3a",
    "bell5",
    "biella1",
    "binkar10_1",
    "blend2",
    "dano3_5",
    "fast0507",
    "map10",
    "map18",
    "map20",
    "mik-250-20-75-4",
    "mine-166-5",
    "misc07",
    "n2seq36q",
    "neos-1200887",
    "neos-1215259",
    "neos-4722843-widden",
    "neos-4738912-atrato",
    "neos-480878",
    "neos-504674",
    "neos-504815",
    "neos-512201",
    "neos-584851",
    "neos-603073",
    "neos-612125",
    "neos-612162",
    "neos-662469",
    "neos-686190",
    "neos-801834",
    "neos-803219",
    "neos-807639",
    "neos-820879",
    "neos-829552",
    "neos-839859",
    "neos-892255",
    "neos-950242",
    "neos11",
    "neos12",
    "neos13",
    "neos18",
    "ns1208400",
    "ns1830653",
    "nu25-pr12",
    "nw04",
    "p0201",
    "pg",
    "pp08a",
    "rail507",
    "roll3000",
    "rout",
    "satellites1-25",
    "seymour1",
    "sp98ir",
    "unitcal_7",
]

# MILP-L hard split (16 instances) — Mamba-Branching, logs_miplib_large/hard.
MILP_L_HARD = [
    "atlanta-ip",
    "bab2",
    "bab5",
    "bab6",
    "harp2",
    "map16715-04",
    "msc98-ip",
    "mspp16",
    "n3seq24",
    "neos-4338804-snowy",
    "neos-4387871-tavua",
    "neos-4647030-tutaki",
    "nursesched-medium-hint03",
    "opm2-z10-s4",
    "pigeon-10",
    "radiationm40-10-02",
]

# Instances absent from the MIPLIB 2017 collection/benchmark zips (as of
# 2026-10) and therefore taken from the T-BranT dataset (Mendeley,
# doi 10.17632/8msnxmvdgp.1). bell3a, bell5, harp2 and pp08a are stored
# uncompressed there despite the .mps.gz naming.
MILP_L_MENDELEY = [
    "bell3a",
    "bell5",
    "harp2",
    "map20",
    "msc98-ip",
    "mspp16",
    "neos-1200887",
    "neos-1215259",
    "neos-504674",
    "neos-504815",
    "neos-512201",
    "neos-584851",
    "neos-603073",
    "neos-612125",
    "neos-612162",
    "neos-801834",
    "neos-803219",
    "neos-839859",
    "neos-892255",
    "neos11",
    "neos12",
    "neos13",
    "pg",
    "pp08a",
    "satellites1-25",
]


def find_instance(directory: Path, name: str) -> Path:
    for suffix in (".mps.gz", ".mps"):
        path = directory / f"{name}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"{name} (.mps.gz or .mps) not found in {directory}")


def build_miplib_s(source: Path, data_dir: Path) -> None:
    root = data_dir / "miplib_s"
    for split in ("train", "test"):
        (root / split).mkdir(parents=True, exist_ok=True)
        for path in sorted((source / f"{split}_instances").glob("*.mps.gz")):
            shutil.copy2(path, root / split / path.name)
    manifest = build_strate(
        root,
        provenance={
            "source": "branch-search-trees dataset, https://osf.io/uvg5f/",
            "paper": "Zarpellon, Jo, Lodi, Bengio — Parameterizing Branch-and-Bound "
            "Search Trees to Learn Branching Policies (AAAI 2021)",
            "splits": "original train/test split of the dataset (19 train / 8 test); "
            "the 8 test instances are the evaluation set of the TreeGate model",
            "format": "mps.gz",
        },
    )
    print(f"miplib_s: {len(manifest.entries)} instances, counts={manifest.counts}, hash={manifest.hash}")


def build_milp_l(source: Path, data_dir: Path) -> None:
    root = data_dir / "milp_l"
    for split, names in (("easy", MILP_L_EASY), ("hard", MILP_L_HARD)):
        (root / split).mkdir(parents=True, exist_ok=True)
        for name in names:
            src = find_instance(source, name)
            shutil.copy2(src, root / split / src.name)
    manifest = build_strate(
        root,
        provenance={
            "source": "MIPLIB 2017, https://miplib.zib.de (collection.zip and benchmark.zip); "
            "the instances listed in 'mendeley_instances' are absent from those zips and come "
            "from the T-BranT dataset (Mendeley, doi 10.17632/8msnxmvdgp.1)",
            "splits": "easy/hard split of the Mamba evaluation, "
            "github.com/doctor-watson626/Mamba-Branching (logs_miplib_large); "
            "57 easy / 16 hard",
            "format": "mps.gz (bell3a, bell5, harp2, pp08a are plain MPS)",
            "mendeley_instances": MILP_L_MENDELEY,
        },
    )
    print(f"milp_l: {len(manifest.entries)} instances, counts={manifest.counts}, hash={manifest.hash}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--miplib-s-from", type=Path, help="branch-search-trees dataset directory")
    parser.add_argument("--milp-l-from", type=Path, help="directory with MIPLIB 2017 instance files")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    if args.miplib_s_from:
        build_miplib_s(args.miplib_s_from, args.data_dir)
    if args.milp_l_from:
        build_milp_l(args.milp_l_from, args.data_dir)


if __name__ == "__main__":
    main()
